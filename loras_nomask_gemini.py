# LoRA Slider Training (Txt2Img Style) & Inference (Img2Img Style - Callable Multi-Slider)
# ========================================================================================
# Author: ChatGPT (May 2025) - Adapted for callable multi-slider inference
#
# This single Python module provides **both** a training routine (text-to-image style)
# to learn a LoRA adapter that behaves like a *continuous attribute slider*, **and** an
# inference routine (image-to-image style) that can be called as a Python function
# with a list of slider values to apply that effect to an existing input image and
# get back a list of generated images. The CLI supports single slider inference.
#
# Why a single file? Copy‑paste it, run `python script.py train ‑‑help` for training,
# `python script.py infer ‑‑help` for single CLI inference, or call `run_inference`
# from Python.
#
# The code is written against **diffusers ≥ 0.26.0**, **peft ≥ 0.10.0**,
# **torch ≥ 2.1**, **accelerate**, **safetensors**, and **transformers**.
# It supports SD 1.5/2.x *and* SDXL models – pass the correct
# `--pretrained_model_name_or_path`.
#
# ────────────────────────────────────────────────────────────────────────────
# DATA LAYOUT (Training - No Masks Needed)
# ───────────
# A minimal dataset directory looks like this::
#
#     dataset/
#       positive/           # images that *add* the attribute (slider +1)
#         0001.png
#         0002.png …
#       negative/           # images that *remove* the attribute (slider ‑1)
#         0001.png
#         0002.png …
#
# Prompts (optional) → `0001.txt` beside each image, otherwise a *global*
# `--prompt` is used during training.
#
# During training the script loads *paired* samples (positive[i],
# negative[i]) so that gradients for both ends of the slider are computed
# *within the same batch*.
#
# ────────────────────────────────────────────────────────────────────────────
# CLI USAGE (BASIC)
# ────────────────
# • Training (Same as Txt2Img)::
#
#     python lora_slider_img2img_callable.py train \
#         --pretrained_model runwayml/stable-diffusion-v1-5 \
#         --dataset_dir /path/to/your/dataset \
#         --output_dir  /path/to/your/checkpoints/slider \
#         --resolution 512 --batch_size 4 --rank 8 --alpha 16 \
#         --max_steps 10000 --save_every 1000
#
# • Inference (Image-to-Image via CLI - single slider)::
#
#     python lora_slider_img2img_callable.py infer \
#         --pretrained_model runwayml/stable-diffusion-v1-5 \
#         --adapter_path  /path/to/your/checkpoints/slider/adapter-final.safetensors \
#         --input_image   path/to/your/input.png \
#         --prompt "a portrait of a person" \
#         --slider 0.7    # single slider value for the learned attribute
#         --strength 0.75 # How much to change the input image
#         --output_image  out.png
#
# • Inference (Image-to-Image via Python Function - multi-slider)::
#
#     from lora_slider_img2img_callable import run_inference
#     from PIL import Image
#
#     slider_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
#     output_images = run_inference(
#         pretrained_model="runwayml/stable-diffusion-v1-5",
#         adapter_path="/path/to/your/checkpoints/slider/adapter-final.safetensors",
#         input_image="/path/to/your/input.png",
#         prompt="a portrait of a person",
#         slider=slider_values, # Pass a list of values
#         strength=0.8,
#         num_inference_steps=50,
#         seed=1234,
#         # output_image parameter is ignored when slider is a list
#     )
#
#     # output_images is a list of PIL Image objects, one for each slider value
#     for i, img in enumerate(output_images):
#         img.save(f"output_{slider_values[i]:.2f}.png") # Save each image
#
# ────────────────────────────────────────────────────────────────────────────

import argparse
import itertools
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
import transformers

# ---------------------------
#  UTILITY FUNCTIONS
# ---------------------------

def load_image(path: Path, size: Optional[int] = None) -> Image.Image:
    """Loads an image, optionally resizing it."""
    img = Image.open(path).convert("RGB")
    if size is not None and size > 0:
        # Use LANCZOS filter for high-quality downsampling
        img = img.resize((size, size), Image.LANCZOS)
    return img

# PairedImageDataset remains the same as the Text-to-Image version
class PairedImageDataset(Dataset):
    """Returns ((pos_img, pos_prompt), (neg_img, neg_prompt))"""
    def __init__(self, root: str, resolution: int = 512, default_prompt: str = ""):
        self.root = Path(root)
        self.pos_img_paths = sorted([f for f in (self.root / "positive").glob("*.png")])
        self.neg_img_paths = sorted([f for f in (self.root / "negative").glob("*.png")])
        if len(self.pos_img_paths) != len(self.neg_img_paths):
             raise ValueError(f"Mismatch in positive ({len(self.pos_img_paths)})/negative ({len(self.neg_img_paths)}) image counts.")
        if not self.pos_img_paths:
             raise ValueError(f"No images found in {root}/positive or {root}/negative.")
        self.resolution = resolution
        self.default_prompt = default_prompt
        self.pos_files = self._precompute_paths(self.pos_img_paths)
        self.neg_files = self._precompute_paths(self.neg_img_paths)
        self.img_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS), # Ensure exact size for training
            transforms.CenterCrop(resolution), # Crop if not square after resize
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _precompute_paths(self, img_paths: List[Path]):
        return [{"img": img_path, "txt": img_path.with_suffix(".txt")} for img_path in img_paths]

    def __len__(self):
        return len(self.pos_files)

    def _read_pair(self, file_info: dict):
        img_pil = Image.open(file_info["img"]).convert("RGB")
        img_tensor = self.img_transforms(img_pil)
        prompt = self.default_prompt
        if file_info["txt"].exists():
             try: prompt = file_info["txt"].read_text().strip()
             except Exception as e: print(f"Warn: Read failed {file_info['txt']}: {e}")
        return img_tensor, prompt

    def __getitem__(self, idx):
        return self._read_pair(self.pos_files[idx]), self._read_pair(self.neg_files[idx])


# ---------------------------
#  TRAINING (Same as previous version)
# ---------------------------
def train_slider(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.mixed_precision else "no",
    )
    device = accelerator.device
    set_seed(args.seed)

    print("🧩 Loading base pipeline for training components...")
    base_pipe_cls = StableDiffusionXLPipeline if "xl" in args.pretrained_model.lower() else StableDiffusionPipeline
    try:
        pipe = base_pipe_cls.from_pretrained(args.pretrained_model, torch_dtype=torch.float16 if args.mixed_precision else torch.float32)
    except EnvironmentError as e:
        print(f"Error loading model '{args.pretrained_model}': {e}")
        sys.exit(1)

    pipe.vae.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32).requires_grad_(False)
    pipe.text_encoder.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32).requires_grad_(False)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32).requires_grad_(False)
        print("🧊 Freezing Text Encoder 2...")
    pipe.unet.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32)

    print("🧊 Freezing VAE and Text Encoder(s)...")
    print("🚀 Adding LoRA adapter to UNet...")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Adjust if needed
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    if accelerator.is_main_process:
        pipe.unet.print_trainable_parameters()

    print("💾 Loading Dataset...")
    try:
        dataset = PairedImageDataset(args.dataset_dir, args.resolution, args.prompt)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("⚙️ Setting up Optimizer...")
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.learning_rate)

    print("✨ Preparing with Accelerator...")
    unet, optimizer, dataloader = accelerator.prepare(pipe.unet, optimizer, dataloader)

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    text_encoder_2 = pipe.text_encoder_2 if hasattr(pipe, "text_encoder_2") else None
    tokenizer_2 = pipe.tokenizer_2 if hasattr(pipe, "tokenizer_2") else None

    unet_dtype = unet.dtype
    vae_dtype = vae.dtype

    def encode_latents(img_batch):
        img_batch = img_batch.to(device=device, dtype=vae_dtype)
        with torch.no_grad():
            latents = vae.encode(img_batch).latent_dist.sample() * vae.config.scaling_factor
        return latents

    def encode_prompts(prompts_list):
        text_inputs = tokenizer(prompts_list, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids)[0]
        add_text_embeds = None
        if text_encoder_2:
            text_inputs_2 = tokenizer_2(prompts_list, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids_2 = text_inputs_2.input_ids.to(device)
            with torch.no_grad():
                prompt_embeds_2 = text_encoder_2(text_input_ids_2)[0]
                pooled_prompt_embeds = text_encoder_2(text_input_ids_2)[1]
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
            add_text_embeds = pooled_prompt_embeds
        return prompt_embeds.to(unet_dtype), add_text_embeds.to(unet_dtype) if add_text_embeds is not None else None

    print("🔥 Starting Training Loop...")
    global_step = 0
    loss_vector = []
    for epoch in itertools.count():
        unet.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                (img_p, prompt_p), (img_n, prompt_n) = batch
                bsz = img_p.shape[0]

                latents_p = encode_latents(img_p)
                latents_n = encode_latents(img_n)
                noise = torch.randn_like(latents_p)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents_p = scheduler.add_noise(latents_p, noise, timesteps)
                noisy_latents_n = scheduler.add_noise(latents_n, noise, timesteps)

                if isinstance(prompt_p, str): prompt_p = [prompt_p] * bsz
                if isinstance(prompt_n, str): prompt_n = [prompt_n] * bsz
                prompt_embeds_p, add_text_embeds_p = encode_prompts(prompt_p)
                prompt_embeds_n, add_text_embeds_n = encode_prompts(prompt_n)

                latent_model_input_p = noisy_latents_p.to(unet_dtype)
                latent_model_input_n = noisy_latents_n.to(unet_dtype)

                add_time_ids_p = add_time_ids_n = None
                unet_added_conditions_p = {}
                unet_added_conditions_n = {}
                if text_encoder_2 is not None: # SDXL
                    # Using placeholder zeros for crops, ADAPT if needed
                    add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]] * bsz, device=device)
                    unet_added_conditions_p = {"text_embeds": add_text_embeds_p.to(unet_dtype), "time_ids": add_time_ids.to(unet_dtype)}
                    unet_added_conditions_n = {"text_embeds": add_text_embeds_n.to(unet_dtype), "time_ids": add_time_ids.to(unet_dtype)}


                noise_pred_p = unet(latent_model_input_p, timesteps, encoder_hidden_states=prompt_embeds_p,
                                    cross_attention_kwargs={"scale": 1.0}, added_cond_kwargs=unet_added_conditions_p).sample
                noise_pred_n = unet(latent_model_input_n, timesteps, encoder_hidden_states=prompt_embeds_n,
                                    cross_attention_kwargs={"scale": -1.0}, added_cond_kwargs=unet_added_conditions_n).sample

                loss_p = torch.nn.functional.mse_loss(noise_pred_p.float(), noise.float())
                loss_n = torch.nn.functional.mse_loss(noise_pred_n.float(), noise.float())
                loss = (loss_p + loss_n) / 2.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                         accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                loss_vector.append(loss.item())
                if accelerator.is_main_process:
                    if global_step % args.log_every == 0: print(f"E{epoch} S{global_step} Loss*1k:{loss.item()*1000:.4f}(P*1k:{loss_p.item()*1000:.4f} N*1k:{loss_n.item()*1000:.4f})")
                    if global_step > 0 and global_step % args.save_every == 0: save_progress(unet, accelerator, args.output_dir, global_step)
                if global_step >= args.max_steps: break
        if global_step >= args.max_steps: break

    print("🏁 Training Finished.")
    if accelerator.is_main_process:
        save_progress(unet, accelerator, args.output_dir, "final")


def save_progress(unet: nn.Module, accelerator: Accelerator, out_dir: str, step_or_name):
    """Saves the LoRA adapter state using PEFT utility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"adapter-{step_or_name}.safetensors"
    unwrapped_unet = accelerator.unwrap_model(unet)
    try:
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        if not lora_state_dict:
            print(f"W: No PEFT params found at step {step_or_name}.")
            return
        save_file(lora_state_dict, str(filename))
        print(f"💾 Saved LoRA adapter checkpoint → {filename}")
    except Exception as e:
        print(f"E: Error saving LoRA adapter state_dict: {e}")


# ---------------------------
#  INFERENCE (Image-to-Image - Callable Function)
# ---------------------------

# Define the callable function
def run_inference(
    pretrained_model: str,
    adapter_path: str,
    input_image: Union[str, Path, Image.Image], # Accept path or PIL Image
    prompt: str,
    slider: Union[float, List[float], Tuple[float, ...]] = 0.0, # Accept float or list/tuple
    output_image: Optional[Union[str, Path]] = None, # Optional output path (ignored for list of sliders)
    negative_prompt: Optional[str] = None,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None, # Allow overriding device
    dtype: Optional[torch.dtype] = None, # Allow overriding dtype
) -> Union[Image.Image, List[Image.Image]]: # Return single image or list of images
    """
    Runs Stable Diffusion Image-to-Image inference with a LoRA slider.

    Args:
        pretrained_model: Base model ID (Hugging Face Hub) or path.
        adapter_path: Path to the trained LoRA adapter (.safetensors file).
        input_image: Path to the input image file or a PIL Image object.
        prompt: Text prompt guiding the transformation.
        slider: LoRA scale slider value(s) in [-1, 1]. Can be a single float
                or a list/tuple of floats. Defaults to 0.0 (neutral).
        output_image: Optional path to save the *single* output image (if
                      a single slider value is provided). Ignored if 'slider'
                      is a list/tuple.
        negative_prompt: Optional negative prompt.
        strength: Img2Img strength (0-1). Higher values allow more deviation
                  from input image.
        guidance_scale: Classifier-Free Guidance scale.
        num_inference_steps: Number of diffusion inference steps.
        seed: Random seed for reproducibility.
        device: Optional device to use (e.g., "cuda", "cpu"). Defaults to
                "cuda" if available.
        dtype: Optional torch dtype for the pipeline (e.g., torch.float16).
               Defaults to float16 on CUDA, float32 otherwise.

    Returns:
        A single PIL Image object if a single slider value is provided,
        otherwise a list of PIL Image objects, one for each slider value.
    """
    # Ensure slider is always treated as a list for internal loop
    is_single_slider = not isinstance(slider, (list, tuple))
    slider_values = [slider] if is_single_slider else list(slider)

    if not slider_values:
         print("Warning: No slider values provided. Returning empty list.")
         return [] if not is_single_slider else None # Return appropriate empty type

    # Determine device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Determine dtype if not specified
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # Use bfloat16 on CPU if supported and mixed precision might be intended
        if device.type == "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
             dtype = torch.bfloat16

    print(f"🚀 Loading Img2Img pipeline {pretrained_model} on {device} with {dtype}...")
    pipe_cls = StableDiffusionXLImg2ImgPipeline if "xl" in pretrained_model.lower() else StableDiffusionImg2ImgPipeline
    try:
        pipe = pipe_cls.from_pretrained(pretrained_model, torch_dtype=dtype)
    except EnvironmentError as e:
        print(f"Error loading model '{pretrained_model}': {e}")
        raise # Re-raise the error

    print(f"🚀 Loading LoRA adapter {adapter_path}...")
    try:
        pipe.load_lora_weights(adapter_path)
        print("✅ LoRA Adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA adapter from {adapter_path}: {e}")
        raise # Re-raise the error

    pipe.to(device)
    # Progress bar might be noisy for multi-slider calls, disable by default in callable
    pipe.set_progress_bar_config(disable=True)

    # Load or use the input image
    if isinstance(input_image, (str, Path)):
        print(f"🖼️ Loading input image from {input_image}...")
        try:
            input_image_pil = load_image(Path(input_image), size=None) # Load original size
            print(f"   Input image loaded (Size: {input_image_pil.size})")
        except FileNotFoundError:
            raise FileNotFoundError(f"Input image not found at {input_image}")
        except Exception as e:
            raise RuntimeError(f"Error loading input image {input_image}: {e}")
    elif isinstance(input_image, Image.Image):
        print("🖼️ Using provided PIL Image as input.")
        input_image_pil = input_image
    else:
        raise TypeError(f"input_image must be a path (str/Path) or a PIL Image, not {type(input_image)}")

    # Determine autocast device type
    autocast_device = device.type if device.type != "mps" else "cpu" # MPS doesn't support autocast

    results = []
    # Iterate through each slider value provided
    for i, current_slider_value in enumerate(slider_values):
        print(f"⏳ Generating image for slider value: {current_slider_value:.2f} ({i+1}/{len(slider_values)})")
        # Use a different seed for each image *if* desired for variation,
        # or use the same seed for consistency across slider values.
        # Using same seed + generator for now for clear comparison of slider effect.
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            with torch.autocast(autocast_device, dtype=dtype):
                result = pipe(
                    prompt=prompt,
                    image=input_image_pil,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    generator=generator, # Use generator created above
                    cross_attention_kwargs={"scale": current_slider_value}, # Pass the current slider value
                    num_images_per_prompt=1,
                ).images[0]

            results.append(result) # Add the resulting image to the list

        except Exception as e:
            print(f"😭 Inference failed for slider value {current_slider_value}: {e}")
            # Decide how to handle failure for one value in a list.
            # Option 1: Raise immediately (stops all remaining).
            # Option 2: Print warning, add None to results, continue.
            # Option 3: Print warning, skip this value, continue.
            # Option 2 (adding None) is safer for callable API, lets caller see which failed.
            results.append(None)
            import traceback
            traceback.print_exc() # Print traceback for failed item


    # --- Handle saving for single output only (CLI case) ---
    # The CLI wrapper will call this function with a list containing a single slider value.
    # We only save automatically if *originally* a single slider value was requested
    # AND an output_image path was provided.
    if is_single_slider and output_image:
         if len(results) > 0 and results[0] is not None:
             try:
                 output_path = Path(output_image)
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 results[0].save(output_path)
                 print(f"✅ Saved result → {output_path}")
             except Exception as e:
                 print(f"Error saving output image {output_image}: {e}")
         else:
             print("Warning: No image generated to save.")
    # -------------------------------------------------------

    # Return the results list (or the single image if only one was requested)
    return results[0] if is_single_slider and results else results # Return single image or list


# ---------------------------
#  CLI ENTRY
# ---------------------------

def build_arg_parser():
    # --- Defaults ---
    default_model = "/workspace/models/sd1_5_inpaint"
    default_rank = 8
    default_alpha = 1 #16
    default_batch_size = 4
    default_lr = 1e-4
    default_max_steps = 4000
    default_save_every = 500
    default_resolution = 512 # Training resolution
    default_prompt = "hairline slider control"

    # --- Placeholder paths (USER MUST PROVIDE THESE or change defaults) ---
    is_single_folder = True
    if is_single_folder:
        default_dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single"
    else:
        default_dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai"
    default_output_dir = f"/workspace/my_sliders/models/img2img_slider_rank{default_rank}_alpha{default_alpha}"
    if is_single_folder:
        default_output_dir += "_single"
    default_adapter_path = default_output_dir + "/adapter-final.safetensors"
    default_infer_input_image = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single/neutral/updated_caption_image.png"
    default_infer_prompt = default_prompt

    parser = argparse.ArgumentParser(description="LoRA Slider Trainer (Txt2Img Style) / Inference (Img2Img Style)")
    sub = parser.add_subparsers(dest="command", required=True, help="Choose 'train' or 'infer'")

    # —— Train Arguments (Txt2Img Style) ——
    train = sub.add_parser("train", help="Train a LoRA slider adapter using image pairs")
    train.add_argument("--pretrained_model", default=default_model, help="Base model ID (HF Hub) or path.")
    train.add_argument("--dataset_dir", required=True, default=default_dataset_dir, help="Path to dataset directory with 'positive/' and 'negative/' image subfolders.")
    train.add_argument("--output_dir", required=True, default=default_output_dir, help="Directory to save LoRA checkpoints.")
    train.add_argument("--resolution", type=int, default=default_resolution, help="Resolution to resize images to for training.")
    train.add_argument("--prompt", default=default_prompt, help="Default prompt if no '.txt' files found.")
    train.add_argument("--rank", type=int, default=default_rank, help="LoRA rank.")
    train.add_argument("--alpha", type=int, default=default_alpha, help="LoRA alpha.")
    train.add_argument("--batch_size", type=int, default=default_batch_size, help="Training batch size (per GPU).")
    train.add_argument("--learning_rate", type=float, default=default_lr, help="Optimizer learning rate.")
    train.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    train.add_argument("--max_steps", type=int, default=default_max_steps, help="Total training steps.")
    train.add_argument("--save_every", type=int, default=default_save_every, help="Save checkpoint every N steps.")
    train.add_argument("--log_every", type=int, default=50, help="Log loss every N steps.")
    train.add_argument("--mixed_precision", action="store_true", help="Use FP16 mixed precision.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")
    train.add_argument("--num_workers", type=int, default=4, help="Dataloader worker processes.")
    train.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")

    # —— Infer Arguments (Img2Img Style - for CLI use) ——
    # These map to the parameters of the run_inference function, but CLI takes single slider
    infer = sub.add_parser("infer", help="Run image-to-image inference via command line")
    infer.add_argument("--pretrained_model", default=default_model, help="Base model ID (HF Hub) or path.")
    infer.add_argument("--adapter_path", required=True, default=default_adapter_path, help="Path to the trained LoRA adapter (.safetensors).")
    infer.add_argument("--input_image", required=True, default=default_infer_input_image, help="Path to the input image.")
    infer.add_argument("--prompt", required=True, default=default_infer_prompt, help="Text prompt guiding the transformation.")
    infer.add_argument("--output_image", default="result.png", help="Path to save the output image.") # Optional in function, default here
    infer.add_argument("--negative_prompt", default=None, help="Optional negative prompt.")
    infer.add_argument("--strength", type=float, default=0.75, help="Img2Img strength (0-1). Higher values allow more deviation.")
    infer.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale.")
    infer.add_argument("--num_steps", type=int, default=30, help="Number of inference steps.")
    infer.add_argument("--seed", type=int, default=42, help="Random seed.")
    infer.add_argument("--slider", type=float, default=0.0, help="LoRA scale slider value in [-1, 1].")
    # Note: CLI inference only supports a single slider value directly.
    # To run multiple slider values via CLI, you'd write a small shell/Python script
    # that loops and calls this script repeatedly with different --slider values.

    return parser

# Keep the _run_inference_cli wrapper function to handle CLI args -> callable function
def _run_inference_cli(args):
    """Wrapper function to call the main run_inference from CLI args."""
    # Call the callable function with the single slider value wrapped in a list
    # This ensures the callable function's multi-slider logic is used,
    # but since it's a single value, it will return a single image.
    # The output_image path is handled by the callable function itself
    # when it detects a single slider value was originally requested.
    run_inference(
        pretrained_model=args.pretrained_model,
        adapter_path=args.adapter_path,
        input_image=args.input_image, # Path from CLI
        prompt=args.prompt,
        slider=args.slider, # Pass as a single float for the callable's logic
        output_image=args.output_image, # Pass path to save
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        # Device and dtype are defaulted in the callable function if not passed
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        if not args.dataset_dir or args.dataset_dir == build_arg_parser().get_default("dataset_dir"):
             parser.error("The --dataset_dir argument is required for training.")
        if not args.output_dir or args.output_dir == build_arg_parser().get_default("output_dir"):
             parser.error("The --output_dir argument is required for training.")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_slider(args)
    elif args.command == "infer":
        if not args.adapter_path or args.adapter_path == build_arg_parser().get_default("adapter_path"):
             parser.error("The --adapter_path argument is required for inference.")
        if not args.input_image or args.input_image == build_arg_parser().get_default("infer_input_image"):
             parser.error("The --input_image argument is required for inference.")
        if not args.prompt or args.prompt == build_arg_parser().get_default("infer_prompt"):
             parser.error("The --prompt argument is required for inference.")
        _run_inference_cli(args) # Call the wrapper for CLI mode
    else:
        parser.print_help()