# LoRA Slider Training & Inference for Stable Diffusion Inpainting
# ================================================================
# Author: ChatGPT (May 2025) - Modified through collaborative debugging
#
# This single Python module provides **both** a training routine to learn a
# LoRA adapter that behaves like a *continuous attribute slider* **and** an
# inference routine that lets you in‑paint images by moving that slider
# anywhere in the range [‑1 … +1].
#
# Why a single file? You can copy‑paste it to any host, run `python
# lora_slider_inpaint.py train ‑‑help` to start training, or
# `python lora_slider_inpaint.py infer ‑‑help` to perform inference. Feel
# free to split it into separate scripts later.
#
# The code is written against **diffusers ≥ 0.26.0**, **peft ≥ 0.10.0**,
# **torch ≥ 2.1**, **accelerate**, **safetensors**, and **transformers**.
# It supports SD 1.5/2.x *and* SDXL in‑painting models – pass the correct
# `--pretrained_model_name_or_path`.
#
# ────────────────────────────────────────────────────────────────────────────
# DATA LAYOUT
# ───────────
# A minimal dataset directory looks like this::
#
#     dataset/
#       positive/           # images that *add* the attribute (slider +1)
#         0001.png
#         0001_mask.png     # same size as 0001.png, white=region to edit
#         0002.png …
#       negative/           # images that *remove* the attribute (slider ‑1)
#         0001.png
#         0001_mask.png
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
# • Training::
#
#     python lora_slider_inpaint.py train \
#         --pretrained_model runwayml/stable-diffusion-inpainting \
#         --dataset_dir /path/to/your/dataset \
#         --output_dir  /path/to/your/checkpoints/slider \
#         --resolution 512 --batch_size 4 --rank 4 --alpha 16 \
#         --max_steps 10000 --save_every 1000
#
# • Inference::
#
#     python lora_slider_inpaint.py infer \
#         --pretrained_model runwayml/stable-diffusion-inpainting \
#         --adapter_path  /path/to/your/checkpoints/slider/adapter-final.safetensors \
#         --input_image   example.png \
#         --mask_image    example_mask.png \
#         --prompt "a portrait with short bangs" \
#         --slider 0.3    # anywhere in [-1,1]
#         --output_image  out.png
#
# ────────────────────────────────────────────────────────────────────────────

import argparse
import itertools
import math
import os
import sys # Added for default command logic
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers.loaders import LoraLoaderMixin # Changed import for saving LoRA
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
import transformers # Ensure transformers is imported if needed for tokenizer
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

import warnings
warnings.simplefilter("ignore", FutureWarning)

# ---------------------------
#  UTILITY FUNCTIONS
# ---------------------------

def load_image(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size > 0:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_mask(path: Path, size: int) -> Image.Image:
    # mask assumed single channel, white = editable, black = keep
    mask = Image.open(path).convert("L")
    if size > 0:
        mask = mask.resize((size, size), Image.NEAREST)
    # Ensure mask is binary 0 or 1 for calculations later
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    return mask


class PairedInpaintDataset(Dataset):
    """Returns ((pos_img, pos_mask, pos_prompt), (neg_img, neg_mask, neg_prompt))"""

    def __init__(self, root: str, resolution: int = 512, default_prompt: str = ""):
        self.root = Path(root)
        self.pos_img_paths = sorted([f for f in (self.root / "positive").glob("*.png") if '_mask' not in f.stem])
        self.neg_img_paths = sorted([f for f in (self.root / "negative").glob("*.png") if '_mask' not in f.stem])

        if len(self.pos_img_paths) != len(self.neg_img_paths):
             raise ValueError(f"Number of images in positive ({len(self.pos_img_paths)}) and negative ({len(self.neg_img_paths)}) folders must match.")
        if len(self.pos_img_paths) == 0:
             raise ValueError(f"No images found in positive/negative folders under {root}")

        self.resolution = resolution
        self.default_prompt = default_prompt

        # Pre-compute file paths for masks and prompts to avoid repeated path logic
        self.pos_files = self._precompute_paths(self.pos_img_paths)
        self.neg_files = self._precompute_paths(self.neg_img_paths)

        # transformations -> tensor in [-1,1] for images, [0,1] for masks
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor(), # Converts L image (H, W) to (1, H, W) in [0, 1]
        ])

    def _precompute_paths(self, img_paths: List[Path]):
        file_triplets = []
        for img_path in img_paths:
            mask_path = img_path.with_name(img_path.stem + "_mask" + img_path.suffix)
            txt_path = img_path.with_suffix(".txt")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found for image {img_path}: expected {mask_path}")
            file_triplets.append({"img": img_path, "mask": mask_path, "txt": txt_path})
        return file_triplets

    def __len__(self):
        return len(self.pos_files)

    def _read_triplet(self, file_info: dict):
        img = load_image(file_info["img"], self.resolution)
        mask = load_mask(file_info["mask"], self.resolution)

        img_tensor = self.img_transforms(img)
        mask_tensor = self.mask_transforms(mask)

        prompt = self.default_prompt
        if file_info["txt"].exists():
             try:
                 prompt = file_info["txt"].read_text().strip()
             except Exception as e:
                 print(f"Warning: Could not read prompt file {file_info['txt']}. Using default. Error: {e}")

        return img_tensor, mask_tensor, prompt

    def __getitem__(self, idx):
        pos_triplet = self._read_triplet(self.pos_files[idx])
        neg_triplet = self._read_triplet(self.neg_files[idx])
        return pos_triplet, neg_triplet


# ---------------------------
#  TRAINING
# ---------------------------

def train_slider(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.mixed_precision else "no", # Use accelerator's mixed precision
    )
    device = accelerator.device
    set_seed(args.seed) # Add seed for reproducibility

    print("🧩 Loading in‑paint pipeline…")
    pipe_cls = StableDiffusionXLInpaintPipeline if "xl" in args.pretrained_model.lower() else StableDiffusionInpaintPipeline
    # Load with the precision expected by the accelerator for consistency
    try:
        pipe = pipe_cls.from_pretrained(args.pretrained_model, torch_dtype=torch.float16 if args.mixed_precision else torch.float32)
    except EnvironmentError as e:
        print(f"Error loading model '{args.pretrained_model}'. Is the path correct and model downloaded? Error: {e}")
        sys.exit(1)

    # --- Move components to the correct device ---
    # VAE and Text Encoder are not trained, move them explicitly
    pipe.vae.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32)
    pipe.text_encoder.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32)
    # UNet will be handled by accelerator.prepare later, but move base model first
    pipe.unet.to(device, dtype=torch.float16 if args.mixed_precision else torch.float32)
    # ---------------------------------------------

    print("🧊 Freezing VAE and Text Encoder...")
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    # UNet needs gradients, will be handled by PEFT

    print("🚀 Adding LoRA adapter to UNet...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Apply PEFT to the UNet already on the device
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    if accelerator.is_main_process:
        pipe.unet.print_trainable_parameters()

    print("💾 Loading Dataset...")
    try:
        dataset = PairedInpaintDataset(args.dataset_dir, args.resolution, args.prompt)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers # Add num_workers arg
    )

    print("⚙️ Setting up Optimizer...")
    # Optimizer should work on trainable parameters (LoRA weights)
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.learning_rate)

    # --- Use accelerator.prepare ---
    print("✨ Preparing with Accelerator...")
    pipe.unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )
    # Ensure text_encoder and tokenizer are accessible (they are part of the original pipe)
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    # VAE is also needed
    vae = pipe.vae
    # Scheduler is needed
    scheduler = pipe.scheduler
    # We also need the unet's dtype for casting later
    unet_dtype = pipe.unet.dtype
    vae_dtype = pipe.vae.dtype # Get VAE dtype too

    # Define encode function (using prepared vae)
    def encode(img_batch):
        # Ensure input is on the correct device/dtype for VAE
        img_batch = img_batch.to(device=device, dtype=vae_dtype)
        with torch.no_grad():
            latents = vae.encode(img_batch).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        return latents

    print("🔥 Starting Training Loop...")
    global_step = 0
    loss_vector = []
    for epoch in itertools.count():
        pipe.unet.train() # Use pipe.unet as it's the one prepared by accelerator
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                (img_p, mask_p, prompt_p), (img_n, mask_n, prompt_n) = batch
                # Data is assumed to be on device via accelerator.prepare(dataloader)

                # --- Encode Images to Latent Space ---
                latents_p = encode(img_p)
                latents_n = encode(img_n)

                # --- Sample noise in the latent space ---
                noise = torch.randn_like(latents_p) # Already on correct device/dtype

                # --- Sample Timesteps ---
                bsz = latents_p.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()

                # --- Add noise to latents ---
                noisy_latents_p = scheduler.add_noise(latents_p, noise, timesteps)
                noisy_latents_n = scheduler.add_noise(latents_n, noise, timesteps)

                # --- Manually Tokenize and Encode Text Prompts ---
                # Ensure prompts are lists of strings
                if isinstance(prompt_p, str): prompt_p = [prompt_p] * bsz
                if isinstance(prompt_n, str): prompt_n = [prompt_n] * bsz

                # Tokenize
                tokens_p = tokenizer(prompt_p, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
                tokens_n = tokenizer(prompt_n, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

                # Encode with text_encoder (ensure it's on the right device)
                with torch.no_grad():
                     # text_encoder should already be on 'device' and correct dtype
                     prompt_embeds_p = text_encoder(tokens_p.to(device))[0] # Get last_hidden_state
                     prompt_embeds_n = text_encoder(tokens_n.to(device))[0]

                # Cast embeds to UNet's expected dtype for mixed precision
                prompt_embeds_p = prompt_embeds_p.to(unet_dtype)
                prompt_embeds_n = prompt_embeds_n.to(unet_dtype)
                # -----------------------------------------------------

                # --- Prepare Mask and Masked Image Latents ---
                latent_h, latent_w = latents_p.shape[2], latents_p.shape[3]
                mask_p_latent = transforms.functional.resize(
                    mask_p, (latent_h, latent_w), interpolation=transforms.InterpolationMode.NEAREST
                ).to(device) # Ensure mask is on device
                mask_n_latent = transforms.functional.resize(
                    mask_n, (latent_h, latent_w), interpolation=transforms.InterpolationMode.NEAREST
                ).to(device) # Ensure mask is on device

                masked_image_latents_p = latents_p * (1.0 - mask_p_latent)
                masked_image_latents_n = latents_n * (1.0 - mask_n_latent)

                # Concatenate inputs for the UNet
                latent_model_input_p = torch.cat([noisy_latents_p, mask_p_latent, masked_image_latents_p], dim=1)
                latent_model_input_n = torch.cat([noisy_latents_n, mask_n_latent, masked_image_latents_n], dim=1)

                # Ensure concatenated inputs have the UNet's expected dtype
                latent_model_input_p = latent_model_input_p.to(unet_dtype)
                latent_model_input_n = latent_model_input_n.to(unet_dtype)

                # --- Forward pass through UNet (use the prepared pipe.unet) ---
                noise_pred_p = pipe.unet( # Access the prepared UNet directly
                    latent_model_input_p,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_p,
                    cross_attention_kwargs={"scale": 1.0}
                ).sample

                noise_pred_n = pipe.unet( # Access the prepared UNet directly
                    latent_model_input_n,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_n,
                    cross_attention_kwargs={"scale": -1.0}
                ).sample

                # --- Calculate Loss ---
                loss_p = torch.nn.functional.mse_loss(noise_pred_p.float(), noise.float())
                loss_n = torch.nn.functional.mse_loss(noise_pred_n.float(), noise.float())
                loss = (loss_p + loss_n) / 2.0

                # --- Backward Pass & Optimizer Step ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                         accelerator.clip_grad_norm_(pipe.unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # --- Logging and Saving ---
            if accelerator.sync_gradients: # Log/Save only on actual optimizer steps
                global_step += 1
                loss_vector.append(loss.item())

                if accelerator.is_main_process:
                    if global_step % args.log_every == 0:
                        print(f"Epoch {epoch}, Step {global_step} – Loss: {loss.item():.4f} (Pos: {loss_p.item():.4f}, Neg: {loss_n.item():.4f})")

                    if global_step % args.save_every == 0 and global_step > 0:
                        save_progress(pipe.unet, accelerator, args.output_dir, global_step)

            # Check if max steps reached
            if global_step >= args.max_steps:
                break # Exit inner loop (step loop)

        # Check again after epoch finishes in case max_steps lines up with epoch end
        if global_step >= args.max_steps:
            break # Exit outer loop (epoch loop)

    print("🏁 Training Finished.")
    if accelerator.is_main_process:
        save_progress(pipe.unet, accelerator, args.output_dir, "final")


def save_progress(unet: nn.Module, accelerator: Accelerator, out_dir: str, step_or_name):
    """Saves the LoRA adapter state."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"adapter-{step_or_name}.safetensors"

    # Unwrap the model before saving to get the underlying PEFT model
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Use PEFT's utility function to get only the adapter weights' state_dict
    try:
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
    except Exception as e:
        # Log the error and potentially exit or return if extraction fails
        print(f"Error getting PEFT model state dict at step {step_or_name}: {e}")
        # Depending on severity, you might want to return here
        return # Let's return if we can't get the state dict

    # Check if the state dict is empty (might happen if PEFT setup failed)
    if not lora_state_dict:
        print(f"WARNING: No parameters found in PEFT state_dict to save at step {step_or_name}.")
        return

    # Save the extracted state dictionary using safetensors
    try:
        save_file(lora_state_dict, str(filename))
        print(f"💾 Saved LoRA adapter checkpoint → {filename}")
    except Exception as e:
        print(f"Error saving LoRA adapter state_dict to file {filename}: {e}")
# ---------------------------
#  INFERENCE
# ---------------------------

def run_inference(args):
    device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float16 for inference if available and desired
    dtype = torch.float16 if device != torch.device("cpu") else torch.float32

    print(f"🚀 Loading pipeline {args.pretrained_model} on {device} with {dtype}...")
    pipe_cls = StableDiffusionXLInpaintPipeline if "xl" in args.pretrained_model.lower() else StableDiffusionInpaintPipeline
    try:
        pipe = pipe_cls.from_pretrained(args.pretrained_model, torch_dtype=dtype)
    except EnvironmentError as e:
        print(f"Error loading model '{args.pretrained_model}'. Is the path correct and model downloaded? Error: {e}")
        sys.exit(1)

    print(f"🚀 Loading LoRA adapter {args.adapter_path}...")
    try:
        # Use the pipeline's method to load LoRA weights
        pipe.load_lora_weights(args.adapter_path)
        print("✅ LoRA Adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA adapter from {args.adapter_path}: {e}")
        print("Continuing inference without LoRA adapter.")
        # Depending on the error, you might want to exit sys.exit(1)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=False) # Show progress bar

    print("🖼️ Loading input image and mask...")
    try:
        # Use the same loading logic but without resizing if pipeline handles it
        # Or resize to a specific inference size if needed
        raw_img = load_image(Path(args.input_image), size=args.resolution) # Use training resolution?
        raw_mask = load_mask(Path(args.mask_image), size=args.resolution)
    except FileNotFoundError as e:
        print(f"Error loading image or mask: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"Error processing image or mask: {e}")
         sys.exit(1)

    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"⏳ Running inpainting with slider value: {args.slider:.2f}")
    try:
        with torch.autocast(device.type, dtype=dtype if device.type == "cuda" else torch.bfloat16): # Autocast for inference
            result = pipe(
                prompt=args.prompt,
                image=raw_img,
                mask_image=raw_mask,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
                negative_prompt=args.negative_prompt,
                generator=generator,
                cross_attention_kwargs={"scale": args.slider}, # Pass the slider value
                num_images_per_prompt=1, # Generate one image
            ).images[0]

        # Try to show image if possible (might fail in non-GUI environments)
        try:
            result.show()
        except Exception:
            print("(Skipping image display in non-GUI environment)")

        output_path = Path(args.output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"✅ Saved result → {output_path}")

    except Exception as e:
        print(f"😭 Inference failed: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        sys.exit(1)

# ---------------------------
#  CLI ENTRY
# ---------------------------

def build_arg_parser():
    command = "infer" # "train" or "infer"
    model_path = "/workspace/models/sd1_5_inpaint"
    rank = 8# 4
    alpha = 1# 16
    batch_size = 4
    default_resolution = 512
    is_single_folder = True
    if is_single_folder:
        dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single"
    else:
        dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai"
    output_dir = f"/workspace/my_sliders/lora_adaptors/bs_oai_rank{rank}"
    if is_single_folder:
        output_dir += "_single"
    adapter_path = output_dir + "/adapter-530.safetensors"
    infer_input_image = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single/neutral/updated_caption_image.png"
    infer_mask_image = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single/neutral/updated_caption_image_mask.png"
    infer_prompt = "hairline control slider"
    train_prompt = infer_prompt

    parser = argparse.ArgumentParser(description="LoRA Slider Trainer / Inference for Stable Diffusion In‑Painting")
    # sub = parser.add_subparsers(dest="command", required=True)
    sub = parser.add_subparsers(dest="command", required=False)

    # —— Train ——
    train = sub.add_parser("train")
    # train.add_argument("--pretrained_model", required=True, help="HF model id or path, e.g. runwayml/stable-diffusion-inpainting")
    # train.add_argument("--dataset_dir", required=True, help="Folder with positive/ and negative/")
    # train.add_argument("--output_dir", required=True)
    train.add_argument("--resolution", type=int, default=default_resolution)
    train.add_argument("--prompt", default=train_prompt)
    train.add_argument("--rank", type=int, default=rank)
    train.add_argument("--alpha", type=int, default=alpha)
    train.add_argument("--batch_size", type=int, default=batch_size)
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--max_steps", type=int, default=10000)
    train.add_argument("--save_every", type=int, default=300)
    train.add_argument("--log_every", type=int, default=25)
    train.add_argument("--mixed_precision", action="store_true")
    train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    train.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")
    train.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (set to None to disable).")
    train.add_argument("--num_validation_images", type=int, default=1, help="Number of images to generate per prompt for validation (used internally for encoding). Should be 1 for this script.") # Added for _encode_prompt consistency

    train.add_argument("--pretrained_model", required=False, default=model_path, help="HF model id or path, e.g. runwayml/stable-diffusion-inpainting")
    train.add_argument("--dataset_dir", required=False, default=dataset_dir, help="Folder with positive/ and negative/")
    train.add_argument("--output_dir", required=False, default=output_dir)

    # —— Infer ——
    infer = sub.add_parser("infer")
    # infer.add_argument("--pretrained_model", required=True)
    # infer.add_argument("--adapter_path", required=True)
    # infer.add_argument("--input_image", required=True)
    # infer.add_argument("--mask_image", required=True)
    # infer.add_argument("--prompt", required=True)
    infer.add_argument("--output_image", default="result.png")
    infer.add_argument("--negative_prompt", default=None)
    # infer.add_argument("--rank", type=int, default=rank)
    # infer.add_argument("--alpha", type=int, default=alpha)
    infer.add_argument("--guidance_scale", type=float, default=7.5)
    infer.add_argument("--num_steps", type=int, default=50)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--slider", type=float, default=0.5, help="LoRA scale in [-1,1]")

    infer.add_argument("--pretrained_model", required=False, default=model_path)
    infer.add_argument("--adapter_path", required=False, default=adapter_path)
    infer.add_argument("--input_image", required=False, default=infer_input_image)
    infer.add_argument("--mask_image", required=False, default=infer_mask_image)
    infer.add_argument("--prompt", required=False, default=infer_prompt)
    infer.add_argument("--resolution", type=int, default=default_resolution, help="Resolution used during training (important for model compatibility).") # Added resolution arg to inference

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_slider(args)
    elif args.command == "infer":
        run_inference(args)

