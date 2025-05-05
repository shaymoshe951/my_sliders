"""
LoRA Slider Training & Inference for Stable Diffusion Inpainting
================================================================
Author: ChatGPT (May 2025)

This single Python module provides **both** a training routine to learn a
LoRA adapter that behaves like a *continuous attribute slider* **and** an
inference routine that lets you in‑paint images by moving that slider
anywhere in the range [‑1 … +1].

Why a single file? You can copy‑paste it to any host, run `python
lora_slider_inpaint.py train ‑‑help` to start training, or
`python lora_slider_inpaint.py infer ‑‑help` to perform inference.  Feel
free to split it into separate scripts later.

The code is written against **diffusers ≥ 0.26.0**, **peft ≥ 0.10.0**,
**torch ≥ 2.1**, **accelerate**, **safetensors**, and **transformers**.
It supports SD 1.5/2.x *and* SDXL in‑painting models – pass the correct
`--pretrained_model_name_or_path`.

────────────────────────────────────────────────────────────────────────────
DATA LAYOUT
───────────
A minimal dataset directory looks like this::

    dataset/
      positive/           # images that *add* the attribute (slider +1)
        0001.png
        0001_mask.png     # same size as 0001.png, white=region to edit
        0002.png …
      negative/           # images that *remove* the attribute (slider ‑1)
        0001.png
        0001_mask.png
        0002.png …

Prompts (optional) → `0001.txt` beside each image, otherwise a *global*
`--prompt` is used during training.

During training the script loads *paired* samples (positive[i],
negative[i]) so that gradients for both ends of the slider are computed
*within the same batch*.

────────────────────────────────────────────────────────────────────────────
CLI USAGE (BASIC)
────────────────
• **Training**::

    python lora_slider_inpaint.py train \
        --pretrained_model runwayml/stable-diffusion-inpainting \
        --dataset_dir /workspace/dataset/HrSample512 \
        --output_dir  /workspace/checkpoints/slider \
        --resolution 512 --batch_size 4 --rank 4 --alpha 16 \
        --max_steps 10000 --save_every 1000

• **Inference**::

    python lora_slider_inpaint.py infer \
        --pretrained_model runwayml/stable-diffusion-inpainting \
        --adapter_path  /workspace/checkpoints/slider/adapter-final.safetensors \
        --input_image   example.png \
        --mask_image    example_mask.png \
        --prompt "a portrait with short bangs" \
        --slider 0.3    # anywhere in [-1,1]
        --output_image  out.png

────────────────────────────────────────────────────────────────────────────
"""

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers.loaders import AttnProcsLayers
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from safetensors.torch import save_file

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
    return mask


class PairedInpaintDataset(Dataset):
    """Returns ((pos_img, pos_mask, pos_prompt), (neg_img, neg_mask, neg_prompt))"""

    def __init__(self, root: str, resolution: int = 512, default_prompt: str = ""):
        self.pos_files = [fpath for fpath in sorted((Path(root) / "positive").glob("*.png")) if 'mask' not in fpath.stem ]
        self.neg_files = [fpath for fpath in sorted((Path(root) / "negative").glob("*.png")) if 'mask' not in fpath.stem ]
        assert len(self.pos_files) == len(self.neg_files) > 0, "Positive and negative folders must be same length"
        self.resolution = resolution
        self.default_prompt = default_prompt
        # transformations → tensor in [0,1]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1,1]
        ])
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pos_files)

    def _read_triplet(self, img_path: Path, resolution: int):
        mask_path = img_path.parent / (img_path.stem + "_mask" + img_path.suffix)
        txt_path = img_path.with_suffix(".txt")
        img = self.to_tensor(load_image(img_path, resolution))
        mask = self.mask_to_tensor(load_mask(mask_path, resolution))
        prompt = open(txt_path).read().strip() if txt_path.exists() else self.default_prompt
        return img, mask, prompt

    def __getitem__(self, idx):
        return self._read_triplet(self.pos_files[idx], self.resolution), \
               self._read_triplet(self.neg_files[idx], self.resolution)


# ---------------------------
#  TRAINING
# ---------------------------

def train_slider(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    print("🧩 Loading in‑paint pipeline…")
    pipe_cls = StableDiffusionXLInpaintPipeline if "xl" in args.pretrained_model.lower() else StableDiffusionInpaintPipeline
    pipe = pipe_cls.from_pretrained(args.pretrained_model, torch_dtype=torch.float16 if args.mixed_precision else torch.float32)
    pipe.vae.requires_grad_(False)  # freeze VAE
    pipe.text_encoder.requires_grad_(False)  # freeze text encoder

    # Add LoRA to UNet
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    dataset = PairedInpaintDataset(args.dataset_dir, args.resolution, args.prompt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.learning_rate)

    global_step = 0
    pipe.unet.train()

    for epoch in range(10_000):
        for (pos, neg) in dataloader:
            ((img_p, mask_p, prompt_p), (img_n, mask_n, prompt_n)) = pos, neg
            bsz = img_p.size(0)
            img_p, img_n = img_p.to(device), img_n.to(device)
            mask_p, mask_n = mask_p.to(device), mask_n.to(device)

            # Sample timesteps
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(img_p)

            def encode(img):
                latents = pipe.vae.encode(img.to(pipe.vae.dtype)).latent_dist.sample() * pipe.vae.config.scaling_factor
                latents = latents + noise
                return latents

            latents_p = encode(img_p)
            latents_n = encode(img_n)

            # ================= Forward with +1 (adds attribute) =================
            noise_pred_p = pipe.unet(latents_p, timesteps, encoder_hidden_states=None, mask=mask_p, lora_scale=+1.0).sample
            loss_p = torch.nn.functional.mse_loss(noise_pred_p, noise)

            # ================= Forward with -1 (removes attribute) ===============
            noise_pred_n = pipe.unet(latents_n, timesteps, encoder_hidden_states=None, mask=mask_n, lora_scale=-1.0).sample
            loss_n = torch.nn.functional.mse_loss(noise_pred_n, noise)

            loss = (loss_p + loss_n) / 2.0
            accelerator.backward(loss)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % args.log_every == 0:
                print(f"step {global_step} – loss {loss.item():.4f}")

            if accelerator.is_main_process and global_step % args.save_every == 0 and global_step > 0:
                save_adapter(pipe.unet, args.output_dir, global_step)

            global_step += 1
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    if accelerator.is_main_process:
        save_adapter(pipe.unet, args.output_dir, "final")


def save_adapter(unet: nn.Module, out_dir: str, step):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"adapter-{step}.safetensors"
    lora_state_dict = AttnProcsLayers(unet).state_dict()
    save_file(lora_state_dict, str(filename))
    print(f"💾 Saved LoRA adapter → {filename}")

# ---------------------------
#  INFERENCE
# ---------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe_cls = StableDiffusionXLInpaintPipeline if "xl" in args.pretrained_model.lower() else StableDiffusionInpaintPipeline
    pipe = pipe_cls.from_pretrained(args.pretrained_model, torch_dtype=torch.float16).to(device)

    # Load LoRA adapter
    lora_state = torch.load(args.adapter_path, map_location="cpu")
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, bias="none")
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.load_state_dict(lora_state, strict=False)

    # Prepare images
    raw_img = Image.open(args.input_image).convert("RGB")
    raw_mask = Image.open(args.mask_image).convert("L")

    # Inpaint
    with torch.autocast("cuda"):
        result = pipe(
            prompt=args.prompt,
            image=raw_img,
            mask_image=raw_mask,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            negative_prompt=args.negative_prompt,
            generator=torch.Generator(device).manual_seed(args.seed),
            lora_scale=args.slider,
        ).images[0]

    result.show()
    result.save(args.output_image)
    print(f"✅ Saved result → {args.output_image}")

# ---------------------------
#  CLI ENTRY
# ---------------------------

def build_arg_parser():
    command = "train" # infer
    model_path = "/workspace/models/sd1_5_inpaint"
    rank = 4
    alpha = 16
    batch_size = 4
    is_single_folder = True
    if is_single_folder:
        dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai/single"
    else:
        dataset_dir = "/workspace/my_sliders/datasets/Different_hairline_db_oai"
    output_dir = f"/workspace/my_sliders/models/bs_oai_rank{rank}"
    if is_single_folder:
        output_dir += "_single"
    adapter_path = output_dir + "/adapter-final.safetensors"
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
    train.add_argument("--resolution", type=int, default=512)
    train.add_argument("--prompt", default=train_prompt)
    train.add_argument("--rank", type=int, default=rank)
    train.add_argument("--alpha", type=int, default=alpha)
    train.add_argument("--batch_size", type=int, default=batch_size)
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--max_steps", type=int, default=10000)
    train.add_argument("--save_every", type=int, default=1000)
    train.add_argument("--log_every", type=int, default=50)
    train.add_argument("--mixed_precision", action="store_true")

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
    infer.add_argument("--rank", type=int, default=rank)
    infer.add_argument("--alpha", type=int, default=alpha)
    infer.add_argument("--guidance_scale", type=float, default=7.5)
    infer.add_argument("--num_steps", type=int, default=50)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--slider", type=float, default=0.0, help="LoRA scale in [-1,1]")

    infer.add_argument("--pretrained_model", required=False, default=model_path)
    infer.add_argument("--adapter_path", required=False, default=adapter_path)
    infer.add_argument("--input_image", required=False, default=infer_input_image)
    infer.add_argument("--mask_image", required=False, default=infer_mask_image)
    infer.add_argument("--prompt", required=False, default=infer_prompt)

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_slider(args)
    elif args.command == "infer":
        run_inference(args)

