import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from diffusers import DDIMScheduler, StableDiffusionPipeline
import ptp_utils


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_256(image_path, left=0, right=0, top=0, bottom=0):
    image = Image.open(image_path)
    image = np.array(image.resize((256,256)))
    return image

def load_mask_256(mask_path):
    """Load and preprocess mask image to 256x256 binary format."""
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask = np.array(mask.resize((256, 256))) / 255.0
    mask = (mask > 0.5).astype(np.float32)  # Binary: 0 (preserve), 1 (inpaint)
    return mask

class NullInversion:
    def __init__(self, model, guidance_scale = 7.5, device = 'cuda', num_ddim_steps = 50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.guidance_scale = guidance_scale
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        latents_input = latents_input.to(self.model.unet.dtype)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents.to(self.model.vae.dtype)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).to(torch.float16).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.model.vae.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent = latent.to(self.model.unet.dtype)
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)

        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10, early_stop_epsilon=1e-5,
               verbose=False):
        self.init_prompt(prompt)

        ptp_utils.register_attention_control(self.model, None)

        # image_gt = load_512(image_path, *offsets)
        image_gt = load_256(image_path, *offsets)
        print(256)
        # plt.imshow(Image.fromarray(image_gt))
        # plt.show()
        # image_gt = torch.from_numpy(image_gt).to(dtype=torch.float16, device="cuda")

        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

class NullInversionInpaint:
    def __init__(self, model, guidance_scale=7.5, device='cuda', num_ddim_steps=50, is_concat_latent = False):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        self.guidance_scale = guidance_scale
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None
        self.is_concat_latent = is_concat_latent

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context, mask_latent=None, masked_image_latents=None):
        """Predict noise for a single latent with inpainting inputs."""
        if mask_latent is not None and masked_image_latents is not None and self.is_concat_latent:
            # Ensure mask_latent is a tensor
            if isinstance(mask_latent, np.ndarray):
                mask_latent = torch.from_numpy(mask_latent).float().to(latents.device)
            if self.is_concat_latent:
                # Concatenate latents, masked image latents, and mask for 9-channel input
                unet_input = torch.cat([latents, masked_image_latents, mask_latent], dim=1)  # Shape: [1, 9, 32, 32]
        else:
            unet_input = latents
        noise_pred = self.model.unet(unet_input, t, encoder_hidden_states=context)["sample"]
        if not self.is_concat_latent:
            if isinstance(mask_latent, np.ndarray):
                mask_latent = torch.from_numpy(mask_latent).float().to(latents.device)
            noise_pred = mask_latent * noise_pred + (1 - mask_latent) * latents

        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None, mask_latent=None, masked_image_latents=None):
        """Predict noise with classifier-free guidance and inpainting inputs."""
        if mask_latent is not None and masked_image_latents is not None and self.is_concat_latent:
            # Ensure mask_latent is a tensor
            if isinstance(mask_latent, np.ndarray):
                mask_latent = torch.from_numpy(mask_latent).float().to(latents.device)
            # Concatenate inputs for both unconditional and conditional paths
            unet_input = torch.cat([latents, masked_image_latents, mask_latent], dim=1)  # Shape: [1, 9, 32, 32]
            latents_input = torch.cat([unet_input] * 2)  # Shape: [2, 9, 32, 32]
        else:
            latents_input = torch.cat([latents] * 2)
        latents_input = latents_input.to(self.model.unet.dtype)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if mask_latent is not None and not self.is_concat_latent:
            # Ensure mask_latent is a tensor
            if isinstance(mask_latent, np.ndarray):
                mask_latent = torch.from_numpy(mask_latent).float().to(latents.device)
            # Apply mask to noise prediction
            noise_pred = mask_latent * noise_pred + (1 - mask_latent) * latents
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, mask_latent=None, original_latents=None, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents.to(self.model.vae.dtype)
        image = self.model.vae.decode(latents)['sample']
        if mask_latent is not None and original_latents is not None:
            # Ensure mask_latent is a tensor
            if isinstance(mask_latent, np.ndarray):
                mask_latent = torch.from_numpy(mask_latent).float().to(latents.device)
            original_image = self.model.vae.decode(1 / 0.18215 * original_latents.to(self.model.vae.dtype))['sample']
            mask_pixel = torch.nn.functional.interpolate(
                mask_latent, size=image.shape[-2:], mode='nearest'
            )
            image = mask_pixel * image + (1 - mask_pixel) * original_image
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).to(torch.float16).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image, mask=None):
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.model.vae.dtype)
            latents = self.model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
        if mask is not None:
            # Convert mask to tensor in pixel space
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(0).unsqueeze(0)
            # Mask should be [1, 1, 256, 256] for pixel-space operation
            mask_image_level = torch.nn.functional.interpolate(
                mask, size=image.shape[-2:], mode='nearest'
            )
            masked_image = image * (1 - mask_image_level)  # Zero out masked region
            masked_image_latents = self.model.vae.encode(masked_image)['latent_dist'].mean * 0.18215
            return latents, masked_image_latents
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, mask_latent, original_latents, masked_image_latents):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent = latent.to(self.model.unet.dtype)
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, mask_latent, masked_image_latents)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image, mask):
        latents, masked_image_latents = self.image2latent(image, mask)
        original_latents = latents.clone().detach()  # Store original latents for blending
        mask_latent = torch.from_numpy(mask).float().to(self.device).unsqueeze(0).unsqueeze(0)
        mask_latent = torch.nn.functional.interpolate(
            mask_latent, size=latents.shape[-2:], mode='nearest'
        )  # Resize to latent space (e.g., 32x32)
        image_rec = self.latent2image(latents, mask_latent, original_latents)
        ddim_latents = self.ddim_loop(latents, mask_latent, original_latents, masked_image_latents)
        return image_rec, ddim_latents

    def null_optimization(self, latents, mask_latent, original_latents, num_inner_steps, epsilon):
        # Convert mask_latent to tensor in latent space
        if isinstance(mask_latent, np.ndarray):
            mask_latent = torch.from_numpy(mask_latent).float().to(self.device).unsqueeze(0).unsqueeze(0)
            mask_latent = torch.nn.functional.interpolate(
                mask_latent, size=latents[0].shape[-2:], mode='nearest'
            )
        # Generate masked image latents for optimization
        _, masked_image_latents = self.image2latent(
            self.latent2image(original_latents, return_type='np'), mask_latent.cpu().numpy().squeeze()
        )
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, mask_latent, masked_image_latents)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, mask_latent, masked_image_latents)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                # Apply mask to loss computation
                loss = nnf.mse_loss(latents_prev_rec * mask_latent, latent_prev * mask_latent)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, mask_latent, masked_image_latents)
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, mask_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10,
               early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_256(image_path, *offsets)
        mask = load_mask_256(mask_path)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt, mask)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            ddim_latents, mask, self.image2latent(image_gt), num_inner_steps, early_stop_epsilon
        )
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    @property
    def scheduler(self):
        return self.model.scheduler



# null_inversion = NullInversion(ldm_stable)