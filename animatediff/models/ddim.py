# Adapted from https://github.com/shaibagon/diffusers_ddim_inversion
from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
import torch.nn.functional as F

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def ddim_inversion(imgname: str, pipe: StableDiffusionPipeline, num_steps: int = 50, verify: Optional[bool] = False, sample_size: int = 512) -> torch.Tensor:
    dtype = torch.float16

    vae = pipe.vae

    input_img = load_image(imgname).to(device=pipe.device, dtype=dtype)
    input_img = input_img.float()
    input_img = F.interpolate(input_img, size=(sample_size, sample_size), mode='bilinear', align_corners=False)
    input_img = input_img.half()
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=sample_size, height=sample_size,
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    # verify
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained('/home/ubuntu/projects/nicolaus/prompt2draw/models/StableDiffusion', subfolder='scheduler')
        image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
    return inv_latents