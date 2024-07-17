import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora, convert_lora_unet
from diffusers import StableDiffusionPipeline
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer

from PIL import Image

import torchvision.transforms as transforms

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # domain adapter
    adapter_lora_path          = "",
    adapter_lora_scale         = 1.0,
    # image layers
    dreambooth_model_path      = "",
    lora_model_path            = "",
    lora_alpha                 = 0.8,
):

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    # lora layers
    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
                
        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    # assert len(unexpected) == 0
    del unet_state_dict
    
    # domain adapter lora
    if adapter_lora_path != "":
        print(f"load domain lora from {adapter_lora_path}")
        domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    start_index = 0
    if len(unexpected) > 0:
        start_index = 1
        lora_config = LoraConfig.from_pretrained("models/Motion_with_LoRA")
        animation_pipeline.unet.add_adapter(lora_config)
        unet_state_dict = {}
        if motion_module_path != "":
            print(f"load motion module from {motion_module_path}")
            motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
            motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
            unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
            unet_state_dict.pop("animatediff_config", "")
        
        # missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        # if "outputs/Transparentstroke_train_control-2024-05-11T11-32-53/checkpoints/mm-epoch-5.ckpt" != "":
        #     # print(f"load motion module from {"outputs/Transparentstroke_train_control-2024-05-13T04-48-51/checkpoints/mm-epoch-40000.ckpt"}")
        #     motion_module_state_dict = torch.load("outputs/Transparentstroke_train_control-2024-05-11T11-32-53/checkpoints/mm-epoch-5.ckpt", map_location="cpu")
        #     motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        #     unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        #     unet_state_dict.pop("animatediff_config", "")
        
        missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    # motion module lora
    for motion_module_lora_config in motion_module_lora_configs[start_index:]:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")
        # try:
        animation_pipeline.load_lora_weights(path, weight_name=path, adapter_name=path.split("/")[-1].split(".")[0])
        animation_pipeline.set_adapters([item['path'].split("/")[-1].split(".")[0] for item in list(motion_module_lora_configs)], adapter_weights=[1 for item in list(motion_module_lora_configs)])
        reset_pipeline_lora_alpha(animation_pipeline,
                        path.split("/")[-1].split(".")[0],
                        "models/Motion_with_LoRA")
        # except:
        #     motion_lora_state_dict = torch.load(path, map_location="cpu")
        #     assert path.endswith(".safetensors")
        #     motion_lora_state_dict = {}
        #     with safe_open(path, framework="pt", device="cpu") as f:
        #         for key in f.keys():
        #             motion_lora_state_dict[key] = f.get_tensor(key)
        #     motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        #     motion_lora_state_dict.pop("animatediff_config", "")

        #     animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)
        

    # animation_pipeline.set_adapters([item['path'].split("/")[-1].split(".")[0] for item in list(motion_module_lora_configs)], adapter_weights=[0.6 for item in list(motion_module_lora_configs)])
    return animation_pipeline

# @torch.no_grad()
def prepare_control(controlnet, controlnet_images_paths, timesteps, controlnet_noisy_latents, controlnet_prompt_embeds, normalize_condition_images, sample_size, batch_size, vae, video_length, controlnet_image_indexs, controlnet_conditioning_scale):
    if len(controlnet_image_indexs) == 0:
        return None, None
    if normalize_condition_images:
        def image_norm(image):
            image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
            image -= image.min()
            image /= image.max()
            return image
    else: image_norm = lambda x: x
    image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            (sample_size, sample_size), (1.0, 1.0), 
            ratio=(1, 1)
        ),
        transforms.ToTensor(),
    ])
    controlnet_images = torch.stack([torch.stack([image_norm(image_transforms(Image.open(image).convert("RGB"))) for image in images]) for index, images in enumerate(controlnet_images_paths) if index in controlnet_image_indexs]).to(controlnet_noisy_latents.device)
    
    controlnet_images = rearrange(controlnet_images, "f b c h w -> (b f) c h w", b=batch_size)
    controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
    controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", b=batch_size)
    controlnet_images = controlnet_images.to(controlnet_noisy_latents.device)

    controlnet_cond_shape    = list(controlnet_images.shape)
    controlnet_cond_shape[2] = video_length
    controlnet_cond = torch.zeros(controlnet_cond_shape).to(controlnet_noisy_latents.device)

    controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
    controlnet_conditioning_mask_shape[1] = 1
    controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(controlnet_noisy_latents.device)

    # assert controlnet_images.shape[2] >= len(controlnet_image_indexs)
    controlnet_cond[:,:,controlnet_image_indexs] = controlnet_images
    controlnet_conditioning_mask[:,:,controlnet_image_indexs] = 1

    down_block_additional_residuals, mid_block_additional_residual = controlnet(
        controlnet_noisy_latents, timesteps,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=controlnet_cond,
        conditioning_mask=controlnet_conditioning_mask,
        conditioning_scale=controlnet_conditioning_scale,
        guess_mode=False, return_dict=False,
    )

    return down_block_additional_residuals, mid_block_additional_residual

def load_dream_booth(dreambooth_model_path, unet, vae, text_encoder):
    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, vae.config)
        vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)
        unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
    return unet, vae, text_encoder

def generate_normal_shaped_symmetric_corrected(n, k):
    mu = (n - 1) / 2
    sigma = n / 6  # adjust this to control the width

    indices = np.arange(n)
    sequence = np.exp(-(indices - mu) ** 2 / (2 * sigma ** 2))

    current_sum = sequence.sum()
    scale_factor = k / current_sum
    scaled_sequence = sequence * scale_factor

    return scaled_sequence


def load_adapter(unet, adapter_lora_path, alpha=1.0):
    print(f"load domain lora from {adapter_lora_path}")
    state_dict = torch.load(adapter_lora_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    state_dict.pop("animatediff_config", "")

    # directly update weight in diffusers model
    for key in state_dict:
        # only process lora down key
        if "up." in key: continue

        up_key    = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key]
        weight_up   = state_dict[up_key]
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

    return unet


def reset_pipeline_lora_alpha(pipe: StableDiffusionPipeline,
                              adapter: str,
                              unet_lora_config_path: str = None,
                              text_encoder_lora_config_path: str = None):
    if unet_lora_config_path:
        lora_config = LoraConfig.from_pretrained(unet_lora_config_path)
        print(f'Loading unet adapter config from {unet_lora_config_path}')
        print(f'Reset adapter {adapter} lora alpha in unet to '
              f'{lora_config.lora_alpha}')
        reset_model_lora_alpha(pipe.unet, lora_config.lora_alpha, adapter)
    if text_encoder_lora_config_path:
        lora_config = LoraConfig.from_pretrained(text_encoder_lora_config_path)
        print(f'Loading text_encoder adapter config from '
              f'{text_encoder_lora_config_path}')
        print(f'Reset adapter {adapter} lora alpha in text_encoder to '
              f'{lora_config.lora_alpha}')
        reset_model_lora_alpha(pipe.text_encoder,
                               lora_config.lora_alpha,
                               adapter)
        if hasattr(pipe, 'text_encoder_2'):
            print(f'Reset adapter {adapter} lora alpha in text_encoder_2 to '
                  f'{lora_config.lora_alpha}')
            reset_model_lora_alpha(pipe.text_encoder_2,
                                   lora_config.lora_alpha,
                                   adapter)


def reset_model_lora_alpha(model: torch.nn.Module, lora_alpha: float, adapter: str):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            reset_lora_alpha(module, lora_alpha, adapter)


def reset_lora_alpha(lora_layer: LoraLayer, lora_alpha: float, adapter: str):
    # Modified from peft.tuners.lora.layer.LoraLayer.
    if adapter not in lora_layer.active_adapters:
        return

    lora_layer.lora_alpha[adapter] = lora_alpha
    if lora_layer.r[adapter] > 0:
        lora_layer.scaling[adapter] = (lora_layer.lora_alpha[adapter]
                                       / lora_layer.r[adapter])