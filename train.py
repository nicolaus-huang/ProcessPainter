import os
import math
import wandb
import random
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple, Callable, List, Union

import torch
import torchvision
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from accelerate import Accelerator
from accelerate.logging import get_logger

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
import torchvision.transforms as transforms

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print, prepare_control, load_dream_booth, generate_normal_shaped_symmetric_corrected, load_adapter, convert_lora_unet
from animatediff.models.ipa import ImageProjModel, Resampler
from animatediff.models.ipa import is_torch2_available
from animatediff.models.sparse_controlnet import SparseControlNetModel
if is_torch2_available():
    from animatediff.models.ipa import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from animatediff.models.ipa import IPAttnProcessor, AttnProcessor

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from collections import OrderedDict

from PIL import Image

import numpy as np

logger = get_logger(__name__, log_level="INFO")

def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    mm_checkpoint_path: str = "",
    ipa_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 16,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    lora_rank: int = 4,
    lora_layers: Tuple[str] = (None, ),
    lora_alpha: int = 0.75,

    image_prompt_scale: float = 1.0,
    ddim: bool = True,
    enable_ipa: bool = True,
    controlnet_path: str = None,
    controlnet_config: str = None,
    normalize_condition_images: bool = False,
    controlnet_image_indexes: list = [0],
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    random_indexes: bool = True,
    dreambooth_model_path: str = None,
    adapter_lora_path: str = None,
    adapter_lora_alpha: float = 1,
    mixed_training: bool = False,
    ipa_plus: bool = False,
    lora_model_pathes: list = None
):

    check_min_version("0.10.0.dev0")

    *_, config = inspect.getargvalues(inspect.currentframe())

    load_dream_booth_index = 0
    # Initialize distributed training
    accelerator = Accelerator()

    seed = global_seed
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_local_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if enable_ipa:
        image_processor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("models/IP-Adapter", subfolder="image_encoder")
    else:
        image_processor = None
        image_encoder = None
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")


    if enable_ipa:
        ipa = ImageProjModel() if not ipa_plus else Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
        num_tokens = 4 if not ipa_plus else 16

        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") or 'motion_modules' in name else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

        ipa.load_from_checkpoint(ipa_checkpoint_path, adapter_modules)
    else:
        ipa = None


        
    controlnet = None
    if controlnet_config is not None:
        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None
        controlnet_config = OmegaConf.load(controlnet_config)
        controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))
    if controlnet_path is not None:
        assert controlnet_config != None

        print(f"loading controlnet checkpoint from {controlnet_path} ...")
        controlnet_state_dict = torch.load(controlnet_path, map_location="cpu")
        # print(controlnet_state_dict.keys())
        # print(controlnet.state_dict().keys())
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict, strict=False)
    
    
    if accelerator.is_local_main_process:
        # train lora
        if len(lora_layers) > 0:
            named_lora_layers = list()
            for layer in lora_layers:
                for key in unet.state_dict().keys():
                    if layer in key and ('to_k' in key or 'to_q' in key or 'to_v' in key or 'to_out.0' in key):
                        named_lora_layers.append('.'.join(key.split('.')[:-1]))
            unet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=named_lora_layers,
            )

            unet.add_adapter(unet_lora_config)
        # Load domain adapter
        if adapter_lora_path != None and adapter_lora_path != "":
            unet = load_adapter(unet, adapter_lora_path, adapter_lora_alpha)
        # Load pretrained mm weights
        if mm_checkpoint_path != "":
            # accelerator.print(f"from checkpoint: outputs/lora_Characterfulsketch-2024-05-09T18-08-31/checkpoints/mm-epoch-819.ckpt")
            # mm_state_dict = torch.load("outputs/lora_Characterfulsketch-2024-05-09T18-08-31/checkpoints/mm-epoch-819.ckpt", map_location="cpu")

            # m, u = unet.load_state_dict(mm_state_dict, strict=False)
            mm_state_dict = torch.load(mm_checkpoint_path, map_location="cpu")

            m, u = unet.load_state_dict(mm_state_dict, strict=False)
            accelerator.print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
            # assert len(u) == 0
        # Load pretrained unet weights
        if unet_checkpoint_path != "":
            accelerator.print(f"from checkpoint: {unet_checkpoint_path}")
            unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
            if "global_step" in unet_checkpoint_path: accelerator.print(f"global_step: {unet_checkpoint_path['global_step']}")
            state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

            m, u = unet.load_state_dict(state_dict, strict=False)
            accelerator.print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0

        if dreambooth_model_path is not None:
            unet, vae, text_encoder = load_dream_booth(dreambooth_model_path, unet, vae, text_encoder)

        
        # add lora layer
        if lora_model_pathes is not None:
            for lora_model_path in lora_model_pathes:
                print(f"load lora model from {lora_model_path}")
                assert lora_model_path.endswith(".safetensors")
                lora_state_dict = {}
                with safe_open(lora_model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        lora_state_dict[key] = f.get_tensor(key)
                        
                unet, text_encoder = convert_lora_unet(unet, text_encoder, lora_state_dict, alpha=lora_alpha)
                del lora_state_dict
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if enable_ipa: 
        ipa.requires_grad_(False)
        image_encoder.requires_grad_(False)
    
    # accelerator.print("Trainable Modules")
    # Set unet trainable parameters
    if controlnet is not None: controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    trainable_params_controlnet = []
    if 'controlnet' in trainable_modules:
        controlnet.requires_grad_(True)
        trainable_params_controlnet = list(controlnet.parameters())
    
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                accelerator.print(name)
                param.requires_grad = True
                break
    trainable_params_unet = list(filter(lambda p: p.requires_grad, unet.parameters()))

    trainable_params = trainable_params_controlnet + trainable_params_unet
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    accelerator.print(f"trainable params number: {len(trainable_params)}")
    accelerator.print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    train_dataset = WebVid10M(**train_data, control_indexes=list(range(train_data.sample_n_frames)), is_image=image_finetune)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, image_encoder=image_encoder, image_processor=image_processor, ipa=ipa, scheduler=noise_scheduler, ddim=ddim, controlnet=controlnet,
        )
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    unet, optimizer, train_dataloader, ipa, vae, text_encoder, image_encoder, controlnet = accelerator.prepare(unet, optimizer, train_dataloader, ipa, vae, text_encoder, image_encoder, controlnet)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):

            torch.cuda.empty_cache()

            if random_indexes:
                indexes_size = random.randint(1, 8)
                probabilities = [1/3] + list(generate_normal_shaped_symmetric_corrected(train_data.sample_n_frames - 2, 1/3)) + [1/3]
                controlnet_image_indexes = list(np.random.choice(range(train_data.sample_n_frames), size=indexes_size, replace=False, p=probabilities).astype(int))
            
            if mixed_training:
                enable_ipa = random.choice([True, False])

                    
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{accelerator.num_processes}-{idx}'}.gif", rescale=True, fps=train_data.sample_n_frames)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{accelerator.num_processes}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

                if enable_ipa:
                    if len(controlnet_image_indexes) == 0:
                        ipa_indexes = [train_data.sample_n_frames - 1]
                    else:
                        ipa_indexes = controlnet_image_indexes
                    image_inputs = image_processor(images=[Image.open(images[-1]) for images in  [image for index, image in enumerate(batch['image']) if index == ipa_indexes[-1]]], return_tensors="pt").pixel_values
                    if not ipa_plus:
                        image_encoder_hidden_states = image_encoder(image_inputs.to(accelerator.device)).image_embeds
                    else:
                        image_encoder_hidden_states = image_encoder(image_inputs.to(accelerator.device), output_hidden_states=True).hidden_states[-2]
                    ip_tokens = ipa(image_encoder_hidden_states)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
                
            down_block_additional_residuals = mid_block_additional_residual = None
            if controlnet != None:

                down_block_additional_residuals, mid_block_additional_residual = prepare_control(controlnet, batch['image'], timesteps, \
                                                                                                    noisy_latents, encoder_hidden_states, \
                                                                                                    normalize_condition_images, train_data.sample_size, train_batch_size, vae, \
                                                                                                        video_length, controlnet_image_indexes, controlnet_conditioning_scale)
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            # with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # print("DEBUG", noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                    down_block_additional_residuals = down_block_additional_residuals,
                    mid_block_additional_residual   = mid_block_additional_residual).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            accelerator.backward(loss)
            """ >>> gradient clipping >>> """
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            """ <<< gradient clipping <<< """
            optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if accelerator.is_local_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if accelerator.is_local_main_process and (global_step % checkpointing_steps == 0):
                unwrapped_unet, unwrapped_controlnet = accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet)
                save_path = os.path.join(output_dir, f"checkpoints")

                # save sd
                # state_dict = {
                #     "epoch": epoch,
                #     "global_step": global_step,
                #     "state_dict": unwrapped_unet.state_dict(),
                # }
                # if global_step % checkpointing_steps == 0:
                #     torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{global_step}.ckpt"))
                # else:
                #     torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                # logger.info(f"Saved state to {save_path} (global_step: {global_step})")

                # save lora
                if len(lora_layers) > 0:
                    unwrapped_unet = unwrapped_unet.to(torch.float32)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_unet)
                    )
                    
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=save_path,
                        unet_lora_layers=unet_lora_state_dict, 
                        safe_serialization=True,
                    )
                    unet_lora_config.save_pretrained(save_path)
                    logger.info(f"Saved LoRa state to {save_path} (global_step: {global_step})")

                # save mm
                unet_mm_state_dict = OrderedDict()
                for name, param in unwrapped_unet.named_parameters():
                    if 'motion_modules' in name:
                        unet_mm_state_dict[name] = param
                if global_step % checkpointing_steps == 0:
                    torch.save(unet_mm_state_dict, os.path.join(save_path, f"mm-epoch-{global_step}.ckpt"))
                else:
                    torch.save(unet_mm_state_dict, os.path.join(save_path, f"mm.ckpt"))
                logger.info(f"Saved Motion Modules state to {save_path} (global_step: {global_step})")
                
                # save cn
                try:
                    if unwrapped_controlnet != None:
                        
                        cn_state_dict = unwrapped_controlnet.state_dict()
                        if global_step % checkpointing_steps == 0:
                            torch.save(cn_state_dict, os.path.join(save_path, f"cn-epoch-{global_step}.ckpt"))
                        else:
                            torch.save(cn_state_dict, os.path.join(save_path, f"cn.ckpt"))
                        logger.info(f"Saved Controlnet state to {save_path} (global_step: {global_step})")
                except:
                    pass
            # Periodically validation
            if accelerator.is_local_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts
                images = validation_data.images[:2] if global_step < 1000 and (not image_finetune) else validation_data.images

                for idx, (prompt, image) in enumerate(zip(prompts, images)):
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            image,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            device       = latents.device,
                            **validation_data,
                            controlnet_images = image,
                            controlnet_image_indexes = controlnet_image_indexes,
                            controlnet_conditioning_scale = controlnet_conditioning_scale,
                            normalize_condition_images = normalize_condition_images,
                            sample_size=train_data.sample_size,
                            enable_ipa=enable_ipa,
                            ipa_plus=ipa_plus
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif", fps=train_data.sample_n_frames)
                        samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            image,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path, fps=train_data.sample_n_frames)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logger.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="/tiamat-vePFS/share_data/songyiren/nicolaus/siggraph24/configs/training/lora_Characterfulsketch.yaml", required=False)
    parser.add_argument("--wandb",    default=False, action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, use_wandb=args.wandb, **config)
