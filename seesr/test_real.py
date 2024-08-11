'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1,0'
os.chdir(os.path.dirname(__file__))
os.environ["HF_DATASETS_CACHE"] = "/home/tiger/gh/cache/"
os.environ["HF_HOME"] = "/home/tiger/gh/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/tiger/gh/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/home/tiger/gh/cache/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
from einops import rearrange, repeat

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix ,pad_image_to_size

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from seesr_register import regiter_attention_editor_diffusers, AttentionBase,MutualSelfAttentionControl


from pytorch_lightning import seed_everything
seed_everything(1234)


logger = get_logger(__name__, log_level="INFO")

from torchvision.transforms import ToTensor, ToPILImage
to_image = ToPILImage()


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='/home/tiger/gh/cache/recognize_anything_model/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")
    # TODO 劫持pipe的unet /.up_blocks
    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention) 

    # =================== Hijack Here ======================
    STEP = 4
    LAYPER = 10
    # 从第十层，第四个时间步往后，我们施加控制
    editor = MutualSelfAttentionControl(STEP, LAYPER)
    regiter_attention_editor_diffusers(pipeline,editor)


    model = load_tag_model(args, accelerator.device)
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]


        # if 'CUFED' in args.image_path:
        #     new_image_names = []
        #     # cufed5只保留_0
        #     for file_path in image_names:
        #         idx = os.path.basename(file_path).split('_')[-1].split('.')[0]
        #         if int(idx) == 0:
        #             new_image_names.append(file_path)
        #     image_names = new_image_names


        for image_idx, image_name in enumerate(image_names[:]):
            # if image_idx <29:
            #     continue

            print(f'================ process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed,  
            
            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale 
            
            # 如果经过X4上采样之后还比512大。则不进行人工上采样处理
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True
            
            # 和SUPIR一模一样，都是先对LQ做了人工的上采样，并且同样做了64倍的整除处理
            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale)) 
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #


            # ================= Ref-image =======================
            if 'CUFED5' in args.image_path:
                img_id = os.path.basename(image_name).split('_')[0]
                ref_image= Image.open(os.path.join(os.path.dirname(image_name).replace('Real_Deg/LR', 'ref'), img_id+'_1.png')).convert("RGB")
            elif 'WR' in args.image_path:
                if 'Syn' in image_name:
                    ref_im_path = os.path.join(os.path.dirname(image_name).replace('Syn_Deg/Jpeg', 'ref'), os.path.basename(image_name).replace('HR','ref'))
                else:
                    ref_im_path = os.path.join(os.path.dirname(image_name).replace('Real_Deg/LR', 'ref'), os.path.basename(image_name).replace('LR','ref'))
                ref_image = Image.open(ref_im_path).convert("RGB")
            else:
                raise NotImplementedError
            
            ref_validation_prompt, ref_ram_encoder_hidden_states = get_validation_prompt(args, ref_image, model)
            ref_validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            ref_negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            ref_width, ref_height = ref_image.size
            if ref_width *2 < width or ref_height *2 < height: # 防止ref太小
                scale_ = max(width/(2*ref_width), height/(2*ref_height))
                ref_image =  ref_image.resize((int(scale_*ref_width)+8, int(scale_*ref_height)+8))
            ref_image =  ref_image.resize(((ref_image.size[0]//8)*8, (ref_image.size[1]//8)*8))
            

            
            # =========cat ref and LQ ============
            validation_image = tensor_transforms(validation_image).unsqueeze(0)
            ref_image = tensor_transforms(ref_image).unsqueeze(0)
            ref_image = ref_image[:, :, :int(2 * height / 64) * 64, :int(2 * width / 64.0) * 64] 
            ref_height, ref_width = ref_image.shape[2:]

            # pading 组合ref + img
            if ref_height != height or width != ref_width:
                target_h = max(ref_height, height)
                target_w = max(width, ref_width)
                validation_image = torch.cat([validation_image, torch.flip(validation_image, [2])], 2)[:, :, :target_h, :]
                validation_image = torch.cat([validation_image, torch.flip(validation_image, [3])], 3)[:, :, :, :target_w]
                #validation_image = pad_image_to_size(validation_image, target_h, target_w)


                ref_image = torch.cat([ref_image, torch.flip(ref_image, [2])], 2)[:, :, :target_h, :]
                ref_image = torch.cat([ref_image, torch.flip(ref_image, [3])], 3)[:, :, :, :target_w]
                #ref_image = pad_image_to_size(ref_image, target_h, target_w)


            pad_h, pad_w = validation_image.shape[2], validation_image.shape[3]
            cat_image = torch.cat([validation_image, ref_image])

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    image = pipeline(
                            [validation_prompt,ref_validation_prompt], cat_image, num_inference_steps=args.num_inference_steps, generator=generator, height=pad_h, width=pad_w,
                            guidance_scale=args.guidance_scale, negative_prompt=[negative_prompt,negative_prompt], conditioning_scale=args.conditioning_scale,
                            start_point=args.start_point, ram_encoder_hidden_states=torch.cat([ram_encoder_hidden_states,ref_ram_encoder_hidden_states]),
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            args=args,
                        ).images
                
                
                # 可以仿照之后的颜色平衡的代码，实现PIL返回tensor的转换
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image.cpu(), validation_image.cpu())
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image.cpu(), cat_image)
                
                
                inp_image=to_image(image[0][:,:height,:width].clamp_(0.0, 1.0))
                inp_image = inp_image.resize((ori_width*rscale, ori_height*rscale))
                image = inp_image
                # if resize_flag: 
                #     image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default='/home/tiger/gh/cache/SeeSR/seesr')
    parser.add_argument("--ram_ft_path", type=str, default='/home/tiger/gh/cache/SeeSR/DAPE.pth')
    parser.add_argument("--pretrained_model_path", type=str, default='/home/tiger/gh/cache/stable-diffusion-2-base')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default='/home/tiger/gh/dataset/WR-SR-testset/Real_Deg/LR')
    parser.add_argument("--output_dir", type=str, default='/home/tiger/gh/dataset/results/Real_Deg/seeSR/wr-sr')
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=2048)#224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=2048)#1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=2048)#96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    args = parser.parse_args()
    main(args)



