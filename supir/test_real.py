from ast import mod
import os
from turtle import shape
import numpy as np

# os.environ["HF_DATASETS_CACHE"] = "/home/tiger/gh/cache/"
# os.environ["HF_HOME"] = "/home/tiger/gh/cache/"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/tiger/gh/cache/"
# os.environ["TRANSFORMERS_CACHE"] = "/home/tiger/gh/cache/"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype, mod_crop
from PIL import Image
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import os
from sgm.modules.attention import MemoryEfficientCrossAttention
# os.chdir(os.path.dirname(__file__))
from torch.nn.functional import interpolate

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default='/home/tiger/gh/dataset/CUFED5/Real_Deg/LR')
parser.add_argument("--save_dir", type=str, default='/home/tiger/gh/dataset/results/Real_Deg/SUPIR/cufed')
parser.add_argument("--upscale", type=int, default=4)
parser.add_argument("--refir_scale", type=float, default=1.0)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--edm_steps", type=int, default=50)
parser.add_argument("--s_stage1", type=int, default=-1)
parser.add_argument("--s_churn", type=int, default=5)
parser.add_argument("--s_noise", type=float, default=1.003)
parser.add_argument("--s_cfg", type=float, default=7.5)
parser.add_argument("--s_stage2", type=float, default=1.)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--a_prompt", type=str,
                    default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                            'hyper sharpness, perfect without deformations.')
parser.add_argument("--n_prompt", type=str,
                    default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, over-smooth')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
parser.add_argument("--linear_CFG", action='store_true', default=True)
parser.add_argument("--linear_s_stage2", action='store_true', default=False)
parser.add_argument("--spt_linear_CFG", type=float, default=4.0)
parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
parser.add_argument("--ae_dtype", type=str, default="fp32", choices=['fp32', 'bf16'])
parser.add_argument("--diff_dtype", type=str, default="fp32", choices=['fp32', 'fp16', 'bf16'])

parser.add_argument("--no_llava", action='store_true', default=False)

parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
args = parser.parse_args()
print(args)
use_llava = not args.no_llava

# load SUPIR
model = create_SUPIR_model('/home/tiger/gh/SUPIR/options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
if args.loading_half_params:
    model = model.half()
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
model.ae_dtype = convert_dtype(args.ae_dtype)
model.model.dtype = convert_dtype(args.diff_dtype) # MemoryEfficientCrossAttention
model = model.to(SUPIR_device)

# set refir_scale here
for module in model.modules():
    if isinstance(module, MemoryEfficientCrossAttention):
        module.refir_scale = args.refir_scale


# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None


excpt_list = ['.DS_Store'] 
os.makedirs(args.save_dir, exist_ok=True)

for img_pth in sorted(os.listdir(args.img_dir)):
    print(img_pth)
    # caption from LLaVA
    captions = []
    img_name = os.path.splitext(img_pth)[0].split('_')[0]
    #  down-smaple to LQ  MANUAL DOWN-SAMPLING
    LQ_img = Image.open(os.path.join(args.img_dir, img_pth))
    h_LQ, w_LQ = LQ_img.height, LQ_img.width

    if use_llava:
        # Pre-denoise for LLaVA
        LQ_img_512 = PIL2Tensor(LQ_img, upsacle=args.upscale, min_size=args.min_size, fix_resize=512)[0]
        LQ_img_512 = LQ_img_512.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
        clean_imgs = model.batchify_denoise(LQ_img_512)
        clean_PIL_img = Tensor2PIL(clean_imgs[0], h_LQ, w_LQ) 
        captions += llava_agent.gen_image_caption([clean_PIL_img])
    else:
        captions += ['']

    scale = args.upscale
    LQ_img = PIL2Tensor(LQ_img, upsacle=scale, min_size=128)[0]  
    LQ_img = LQ_img.unsqueeze(0)[:, :3, :, :]
    LQ_h_resize, LQ_w_resize = LQ_img.shape[2:]


    if 'CUFED5' in args.img_dir: 
        ref_img = Image.open(os.path.join(args.img_dir.replace('Real_Deg/LR', 'ref'), img_name+'_1.png'))
    elif 'WR' in args.img_dir:
        ref_img = Image.open(os.path.join(args.img_dir.replace('Real_Deg/LR', 'ref'), img_name+'_ref'+'.png'))
    else:
        raise NotImplementedError

    if use_llava:
        captions += llava_agent.gen_image_caption([ref_img])  
    else:
        captions += ['']

    ref_img, h1, w1 = PIL2Tensor(ref_img, upsacle=1, min_size=128) 
    ref_img = ref_img.unsqueeze(0)[:, :3, :, :]
    ref_img = ref_img[:, :, :int(2 * LQ_h_resize / 64) * 64, :int(2 * LQ_w_resize / 64.0) * 64] 
    ref_img_h, ref_img_w = ref_img.shape[2:]

    if ref_img_h != LQ_h_resize or ref_img_w != LQ_w_resize:
        use_padding = True
        target_h = max(LQ_h_resize, ref_img_h)
        target_w = max(LQ_w_resize, ref_img_w)
        LQ_img = torch.cat([LQ_img, torch.flip(LQ_img, [2])], 2)[:, :, :target_h, :]
        LQ_img = torch.cat([LQ_img, torch.flip(LQ_img, [3])], 3)[:, :, :, :target_w]

        ref_img = torch.cat([ref_img, torch.flip(ref_img, [2])], 2)[:, :, :target_h, :]
        ref_img = torch.cat([ref_img, torch.flip(ref_img, [3])], 3)[:, :, :, :target_w]

    pad_h, pad_w = LQ_img.shape[2], LQ_img.shape[3]

    cat_img = torch.cat([LQ_img, ref_img]).to(SUPIR_device)

    print(captions)
    # step 3: Diffusion Process
    samples = model.batchify_sample(cat_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1,
                                    s_churn=args.s_churn,
                                    s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2,
                                    seed=args.seed,
                                    num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt,
                                    color_fix_type=args.color_fix_type,
                                    use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                    cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)
    # save
    for _i, sample in enumerate(samples):
        Tensor2PIL(sample[:, :LQ_h_resize, :LQ_w_resize], scale*h_LQ, scale*w_LQ).save(f'{args.save_dir}/{img_name}.png')
        break

