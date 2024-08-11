# evaluate the restored images with IQA
# PSNR, SSIM, LPIPS are given as example, you can add more IQA in this file
import cv2
import argparse, os, sys, glob
import logging
from datetime import datetime

import os
os.chdir(os.path.dirname(__file__))
import pyiqa

from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k in opt:
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """from BasicSR
    Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--img_path",
		nargs="+",
		help="path to the input image",
		default='/home/tiger/gh/dataset/results/Real_Deg/seeSR/realPhoto_nomask/sample00',
		#default='/home/tiger/gh/dataset/results/Real_Deg/SUPIR/realPhoto',
	)

	# parser.add_argument(
	# 	"--init-imgs-names",
	# 	nargs="+",
	# 	help="name of the input image",
	# 	default=['****-0', '****-1', '****-2', '****-3', '****-4', '****-5', '****-6', '****-7', '****-8', '****-9'
	# 	],
	# 	)

	# parser.add_argument(
	# 	"--gt_path",
	# 	nargs="+",
	# 	help="path to the gt image, you need to add the paths of gt folders corresponding to init-imgs",
	# 	#default='/home/tiger/gh/dataset/CUFED5/Real_Deg/HR'
	# 	default='/home/tiger/gh/dataset/CUFED5/Real_Deg/HR',
	# )
	
	parser.add_argument(
		"--log",
		type=str,
		nargs="?",
		help="path to the log",
		default='./iqa_results')

	parser.add_argument(
		"--log-name",
		type=str,
		nargs="?",
		help="name of your log",
		default='test',
	)

	parser.add_argument(
		"--num_img",
		type=int,
		nargs="?",
		help="the number of images evaluated in the folder; 0: all the images are evaludated.",
		default=0,
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	os.makedirs(opt.log, exist_ok=True)
	# init logger
	setup_logger('base', opt.log, 'test_' + opt.log_name, level=logging.INFO,
                  screen=True, tofile=True)
	logger = logging.getLogger('base')
	logger.info(opt)

	# init metrics: you can add more metrics here
	iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_lpips = pyiqa.create_metric('lpips', device=device)
	iqa_niqe =   pyiqa.create_metric('niqe', device=device)
	iqa_musiq = pyiqa.create_metric('musiq-koniq',device=device)
	iqa_clipiqa = pyiqa.create_metric('clipiqa',device=device)
	#iqa_maniqa= pyiqa.create_metric('maniqa',device=device)

	iqa_fid   = pyiqa.create_metric('fid',device=device)

	# record metrics
	metrics = { 'niqe': [], 'musiq': [], 'clipiqa': []}#, 'maniqa': []}


	for img_name in sorted(os.listdir(opt.img_path)):

		input_sr_path = os.path.join(opt.img_path,img_name)
		

		input_sr_img = cv2.imread(input_sr_path, cv2.IMREAD_COLOR)
		sr = img2tensor(input_sr_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

		
	
		# PSNR: convert the ycbcr to calculate
	
		sr = sr[..., 4:-4, 4:-4] / 255.

		

		niqe_now = iqa_niqe(sr).item()
		metrics['niqe'].append(niqe_now)

		musiq_now = iqa_musiq(sr).item()
		metrics['musiq'].append(musiq_now)

		clipiqa_now = iqa_clipiqa(sr).item()
		metrics['clipiqa'].append(clipiqa_now)

	

		# maniqa_now = iqa_maniqa(sr).item()
		# metrics['maniqa'].append(maniqa_now)

	

	for key,value in metrics.items():
		logger.info('{}:{:.6f}'.format(key,sum(value)/len(value)))
		
	# save metrics
	# npy_path = os.path.join(opt.log, 'test_' + opt.log_name + '_npy')
	# os.makedirs(npy_path, exist_ok=True)
	# np.save(npy_path + '/' + 'save.npy', metrics)


if __name__ == '__main__':
	main()