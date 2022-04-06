import os, argparse
import numpy as np

import torch
import torch.nn as nn

import utils
from EDSR.edsr import EDSR
from modules import DSN
from adaptive_gridsampler.gridsampler import Downsampler

class CAR:
    def __init__(
        self,
        SCALE=2, 
        model_dir="./models",
        gpu=0,
    ):
        # Load Downsampler kernel characteristics
        self.SCALE = SCALE
        self.KSIZE = 3 * SCALE + 1
        self.OFFSET_UNIT = SCALE

        # Load nets
        self.kernel_generation_net = DSN(k_size=self.KSIZE, scale=self.SCALE).cuda(gpu)
        self.downsampler_net = Downsampler(self.SCALE, self.KSIZE).cuda(gpu)
        self.upscale_net = EDSR(32, 256, scale=self.SCALE).cuda(gpu)

        self.kernel_generation_net = nn.DataParallel(self.kernel_generation_net, [0])
        self.downsampler_net = nn.DataParallel(self.downsampler_net, [0])
        self.upscale_net = nn.DataParallel(self.upscale_net, [0])

        self.kernel_generation_net.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
        self.upscale_net.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(SCALE), 'usn.pth')))
        torch.set_grad_enabled(False)

        # Set nets to eval
        self.kernel_generation_net.eval()
        self.downsampler_net.eval()
        self.upscale_net.eval()

    def run_upscale(img_file):
        img = utils.load_img(img_file)
        reconstructed_img = upscale_net(img)
        return reconstructed_img

    def run_downscale(img_file):
        img = utils.load_img(img_file)
        kernels, offsets_h, offsets_v = kernel_generation_net(img)
        downscaled_img = downsampler_net(img, kernels, offsets_h, offsets_v, OFFSET_UNIT)
        return downscaled_img
