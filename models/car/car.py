import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from .utils import load_img
from .EDSR.edsr import EDSR
from .modules import DSN
from .adaptive_gridsampler.gridsampler import Downsampler

class CAR:
    def __init__(
        self,
        SCALE=2, 
        model_dir="./models",
        gpu=0,
    ):
        # Save some params
        self.model_dir = model_dir
        self.gpu = gpu

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

    def run_upscale(self, img_file):
        print(f"Running CAR upscale network over {img_file}")
        img = load_img(img_file)
        reconstructed_img = self.upscale_net(img)
        return reconstructed_img

    def run_downscale(self, img_file):
        print(f"Running CAR downscale network over {img_file}")
        img = load_img(img_file)
        kernels, offsets_h, offsets_v = self.kernel_generation_net(img)
        downscaled_img = self.downsampler_net(img, kernels, offsets_h, offsets_v, OFFSET_UNIT)
        return downscaled_img

    def pytensor2pil(self, img):
        img = torch.clamp(img, 0, 1) * 255
        img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
        img = np.uint8(img)
        img = img[0, ...].squeeze()
        img = Image.fromarray(img)
        return img

    def pilsaveimage(self, img, path):
        img.save(path)