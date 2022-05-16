import argparse
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from PIL import Image
from torchvision.transforms import InterpolationMode

from .srgan_pytorch import models as models
from .srgan_pytorch.utils.common import configure
from .srgan_pytorch.utils.common import create_folder
from .srgan_pytorch.utils.estimate import iqa
from .srgan_pytorch.utils.transform import process_image

import numpy as np
import kornia
import torchvision.transforms as transforms

import sys
sys.path.append("../..")
from custom_transforms import blur_image, rescale_image

class SRGAN:
    def __init__(
        self,
        arch="srgan_2x2",
        model_path="./weights/PSNR.pth",
        # pretrained=False,
        seed=666,
        gpu=0, # set None to use cpu
    ):
        # Save some params
        self.arch = arch
        self.model_path = model_path
        self.gpu = gpu
        self.seed = seed

        # Seed and / cuda Config
        self.model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

        # Load model
        self.model = models.__dict__[arch]()
        ''' 
        if pretrained:
            self.model = models.__dict__[arch](pretrained=True) # this crashes for latest implementations (model urls down)
        '''
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            self.model = self.model.cuda(gpu)

        # Set Eval mode
        self.model.eval()

        cudnn.benchmark = True
    
    def run_sr_mod(self, img_file, rescale = False, zoom = 3, blur = False):
        lr = Image.open(img_file)

        if blur is True:
            lr = blur_image(lr, zoom)

        if rescale is True:
            lr = rescale_image(lr, zoom)

        lr = process_image(lr, self.gpu) # convert to pytorch tensor
        with torch.no_grad():
            sr = self.model(lr)
        return sr

    def run_sr(self, img_file):
        lr = Image.open(img_file)
        lr = process_image(lr, self.gpu) # convert to pytorch tensor
        with torch.no_grad():
            sr = self.model(lr)
        return sr

    def run_sr_resized(self, img_file, scale = 2):
        lr = Image.open(img_file)
        width, height = lr.size
        pscale = float(1.0 / float(scale))
        lr = resize_pil_img(lr, int(pscale*width), int(pscale*height))
        lr = process_image(lr, self.gpu) # convert to pytorch tensor
        with torch.no_grad():
            sr = self.model(lr)
        return sr

def load_srgan_model(model_fn = "./weights/PSNR.pth", arch = "srgan_2x2"):
    device = torch.device('cuda')
    model = models.__dict__[arch]()
    model.load_state_dict(torch.load(model_fn))
    model = model.to(device)
    model.eval()
    return model

def resize_pil_img(img, target_width, target_height):
    return img.resize((target_width,target_height), Image.ANTIALIAS)
def resize_torch_img(img, target_width, target_height):
    n_channels = img.shape[2]
    return torch.nn.functional.interpolate(img, size=(target_height, target_width, n_channels), mode='bicubic')