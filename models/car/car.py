import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

#from .utils import load_img
from .EDSR.edsr import EDSR
from .modules import DSN
#from .adaptive_gridsampler.gridsampler import Downsampler

import numpy as np
import kornia
import torchvision.transforms as transforms
def blur_image(image, scale):
    img_type = type(image).__name__
    if "Tensor" not in img_type:
        img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    else:
        img_tensor = image
    sigma = 0.5 * scale if scale is not None else 7.0
    kernel_size = int(np.ceil(sigma * 3 + 4))
    kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))   
    image_blur = kornia.filters.filter2d(img_tensor, kernel_tensor[None])
    if "Tensor" not in img_type:
        image = transforms.ToPILImage()(image_blur.squeeze_(0))
    else:
        image = image_blur
    return image

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
        #self.downsampler_net = Downsampler(self.SCALE, self.KSIZE).cuda(gpu)
        self.upscale_net = EDSR(32, 256, scale=self.SCALE).cuda(gpu)

        self.kernel_generation_net = nn.DataParallel(self.kernel_generation_net, [0])
        #self.downsampler_net = nn.DataParallel(self.downsampler_net, [0])
        self.upscale_net = nn.DataParallel(self.upscale_net, [0])

        self.kernel_generation_net.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
        self.upscale_net.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(SCALE), 'usn.pth')))
        torch.set_grad_enabled(False)

        # Set nets to eval
        self.kernel_generation_net.eval()
        #self.downsampler_net.eval()
        self.upscale_net.eval()
    
    def run_upscale_mod(self, img_file, scale = None, zoom = None, blur = False):
        if scale is not None:
            img = load_img_resized(img_file, scale)
        else:
            img = load_img(img_file)
        if blur is True:
            if scale is None:
                img = blur_image(img, zoom)
            else:
                img = blur_image(img, scale)
        reconstructed_img = self.upscale_net(img)
        return reconstructed_img

    def run_upscale(self, img_file):
        img = load_img(img_file)
        reconstructed_img = self.upscale_net(img)
        return reconstructed_img
    '''
    def run_downscale(self, img_file):
        print(f"Running CAR downscale network over {img_file}")
        img = load_img(img_file)
        kernels, offsets_h, offsets_v = self.kernel_generation_net(img)
        downscaled_img = self.downsampler_net(img, kernels, offsets_h, offsets_v, OFFSET_UNIT)
        return downscaled_img
    '''
    def run_upscale_resized(self, img_file, scale = 2):
        img = load_img_resized(img_file, scale)
        reconstructed_img = self.upscale_net(img)
        return reconstructed_img

def pytensor2pil(img):
    img = torch.clamp(img, 0, 1) * 255
    img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
    img = np.uint8(img)
    img = img[0, ...].squeeze()
    img = Image.fromarray(img)
    return img

def pilsaveimage(img, path):
    img.save(path)

def load_car_model(model_fn = "./models/2x/usn.pth", SCALE = 2):
    device = torch.device('cuda')
    model = EDSR(32, 256, scale=self.args.SCALE).cuda()
    model = nn.DataParallel(model, [0])
    model.load_state_dict(torch.load(model_fn)) #usn.pth
    model.eval()
    model = model.to(device)
    return model

def load_img_resized(img_file, scale = 2):
    img = Image.open(img_file).convert('RGB')
    width, height = img.size
    pscale = float(1.0 / float(scale))
    img = resize_pil_img(img, int(pscale*width), int(pscale*height))
    img = np.array(img)
    h, w, _ = img.shape
    #img = img[:h // 8 * 8, :w // 8 * 8, :]
    img = np.array(img) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    return img

def load_img(img_file):
    img = Image.open(img_file).convert('RGB')
    img = np.array(img)
    h, w, _ = img.shape
    img = img[:h // 8 * 8, :w // 8 * 8, :]
    img = np.array(img) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    return img

def resize_pil_img(img, target_width, target_height):
    return img.resize((target_width,target_height), Image.ANTIALIAS)
def resize_torch_img(img, target_width, target_height):
    n_channels = img.shape[2]
    return torch.nn.functional.interpolate(img, size=(target_height, target_width, n_channels), mode='bicubic')