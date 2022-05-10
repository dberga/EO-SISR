import os
import tempfile
import sys
import json
import cv2
import piq
import torch
import yaml
import signal
import time
import math
import shutil

import numpy as np
import PIL.Image as pil_image

from glob import glob
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, List,Union,Tuple
from iq_tool_box.datasets import DSModifier
from iq_tool_box.metrics import Metric
from iq_tool_box.experiments import ExperimentInfo

from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

import torch.backends.cudnn as cudnn

from lowresgen import LRSimulator
from torchvision import transforms
import kornia

# MSRN
from msrn.msrn import load_msrn_model, process_file_msrn

# FSRCNN
from utils.utils_fsrcnn import convert_ycbcr_to_rgb, preprocess
from models.model_fsrcnn import FSRCNN

# LIIF
from datasets.liif import datasets as datasets_liif
from utils import utils_liif
from models.liif import models as models_liif

# ESRGAN
import esrgan
from models.esrgan import RRDBNet_arch as arch

# CAR
from models.car.car import CAR, pytensor2pil, pilsaveimage, load_car_model, resize_pil_img

# SRGAN
from models.srgan.srgan import SRGAN, load_srgan_model

# Metrics
from swd import SlicedWassersteinDistance

###### SOME TRANSFORMS

def blur_image(image, scale):
    init_type = type(image).__name__
    if init_type == "ndarray":
        image = pil_image.fromarray(image)
    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    sigma = 0.5 * scale if scale is not None else 7.0
    kernel_size = math.ceil(sigma * 3 + 4)
    kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
    image_blur = kornia.filters.filter2d(img_tensor, kernel_tensor[None])
    image = transforms.ToPILImage()(image_blur.squeeze_(0))
    if init_type == "ndarray":
        image = np.array(image)
    return image
def rescale_image(image, scale, interpolation=pil_image.BICUBIC):
    if scale is None or scale == 1.0:
        return image
    if type(image).__name__ == "ndarray":
        image = pil_image.fromarray(image)
        image = image.resize((image.width // scale, image.height // scale), resample=interpolation)
        image = np.array(image)
    else:
        image = image.resize((image.width // scale, image.height // scale), resample=interpolation)
    return image
def rescale_image_exact(image, width, height, interpolation=pil_image.BICUBIC):
    if scale is None or scale == 1.0:
        return image
    if type(image).__name__ == "ndarray":
        image = pil_image.fromarray(image)
        image = image.resize((width, height), resample=interpolation)
        image = np.array(image)
    else:
        image = image.resize((width, height), resample=interpolation)
    return image

###### MODEL READOUT
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()

class Args():
    def __init__(self):
        pass
    
class ModelConfS3Loader():
    
    def __init__(
        self,
        model_fn      = "single_frame_sr/SISR_MSRN_X2_BICUBIC.pth",
        config_fn_lst = [],
        bucket_name   = "image-quality-framework",
        algo          = "FSRCNN",
        zoom          = 3,
        tmpdir        = "checkpoints/", #tempfile.TemporaryDirectory()
        kwargs        = {},
    ):
        
        self.fn_dict = {
            f"conf{enu}":fn
            for enu,fn in enumerate(config_fn_lst)
        }
        
        self.fn_dict["model"]   =  model_fn
        self.model_fn           =  model_fn
        self.config_fn_lst      =  config_fn_lst
        self.bucket_name        =  bucket_name
        self.algo               =  algo
        self.zoom               =  zoom
        self.tmpdir             = tmpdir
        self.kwargs             = kwargs
        
    def load_ai_model_and_stuff(self) -> List[Any]:
        
        # Rename to full bucket subdirs...
        # First file is the model
        
        print( self.fn_dict["model"] )

        fn_lst = [
            os.path.join("iq-sisr-use-case/models/weights",self.fn_dict["model"])
        ] + [
            os.path.join("iq-sisr-use-case/models/config",fn)
            for fn in self.config_fn_lst
        ]
        tmpdirname = self.tmpdir
        os.makedirs(tmpdirname, exist_ok = True)
        #with tempfile as tmpdirname:
        
        print( self.fn_dict )
        fn_dict_aux = {}
        for k in self.fn_dict:

            bucket_fn = self.fn_dict[k]

            aux_k = k+"_"+self.fn_dict[k].replace("/","_")
            local_fn = os.path.join(
                tmpdirname , aux_k
            )
            fn_dict_aux[k] = local_fn
            
            kind = ('weights' if k=='model' else 'config')
            
            url = f"https://{self.bucket_name}.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/models/{kind}/{bucket_fn}"

            print( url , ' ' , local_fn )
            
            if not os.path.exists(local_fn):
                os.system( f"wget {url} -O {local_fn}" )

            if not os.path.exists( local_fn ):
                print( 'AWS model file not found' )
                raise

        if self.algo=='FSRCNN':

            args = self._load_args(fn_dict_aux["conf0"])
            model = self._load_model_fsrcnn(fn_dict_aux["model"], args )

        elif self.algo=='LIIF':
            
            args = self._load_args(fn_dict_aux["conf0"])
            model = self._load_model_liif(fn_dict_aux["model"], args, fn_dict_aux["conf1"])
            
        elif self.algo=='MSRN':
            
            args = None
            model = self._load_model_msrn(fn_dict_aux["model"],n_scale=self.zoom)

        elif self.algo=='ESRGAN':
            
            args = esrgan.get_args( self.zoom )
            model = self._load_model_esrgan(fn_dict_aux["model"])

        elif self.algo=='CAR':
            args = Args()
            args.SCALE = self.kwargs["SCALE"]
            args.zoom = self.kwargs["zoom"]
            model = self._load_model_car(fn_dict_aux["model"], args.SCALE)

        elif self.algo=='SRGAN':
            args = Args()
            args.arch = self.kwargs["arch"]
            args.zoom = self.kwargs["zoom"]
            model = self._load_model_srgan(fn_dict_aux["model"], args.arch )

        else:
            print(f"Error: unknown algo: {self.algo}")
            raise

        return model , args
    
    def _load_args( self, config_fn: str ) -> Any:
        """Load Args"""

        args = Args()
        
        with open(config_fn) as jsonfile:
            args_dict = json.load(jsonfile)

        for k in args_dict:
            setattr( args, k, args_dict[k] )

        return args
    
    def _load_model_fsrcnn( self, model_fn: str,args: Any ) -> Any:
        """Load Model"""

        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = FSRCNN(scale_factor=args.scale).to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(model_fn, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()

        return model
    
    def _load_model_liif(self,model_fn: str,args: Any,yml_fn: str) -> Any:
        """Load Model"""
        
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        
        with open(yml_fn, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        spec = config['test_dataset']
        # Save config and spec in self
        self.config = config
        self.spec = spec
        
        model_spec = torch.load(model_fn)['model']
        model = models_liif.make(model_spec, load_sd=True).cuda()
        
        model.eval()
        
        return model
    
    def _load_model_msrn(self,model_fn: str,n_scale:int = 3) -> Any:
        """Load MSRN Model"""
        return load_msrn_model(model_fn,n_scale=n_scale)

    def _load_model_esrgan(self,model_fn: str) -> Any:
        """Load ESRGAN Model"""
        device = torch.device('cuda')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        #model.load_state_dict(torch.load(args.model_path), strict=True)
        weights = torch.load(model_fn)
        model.load_state_dict(weights['params'])
        model.eval()
        model = model.to(device)
        return model

    def _load_model_car(self,model_fn: str, SCALE: int = 2) -> Any:
        """Load CAR Model"""
        return load_car_model(model_fn, SCALE)

    def _load_model_srgan(self, model_fn: str, arch: str) -> Any:
        """Load SRGAN Model"""
        return load_srgan_model(model_fn, arch)


#########################
# Custom IQF
#########################

class DSModifierLIIF(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"LIIF",
            "config0": "LIIF_config.json",
            "config1": "test_liif.yaml",
            "model": "liif_UCMerced/epoch-best.pth",
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        params['algo'] = 'LIIF'
        algo           = params['algo']
        
        subname = algo + '_' + \
                    os.path.splitext(params['config0'])[0]+ \
                    '_' + os.path.splitext(params['config1'])[0]+ \
                    '_'+os.path.splitext(params['model'])[0].replace('/','-')
        
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [params['config0'],params['config1']],
                bucket_name   = "image-quality-framework",
                algo          = "LIIF"
        )
        
        model,args = model_conf.load_ai_model_and_stuff()
        
        self.spec   =  model_conf.spec
        self.args   =  args
        self.model  =  model
        if "blur" in self.params.keys():
            self.blur           = params['blur']
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
        else:
            self.resize_preprocess = True
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
        else:
            self.resize_postprocess = False
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path
        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        data_norm, eval_type, eval_bsize = None,None, None
        
        spec = self.spec
        
        spec['batch_size'] = 1
        
        spec['dataset'] = {
            'name': 'image-folder',
            'args': {
                'root_path': data_input
            }
        }
        spec['wrapper']['args']['blur'] = self.blur
        spec['wrapper']['args']['resize'] = self.resize_preprocess

        dataset = datasets_liif.make(spec['dataset'])
        dataset = datasets_liif.make(spec['wrapper'], args={'dataset': dataset})
        loader = DataLoader(dataset, batch_size=spec['batch_size'],shuffle=False,
                           num_workers=1, pin_memory=False ) #pin_memory=True
        
        if data_norm is None:
        
            data_norm = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        gt_sub  = torch.FloatTensor(data_norm['gt']['sub']).view(1, 1, -1).cuda()
        gt_div  = torch.FloatTensor(data_norm['gt']['div']).view(1, 1, -1).cuda()

        if eval_type is None:
            metric_fn = utils_liif.calc_psnr
        elif eval_type.startswith('div2k'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(utils_liif.calc_psnr, dataset='div2k', scale=scale)
        elif eval_type.startswith('benchmark'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(utils_liif.calc_psnr, dataset='benchmark', scale=scale)
        else:
            raise NotImplementedError

        val_res  = utils_liif.Averager()
        val_ssim = utils_liif.Averager() #test

        #pbar = tqdm(loader, leave=False, desc='val')
        image_file_lst = sorted( glob( os.path.join(data_input,'*.tif') ) )
        count = 0
        for enu,batch in enumerate(loader):
            
            image_file = image_file_lst[enu]
            orig_w, orig_h = pil_image.open(image_file).size
            try:
                imgp = self._mod_img( batch, inp_sub, inp_div, eval_bsize,gt_div , gt_sub )
                print(f"Running {self.name} over {os.path.basename(image_file)}")
                output = pil_image.fromarray((imgp*255).astype(np.uint8))
                if self.resize_postprocess is True:
                    rec_img = rescale_image_exact(output, orig_w, orig_h)
                output.save( os.path.join(dst, os.path.basename(image_file)) )
                #cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except Exception as e:
                print(e)
            
        return input_name

    def _mod_img(self, batch: Any, inp_sub: Any, inp_div: Any, eval_bsize: Any, gt_div: Any, gt_sub: Any) -> None:

        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div

        if eval_bsize is None:
            with torch.no_grad():
                pred = self.model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(self.model, inp,
                                   batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        ih, iw = batch['inp'].shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape) \
            .permute(0, 1, 2, 3).contiguous()
        rec_img = pred.detach().cpu().numpy().squeeze()


        return rec_img

class DSModifierFSRCNN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"FSRCNN",
            "config": "test.json",
            "model": "FSRCNN_1to033_x3_noblur/best.pth",
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        
        params['algo'] = 'FSRCNN'
        algo           = params['algo']
        
        subname = algo + '_' + os.path.splitext(params['config'])[0]+'_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        if "blur" in self.params.keys():
            self.blur           = params['blur']
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
        else:
            self.resize_preprocess = True
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
        else:
            self.resize_postprocess = False
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [params['config']],
                bucket_name   = "image-quality-framework",
                algo          = "FSRCNN"
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.args = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """
        Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):

            try:
                imgp = self._mod_img( image_file )
                print(f"Running {self.name} over {os.path.basename(image_file)}")
                output = pil_image.fromarray(imgp)
                output.save(os.path.join(dst, os.path.basename(image_file)))
                #cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except Exception as e:
                print(e)
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        args = self.args
        model = self.model

        image = pil_image.open(image_file).convert('RGB')
        orig_w, orig_h = image.size
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        # AFEGIT PER FER EL BLUR
        if self.blur is True:
            image = blur_image(image, args.scale)
        ########
        if self.resize_preprocess is True:
            hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            lr = rescale_image(hr, args.scale)
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC) 
        else:
            hr = image.copy()
            lr = image.copy()
            #bicubic = image.copy()
            bicubic = rescale_image_exact(image.copy(),orig_w*args.scale, orig_h*args.scale)
        lr, _ = preprocess(lr, self.device)
        _, ycbcr = preprocess(bicubic, self.device) # convert_rgb_to_ycbcr
        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        if self.resize_postprocess is True:
            output = rescale_image_exact(output, orig_w, orig_h)
        return output

class DSModifierMSRN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"MSRN",
            "zoom": 2,
            "model": "MSRN/SISR_MSRN_X2_BICUBIC.pth",
            "compress": False,
            "add_noise": False,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        
        params['algo'] = 'MSRN'
        algo           = params['algo']
        subname = algo + '_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        if "compress" in self.params.keys():
            self.compress = self.params["compress"]
            if self.compress is True:
                self.name += "_compress"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.compress = False
        if "add_noise" in self.params.keys():
            self.add_noise = self.params["add_noise"]
            if self.add_noise is True:
                self.name += "_addnoise"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.add_noise = None
        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_postprocess = False
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [],
                bucket_name   = "image-quality-framework",
                algo          = "MSRN",
                zoom          = self.params['zoom'],
        )
        
        self.model,_ = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):
            
            #signal.signal(signal.SIGALRM, alarm_handler)
            #signal.alarm(5)
            
            try:
                imgp = self._mod_img( image_file )
                print(f"Running {self.name} over {os.path.basename(image_file)}")
                cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except TimeOutException as ex:
                print(ex)
            except Exception as e:
                print(e)
            
            signal.alarm(0)
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        zoom       = self.params["zoom"]
        loaded     = cv2.imread(image_file, -1)
        orig_w, orig_h = pil_image.open(image_file).size
        wind_size  = loaded.shape[1]
        gpu_device = "0"
        res_output = 1/zoom # inria resolution

        if self.resize_preprocess is True:
            loaded = rescale_image(loaded, zoom)

        out_win = loaded.shape[-2] if self.resize_postprocess is True or self.resize_preprocess is False else loaded.shape[-2]*zoom

        rec_img = process_file_msrn(
            loaded,
            self.model,
            compress=self.compress, # True,
            out_win = out_win,
            wind_size=wind_size+10, stride=wind_size+10,
            scale=zoom,
            batch_size=1,
            padding=5,
            add_noise=self.add_noise,# None,
            blur=self.blur, #True
        )
        if self.resize_postprocess is True:
            rec_img = rescale_image_exact(rec_img, orig_w, orig_h)
        return rec_img


class DSModifierESRGAN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"ESRGAN",
            "zoom": 2,
            "model": "./ESRGAN_1to033_x3_blur/net_g_latest.pth",
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        
        params['algo']              = 'ESRGAN'
        algo                        = params['algo']
        subname                     = algo + '_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name                   = f"sisr+{subname}"
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.device                 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.params.update({"modifier": "{}".format(self._get_name())})
        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_postprocess = False

        model_conf = ModelConfS3Loader(
            model_fn      = params['model'],
            config_fn_lst = [],
            bucket_name   = "image-quality-framework",
            algo          = "ESRGAN"
        )
        
        self.model , self.args = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):
            
            imgp = self._mod_img( image_file )
            print(f"Running {self.name} over {os.path.basename(image_file)}")
            
            cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:

        img = esrgan.generate_lowres( image_file , scale=self.args.zoom, blur=self.blur, resize=self.resize_preprocess )
        orig_w, orig_h = pil_image.open(image_file).size
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)
        
        with torch.no_grad():

            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        
        rec_img = (output*255).astype(np.uint8)
        
        if self.resize_postprocess is True:
            rec_img = rescale_image_exact(rec_img, orig_w, orig_h)
        return rec_img

class DSModifierCAR(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "SCALE": 2,
            "model_dir": "./models/car/models",
            "gpu": 0,
            "zoom": 2,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        self.params = params
        self.ds_modifier = ds_modifier
        '''
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [],
                bucket_name   = "image-quality-framework",
                algo          = "CAR",
                zoom          = params['SCALE'],
        )
        self.model,_ = model_conf.load_ai_model_and_stuff()
        ''' # save in self.params["model_dir"]

        self.CAR = CAR(SCALE=self.params["SCALE"],model_dir=self.params["model_dir"],gpu=self.params["gpu"])
        self.name = f"CAR_scale{params['SCALE']}" # _modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_postprocess = False

    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        print(f'{self.name} For each image file in <{data_input}>...')
        for data_file in os.listdir(data_input):
            file_path = os.path.join(data_input, data_file)
            dst_file = os.path.join(dst, data_file)
            print(f"Running {self.name} over {os.path.basename(file_path)}")
            rec_img = self._mod_img(file_path)
            rec_img = pytensor2pil(rec_img)
            # check image size, resize if reconstructed image is not the same
            rec_w, rec_h = rec_img.size
            orig_w, orig_h = pil_image.open(file_path).size
            if rec_w != orig_w or rec_h != orig_h or self.resize_postprocess is True:
                rec_img = rescale_image_exact(rec_img, orig_w, orig_h) # downscale in case of 4x
            
            pilsaveimage(rec_img,dst_file)
            
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        if self.resize_preprocess is True:
            rec_img = self.CAR.run_upscale_mod(image_file, self.params['zoom'], self.params['zoom'], self.blur)
            #rec_img = self.CAR.run_upscale_resized(image_file, self.params['zoom'])
        else:
            rec_img = self.CAR.run_upscale_mod(image_file, None, self.params['zoom'], self.blur)
            #rec_img = self.CAR.run_upscale(image_file)
        return rec_img

class DSModifierSRGAN(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "arch": "srgan_2x2",
            "model_path": "./models/srgan/weights/PSNR.pth",
            "gpu": 0,
            "seed": 666,
            "zoom": 2,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        self.params = params
        self.ds_modifier = ds_modifier

        '''
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [],
                bucket_name   = "image-quality-framework",
                algo          = "SRGAN",
                kwargs          = {"arch": params['arch'], "zoom": params['zoom']},
        )
        self.model,_ = model_conf.load_ai_model_and_stuff()
        ''' # save in self.params["model_path"]

        checkpoint_name = os.path.splitext(os.path.basename(self.params["model_path"]))[0]
        self.SRGAN = SRGAN(arch=self.params["arch"],model_path=self.params["model_path"],gpu=self.params["gpu"],seed=self.params["seed"])
        self.name = f"SRGAN_arch_{self.params['arch']}_{checkpoint_name}_x{self.params['zoom']}" # _modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_postprocess = False
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        print(f'{self.name} For each image file in <{data_input}>...')
        for data_file in os.listdir(data_input):
            file_path = os.path.join(data_input, data_file)
            dst_file = os.path.join(dst, data_file)
            print(f"Running {self.name} over {os.path.basename(file_path)}")
            rec_img = self._mod_img(file_path)
            #torchvision.utils.save_image(rec_img,dst_file)
            rec_img = pytensor2pil(rec_img)
            # check image size, resize if reconstructed image is not the same
            rec_w, rec_h = rec_img.size
            orig_w, orig_h = pil_image.open(file_path).size
            if rec_w != orig_w or rec_h != orig_h or self.resize_postprocess is True:
                rec_img = rescale_image_exact(rec_img, orig_w, orig_h) # downscale in case of 4x
                
            pilsaveimage(rec_img,dst_file)

        return input_name

    def _mod_img(self, image_file: str) -> np.array:

        if self.resize_preprocess is True:
            rec_img = self.SRGAN.run_sr_mod(image_file, self.params['zoom'], self.params['zoom'], self.blur)
            #rec_img = self.SRGAN.run_sr_resized(image_file, self.params["zoom"])
        else:
            rec_img = self.SRGAN.run_sr_mod(image_file, None, self.params['zoom'], self.blur)
            #rec_img = self.SRGAN.run_sr(image_file)
        return rec_img

class DSModifierLR(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "zoom": 2,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        self.params = params
        self.ds_modifier = ds_modifier

        self.name = "GT-LR" # _modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_postprocess = False
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        print(f'{self.name} For each image file in <{data_input}>...')
        for data_file in os.listdir(data_input):
            file_path = os.path.join(data_input, data_file)
            dst_file = os.path.join(dst, data_file)
            print(f"Running {self.name} over {os.path.basename(file_path)}")
            rec_img = self._mod_img(file_path)
            # check image size, resize if reconstructed image is not the same
            rec_w, rec_h = rec_img.size
            orig_w, orig_h = pil_image.open(file_path).size
            if rec_w != orig_w or rec_h != orig_h or self.resize_postprocess is True:
                rec_img = rescale_image_exact(rec_img, orig_w, orig_h)

            pilsaveimage(rec_img,dst_file)
            
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        image = pil_image.open(image_file).convert('RGB')
        orig_w, orig_h = image.size
        scale = self.zoom

        if self.blur is True:
            image = blur_image(image, scale)

        if self.resize_preprocess is True:
            rec_img = rescale_image(image, scale, interpolation=pil_image.BICUBIC)
        else:
            rec_img = image

        return rec_img

#########################
# Fake Modifier
#########################

class DSModifierFake(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.
    This modifier copies images from a folder already preexecuted (premodified).

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        images_dir: str. Directory of images to copy from.
        src_ext : str = 'tif'. Extension of reference GT images
        dst_ext : str = 'tif'. Extension of images to copy from.
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
        
    """
    def __init__(
        self,
        name: str,
        images_dir: str,
        src_ext : str = 'tif',
        dst_ext : str = 'tif',
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "zoom": 2
        }
    ):
        self.src_ext                = src_ext
        self.dst_ext                = dst_ext
        self.images_dir             = images_dir
        self.name                   = name
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.params.update({"modifier": "{}".format(self.name)})
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.'+self.src_ext) ):
            
            imgp = self._mod_img( image_file )
            dst_file = os.path.join(dst, os.path.basename(image_file))
            if not os.path.exists(dst_file):
                cv2.imwrite( dst_file, imgp )
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        fn = [
            fn for fn in glob(os.path.join(self.images_dir,'*.'+self.dst_ext))
            if os.path.basename(image_file).split('.')[0]==os.path.basename(fn).split('.')[0]
        ][0]
        
        rec_img = cv2.imread(fn)
        
        return rec_img

#########################
# Similarity Metrics
#########################
    
class SimilarityMetrics( Metric ):
    
    def __init__(
        self,
        experiment_info: ExperimentInfo,
        img_dir_gt: str = "test",
        n_jobs: int = 20,
        ext: str = 'tif',
        n_pyramids:Union[int, None]=None,
        slice_size:int=7,
        n_descriptors:int=128,
        n_repeat_projection:int=128,
        proj_per_repeat:int=4,
        device:str='cpu',
        return_by_resolution:bool=False,
        pyramid_batchsize:int=128,
        use_liif_loader : bool = True,
        blur: bool = False,
        resize_preprocess: bool = False,
    ) -> None:
        
        self.img_dir_gt = img_dir_gt
        self.n_jobs     = n_jobs
        self.ext        = ext
        
        self.metric_names = [
            'ssim',
            'psnr',
            #'gmsd',
            #'mdsi',
            #'haarpsi',
            'swd',
            'fid'
        ]
        self.experiment_info = experiment_info
        
        self.n_pyramids           = n_pyramids
        self.slice_size           = slice_size
        self.n_descriptors        = n_descriptors
        self.n_repeat_projection  = n_repeat_projection
        self.proj_per_repeat      = proj_per_repeat
        self.device               = device
        self.return_by_resolution = return_by_resolution
        self.pyramid_batchsize    = pyramid_batchsize

        self.use_liif_loader      = use_liif_loader
        self.blur                 = blur
        self.resize_preprocess    = resize_preprocess
    def _liff_loader_first_time(self,data_input:str) -> None:
    
        dsm_liif = DSModifierLIIF( params={
                'config0':"LIIF_config.json",
                'config1':"test_liif.yaml",
                'model':"LIIF_blur/epoch-best.pth" 
            } )
                
        spec = dsm_liif.spec

        spec['batch_size'] = 1

        spec['dataset'] = {
            'name': 'image-folder',
            'args': {
                'root_path': data_input
            }
        }

        spec['wrapper']['args']['blur'] = self.blur
        spec['wrapper']['args']['resize'] = self.resize_preprocess
        self.spec = spec

        dataset = datasets_liif.make(spec['dataset'])
        dataset = datasets_liif.make(spec['wrapper'], args={'dataset': dataset})

        self.dataset = dataset

    def _gen_liif_loader(self):

        loader = DataLoader(self.dataset, batch_size=self.spec['batch_size'],shuffle=False,
                num_workers=1, pin_memory=True, multiprocessing_context=get_context('loky'))
        
        return loader

    def _parallel(self, fid:object,swdobj:object, pred_fn:str) -> List[Dict[str,Any]]:

        # pred_fn be like: xview_id1529imgset0012+hrn.png
        img_name = os.path.basename(pred_fn)
        gt_fn = os.path.join(self.data_path,self.img_dir_gt,f"{img_name}")

        if self.use_liif_loader and 'LIIF' in pred_fn:
            
            # pred

            pred = cv2.imread( pred_fn )/255

            pred = pred[...,::-1].copy()

            pred = torch.from_numpy( pred )
            pred = pred.view(1,-1,pred.shape[-2],pred.shape[-1])
            pred = torch.transpose( pred, 3, 1 )

            # gt
            
            loader = self._gen_liif_loader()

            enu_file = [
                enu for enu,fn in enumerate(
                    sorted(glob(
                        os.path.join(self.data_path,self.img_dir_gt,'*')
                        ))
                    )
                if os.path.basename(fn)==img_name
                ][0]

            batch = [batch for enu,batch in enumerate(loader) if enu==enu_file][0]

            gt = torch.clamp( batch['gt'] , min=0.0, max=1.0 )

            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [round(ih * s), round(iw * s), 3]
            gt = torch.clamp( batch['gt'].view(1, *shape).transpose(3,1), min=0.0, max=1.0 )

        elif not self.use_liif_loader and 'LIIF' in pred_fn:
            
            pred = cv2.imread( pred_fn )/255
            gt = cv2.imread( gt_fn )/255

            #     pred = pred[...,::-1].copy()
            
            pred = torch.from_numpy( pred )
            gt = torch.from_numpy( gt )

            pred = pred.view(1,-1,pred.shape[-2],pred.shape[-1])
            gt = gt.view(1,-1,gt.shape[-2],gt.shape[-1])
            
            pred = torch.transpose( pred, 3, 1 )
            gt   = torch.transpose( gt  , 3, 1 )

        else:

            pred = transforms.ToTensor()(pil_image.open(pred_fn).convert('RGB')).unsqueeze_(0)
            image = pil_image.open(gt_fn).convert('RGB')
            img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
            if self.blur is True:
                scale = 3
                sigma = 0.5*scale
                kernel_size = 9
                kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
                gt = torch.clamp( kornia.filters.filter2d(img_tensor, kernel_tensor[None]), min=0.0, max=1.0 )
            else:
                gt = img_tensor

        results_dict = {
            "ssim":None,
            "psnr":None,
            "swd":None,
            "fid":None
        }

        if pred.size()!=gt.size():
            
            #print('different size found', pred.size(), gt.size())
            
            pred_for_metrics = torch.clamp( LRSimulator(None,3)._resize(
                    pred,
                    gt.shape[-2],
                    interpolation = 'bicubic',
                    align_corners = True,
                    side = "short",
                    antialias = False
                ), min=0.0, max=1.0 )
            
            if pred_for_metrics.size()!=gt.size():
                return results_dict

        else:

            pred_for_metrics = pred
        
        results_dict = {
            "ssim":piq.ssim(pred_for_metrics,gt).item(),
            "psnr":piq.psnr(pred_for_metrics,gt).item(),
            "fid":np.sum( [
                fid( torch.squeeze(pred_for_metrics)[i,...], torch.squeeze(gt)[i,...] ).item()
                for i in range( pred_for_metrics.shape[1] )
                ] ) / pred_for_metrics.shape[1]
        }
        
        # Make gt even
        if gt.shape[-2]%2!=0 or gt.shape[-1]%2!=0 : # gt size is odd
            new_gt_h = gt.shape[-2]+1 if (gt.shape[-2]+1)%2 == 0 else gt.shape[-2]
            new_gt_w = gt.shape[-1]+1 if (gt.shape[-1]+1)%2 == 0 else gt.shape[-1]
            gt_for_swd = torch.clamp( LRSimulator(None,3)._resize(
                    gt,
                    (new_gt_h,new_gt_w),
                    interpolation = 'bicubic',
                    align_corners = True,
                    side = "short",
                    antialias = False
                ), min=0.0, max=1.0 )

        else:
            gt_for_swd = gt
            
        # Pred should be same as GT
        if pred.size()!=gt_for_swd.size():
            for interp in ['bicubic', 'bilinear']:
                for side in ['short', 'vert', 'horz', 'long']:
                    pred_for_swd = torch.clamp( LRSimulator(None,3)._resize(
                            pred,
                            gt_for_swd.shape[-2],
                            interpolation = interp,
                            align_corners = True,
                            side = side,
                            antialias = False
                        ), min=0.0, max=1.0 )
                    if pred_for_swd.size()==gt_for_swd.size():
                        break
                if pred_for_swd.size()==gt_for_swd.size():
                    break

            if pred_for_swd.size()!=gt_for_swd.size():
                results_dict['swd'] = None
                return results_dict
                
        else:

            pred_for_swd = pred

        try:
            swd_res = swdobj.run(pred_for_swd.double(),gt_for_swd.double()).item()
        except Exception as e:
            print('Failed SWD > ',str(e))
            print('dimensions', pred_for_swd.shape, gt_for_swd.shape)
            swd_res = None
        results_dict['swd'] = swd_res
        
        return results_dict

    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        In this case gt_path will be a glob criteria to the HR images
        """
        
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        
        #predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]['run_id']==guessed_run_id
        ][0]
        pred_fn_path = os.path.join(os.path.dirname(self.data_path),modifier_subfold,'*',f'*.{self.ext}')
        pred_fn_lst = glob(pred_fn_path)

        if len(pred_fn_lst)==0:
            print('Error > empty list "pred_fn_lst" ')
            modifier_subfold=modifier_subfold.replace("_0","") # for some reason it is written "_0" over the modifier name in some cases of modifier_subfold
            pred_fn_path = os.path.join(os.path.dirname(self.data_path),modifier_subfold,'*',f'*.{self.ext}')
            pred_fn_lst = glob(pred_fn_path)
            if len(pred_fn_lst)==0:
                print('Error > empty list "pred_fn_lst" ')
                print(pred_fn_path)
                raise
        if 'LIIF' in pred_fn_lst[0]:
            self._liff_loader_first_time(
                os.path.join(self.data_path,self.img_dir_gt)
                )
        
        stats = { met:0.0 for met in self.metric_names }
        
        fid = piq.FID()
        
        swdobj = SlicedWassersteinDistance(
            n_pyramids           = self.n_pyramids,
            slice_size           = self.slice_size,
            n_descriptors        = self.n_descriptors,
            n_repeat_projection  = self.n_repeat_projection,
            proj_per_repeat      = self.proj_per_repeat,
            device               = self.device,
            return_by_resolution = self.return_by_resolution,
            pyramid_batchsize    = self.pyramid_batchsize
        )

        results_dict_lst = Parallel(n_jobs=self.n_jobs,verbose=10,prefer='threads')(
            delayed(self._parallel)(fid,swdobj,pred_fn)
            for pred_fn in pred_fn_lst
            )
        
        stats = {
            met:np.median([
                r[met]
                for r in results_dict_lst
                if 'float' in type(r[met]).__name__
                ])
            for met in self.metric_names
        }
                
        return stats
