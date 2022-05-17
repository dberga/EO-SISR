# Generic
import os
import numpy as np
from typing import Any, Dict, Optional
from glob import glob

# Dataset loadout
from torch.utils.data import DataLoader
from S3Loader import ModelConfS3Loader
from datasets import datasets as datasets_liif
from models.fsrcnn.utils_fsrcnn import convert_ycbcr_to_rgb
from models.fsrcnn.utils_fsrcnn import preprocess as fsrcnn_preprocess
from models.car.car import pytensor2pil as car_pytensor2pil

# Model loadout
from models.liif import utils_liif
from models.msrn.msrn import process_file_msrn
from models.car.car import CAR
from models.srgan.srgan import SRGAN
from models.esrgan import esrgan

# Vision
import torch
import PIL.Image as pil_image
import cv2

# Custom Transforms
from custom_transforms import blur_image, rescale_image, rescale_image_wh

# iquaflow
from iquaflow.datasets import DSModifier


#########################
# HR to LR Modifier
#########################

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

        self.name = "LR" # _modifier
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
                rec_img = rescale_image_wh(rec_img, orig_w, orig_h)

            rec_img.save(dst_file)
            
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
# Custom SR Modifiers
#########################

class DSModifierLIIF(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"LIIF",
            "config0": "LIIF_config.json",
            "config1": "test_liif.yaml",
            "model": "liif_UCMerced/epoch-best.pth",
            "zoom": 3,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        params['algo'] = 'LIIF'
        algo           = params['algo']
        print(params['config0'])
        print(params['config1'])
        print(params['model'])

        subname = algo + '_'+os.path.splitext(params['model'])[0].replace('/','-')
                    #'_' + os.path.splitext(params['config0'])[0]+ \
                    #'_' + os.path.splitext(params['config1'])[0]+ \
        
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
        if "zoom" in self.params.keys():
            self.zoom           = params['zoom']
            self.name += f"_x{params['zoom']}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.zoom = None
        if "blur" in self.params.keys():
            self.blur           = params['blur']
            if self.blur is True:
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
            self.params.update({"modifier": "{}".format(self._get_name())})
        if "resize_postprocess" in self.params.keys():
            self.resize_postprocess = self.params["resize_postprocess"]
            if self.resize_postprocess is True:
                self.name += "_rpost"
                self.params.update({"modifier": "{}".format(self._get_name())})
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
                    rec_img = rescale_image_wh(output, orig_w, orig_h)
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
        s = np.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape) \
            .permute(0, 1, 2, 3).contiguous()
        rec_img = pred.detach().cpu().numpy().squeeze()


        return rec_img

class DSModifierFSRCNN(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"FSRCNN",
            "config": "test.json",
            "model": "FSRCNN_1to033_x3_noblur/best.pth",
            "zoom": 3,
            "blur": False,
            "resize_preprocess": True,
            "resize_postprocess": False,
        },
    ):
        
        params['algo'] = 'FSRCNN'
        algo           = params['algo']
        
        subname = algo + '_' + os.path.splitext(params['model'])[0].replace('/','-') # os.path.splitext(params['config'])[0]+'_'+
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
            bicubic = rescale_image_wh(image.copy(),orig_w*args.scale, orig_h*args.scale)
        lr, _ = fsrcnn_preprocess(lr, self.device)
        _, ycbcr = fsrcnn_preprocess(bicubic, self.device) # convert_rgb_to_ycbcr
        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        if self.resize_postprocess is True:
            output = rescale_image_wh(output, orig_w, orig_h)
        return output

class DSModifierMSRN(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"MSRN",
            "model": "MSRN/SISR_MSRN_X2_BICUBIC.pth",
            "compress": False,
            "add_noise": False,
            "zoom": 2,
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):
            
            #signal.signal(signal.SIGALRM, alarm_handler)
            #signal.alarm(5)
            
            #try:
         
            imgp = self._mod_img( image_file )
            print(f"Running {self.name} over {os.path.basename(image_file)}")
            cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            
            #except TimeOutException as ex:
            #    print(ex)
            #except Exception as e:
            #    print(e)
            
            #signal.alarm(0)
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        zoom       = self.params["zoom"]
        loaded     = cv2.imread(image_file, -1)
        orig_w, orig_h = pil_image.open(image_file).size
        wind_size  = loaded.shape[1]
        gpu_device = "0"
        res_output = 1/zoom # inria resolution

        if self.blur is True:
            loaded = blur_image(loaded, zoom)

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
        )
        if self.resize_postprocess is True:
            rec_img = rescale_image_wh(rec_img, orig_w, orig_h)
        return rec_img


class DSModifierESRGAN(DSModifier):
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"ESRGAN",
            "model": "./ESRGAN_1to033_x3_blur/net_g_latest.pth",
            "zoom": 2,
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
            rec_img = rescale_image_wh(rec_img, orig_w, orig_h)
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
            rec_img = car_pytensor2pil(rec_img)
            # check image size, resize if reconstructed image is not the same
            rec_w, rec_h = rec_img.size
            orig_w, orig_h = pil_image.open(file_path).size
            if rec_w != orig_w or rec_h != orig_h or self.resize_postprocess is True:
                rec_img = rescale_image_wh(rec_img, orig_w, orig_h) # downscale in case of 4x
            
            rec_img.save(dst_file)
            
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        if self.resize_preprocess is True:
            rec_img = self.CAR.run_upscale_mod(image_file, True, self.params['zoom'], self.blur)
            #rec_img = self.CAR.run_upscale_resized(image_file, self.params['zoom'])
        else:
            rec_img = self.CAR.run_upscale_mod(image_file, False, self.params['zoom'], self.blur)
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
                self.name += "_blur" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.blur = False
        if "resize_preprocess" in self.params.keys():
            self.resize_preprocess = self.params["resize_preprocess"]
            if self.resize_preprocess is True:
                self.name += "_rpre" + f"{self.zoom}"
                self.params.update({"modifier": "{}".format(self._get_name())})
        else:
            self.resize_preprocess = True
            self.name += "_rpre" + f"{self.zoom}"
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
            rec_img = car_pytensor2pil(rec_img)
            # check image size, resize if reconstructed image is not the same
            rec_w, rec_h = rec_img.size
            orig_w, orig_h = pil_image.open(file_path).size
            if rec_w != orig_w or rec_h != orig_h or self.resize_postprocess is True:
                rec_img = rescale_image_wh(rec_img, orig_w, orig_h) # downscale in case of 4x
                
            rec_img.save(dst_file)

        return input_name

    def _mod_img(self, image_file: str) -> np.array:

        if self.resize_preprocess is True:
            rec_img = self.SRGAN.run_sr_mod(image_file, True, self.params['zoom'], self.blur)
            #rec_img = self.SRGAN.run_sr_resized(image_file, self.params["zoom"])
        else:
            rec_img = self.SRGAN.run_sr_mod(image_file, False, self.params['zoom'], self.blur)
            #rec_img = self.SRGAN.run_sr(image_file)
        return rec_img
