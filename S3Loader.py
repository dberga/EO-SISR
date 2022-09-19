# Generic
import os
import json
import yaml
from typing import Any, List
import tempfile

# Vision
import torch
import torch.backends.cudnn as cudnn

# Models specific loadout
from models.fsrcnn.model_fsrcnn import FSRCNN
from models.liif import models as models_liif
from models.esrgan import esrgan
from models.esrgan import RRDBNet_arch as RRDBNet_arch
from models.msrn import msrn
from models.msrn.msrn import load_msrn_model
from models.car.car import load_car_model
from models.srgan.srgan import load_srgan_model

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
        urldir        = None,
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
        self.urldir             = urldir
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

            if self.urldir is None:
                url = f"https://{self.bucket_name}.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/models/{kind}/{bucket_fn}"  # S3 folder pattern loadout
            else:
                url = urldir  # manual loadout from url

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
            args.zoom = self.zoom
            model = self._load_model_car(fn_dict_aux["model"], args.SCALE)

        elif self.algo=='SRGAN':
            args = Args()
            args.arch = self.kwargs["arch"]
            args.zoom = self.zoom
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
        model = RRDBNet_arch.RRDBNet(3, 3, 64, 23, gc=32)
        #model.load_state_dict(torch.load(args.model_path), strict=True)
        print(model_fn)
        weights = torch.load(model_fn, map_location=device)
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

