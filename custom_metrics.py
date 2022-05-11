# Generic
import os
from glob import glob
from typing import Any, Dict, List, Union, Tuple # Optional, Tuple
from joblib.externals.loky.backend.context import get_context
from joblib import Parallel, delayed

# Dataset loadout
from torch.utils.data import DataLoader
from S3Loader import ModelConfS3Loader
from datasets import datasets as datasets_liif
from custom_transforms import LRSimulator

# Vision
import cv2
import torch
import piq
from metrics.swd import SlicedWassersteinDistance

# iquaflow
from iq_tool_box.experiments import ExperimentInfo
from iq_tool_box.metrics import Metric

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
        device:str='cuda:0', # cpu
        return_by_resolution:bool=False,
        pyramid_batchsize:int=128,
        use_liif_loader : bool = True,
        zoom: int = 3,
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
        self.zoom                 = zoom
        self.blur                 = blur
        self.resize_preprocess    = resize_preprocess
    def _liff_loader_first_time(self,data_input:str) -> None:
    	
    	
        '''
        dsm_liif = DSModifierLIIF( params={
                'config0':"LIIF_config.json",
                'config1':"test_liif.yaml",
                'model':"LIIF_blur/epoch-best.pth" 
            } )
        spec = dsm_liif.spec
        '''
        model_conf = ModelConfS3Loader(
                model_fn      = "LIIF_blur/epoch-best.pth",
                config_fn_lst = ["LIIF_config.json","test_liif.yaml"],
                bucket_name   = "image-quality-framework",
                algo          = "LIIF"
        )
        spec   =  model_conf.spec

        # dataset loader specifications
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
            s = np.sqrt(batch['coord'].shape[1] / (ih * iw))
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

            # todo: use blur_image function instead
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
            
            pred_for_metrics = torch.clamp( LRSimulator(None,self.zoom)._resize(
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

