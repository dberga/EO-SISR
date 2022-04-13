import os
import shutil
import piq
import torch

from glob import glob
from scipy import ndimage
from typing import Any, Dict, Optional, Union, Tuple, List

import cv2
import mlflow
import numpy as np

from iq_tool_box.datasets import DSWrapper
from iq_tool_box.experiments import ExperimentInfo, ExperimentSetup
from iq_tool_box.experiments.task_execution import PythonScriptTaskExecution
from iq_tool_box.metrics import RERMetric, SNRMetric
from iq_tool_box.quality_metrics import RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics

from custom_iqf import DSModifierMSRN, DSModifierFSRCNN,  DSModifierLIIF, DSModifierESRGAN, DSModifierCAR, DSModifierSRGAN, DSModifierFake
from custom_iqf import SimilarityMetrics
from visual_comparison import visual_comp, scatter_plots

def rm_experiment(experiment_name = "SiSR"):
    """Remove previous mlflow records of previous executions of the same experiment"""
    try:
        mlflow.delete_experiment(ExperimentInfo(f"{experiment_name}").experiment_id)
    except:
        pass
    shutil.rmtree("mlruns/.trash/",ignore_errors=True)
    os.makedirs("mlruns/.trash/",exist_ok=True)
    shutil.rmtree(f"./Data/test-ds/.ipynb_checkpoints",ignore_errors=True)
    [shutil.rmtree(x) for x in glob(os.path.join(os.getcwd(), "**", '__pycache__'), recursive=True)]
    
#Define name of IQF experiment
experiment_name = "SiSR"

# Remove previous mlflow records of previous executions of the same experiment
# rm_experiment(experiment_name = experiment_name)

#Define path of the original(reference) dataset
data_path = f"./Data/test-ds"
images_folder = "test"
images_path = os.path.join(data_path, images_folder)
database_name = os.path.basename(data_path)
data_root = os.path.dirname(data_path)

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#List of modifications that will be applied to the original dataset:

ds_modifiers_list = [
    DSModifierESRGAN( params={
        'zoom':3,
        'model':"ESRGAN_1to033_x3_blur/net_g_latest.pth"
    } ),
    DSModifierMSRN( params={
    'zoom':3,
    'model':"MSRN_nonoise/MSRN_1to033/model_epoch_1500.pth"
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        'model':"LIIF_blur/epoch-best.pth" 
    } ),
    DSModifierFSRCNN( params={
        'config':"test_scale3.json",
        'model':"FSRCNN_1to033_x3_blur/best.pth"
    } ),
    DSModifierSRGAN( params={
        #"arch": "srgan_2x2",
        #"model_path": "./models/srgan/weights/PSNR.pth",
        "arch": "srgan",
        "model_path": "./models/srgan/weights/PSNR_inria_scale4.pth",
        "gpu": 0,
        "seed": 666,
        "zoom": 3,
    } ),
    DSModifierCAR( params={
        "SCALE": 4,
        #"SCALE": 2,
        "model_dir": "./models/car/models",
        "gpu": 0,
        "zoom": 3,
    } ),
]

# check existing modified images and replace already processed modifiers by DSModifierFake (only read images)
ds_modifiers_indexes_dict = {}
for idx,ds_modifier in enumerate(ds_modifiers_list):
    ds_modifiers_indexes_dict[ds_modifier._get_name()]=idx
ds_modifiers_found = [name for name in glob(os.path.join(data_root,database_name)+"#*")]
for sr_folder in ds_modifiers_found:
    sr_name = os.path.basename(sr_folder).replace(database_name+"#","")
    sr_dir=os.path.join(sr_folder,images_folder)
    if len(os.listdir(sr_dir)) == len(os.listdir(images_path)) and sr_name in list(ds_modifiers_indexes_dict.keys()):
        index_modifier = ds_modifiers_indexes_dict[sr_name]
        ds_modifiers_list[index_modifier]=DSModifierFake(name=sr_name,images_dir = sr_dir)

#Define path of the training script
python_ml_script_path = 'sr.py'

# Task execution executes the training loop
task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1
)

#Execute the experiment
experiment.execute()
# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)


print('Visualizing examples')

lst_folders_mod = [images_path]+[os.path.join(data_path+'#'+ds_modifier._get_name(),images_folder) for ds_modifier in ds_modifiers_list]
lst_labels_mod = ["GT"]+[ds_modifier._get_name() for ds_modifier in ds_modifiers_list]

visual_comp(lst_folders_mod, lst_labels_mod, True, "comparison/")


print('Calculating similarity metrics...')

win = 28
_ = experiment_info.apply_metric_per_run(
    SimilarityMetrics(
        experiment_info,
        n_jobs               = 5,
        ext                  = 'tif',
        n_pyramids           = 1,
        slice_size           = 7,
        n_descriptors        = win*2,
        n_repeat_projection  = win,
        proj_per_repeat      = 4,
        device               = 'cpu',
        return_by_resolution = False,
        pyramid_batchsize    = win,
        use_liif_loader      = True
    ),
    ds_wrapper.json_annotations,
)

print('Calculating RER Metric...')

_ = experiment_info.apply_metric_per_run(
    RERMetric(
        experiment_info,
        win=16,
        stride=16,
        ext="tif",
        n_jobs=15
    ),
    ds_wrapper.json_annotations,
)

print('Calculating SNR Metric...')

__ = experiment_info.apply_metric_per_run(
     SNRMetric(
         experiment_info,
         ext="tif",
         method="HB",
         patch_size=30, #patch_sizes=[30]
         #confidence_limit=50.0,
         #n_jobs=15
     ),
     ds_wrapper.json_annotations,
 )

df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=['ssim','psnr','swd','snr_median','snr_mean','fid','rer_0','rer_1','rer_2'],
    dropna=False
)
print(df)
df.to_csv(f'./{experiment_name}_metrics.csv')
scatter_plots(df, [['ssim','psnr'],['fid','swd'],['rer_0','snr_mean'],['snr_mean','psnr']], True, "plots/")

print('Calculating Regressor Quality Metrics...') #default configurations
_ = experiment_info.apply_metric_per_run(RERMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(SNRMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(GaussianBlurMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(NoiseSharpnessMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(GSDMetrics(), ds_wrapper.json_annotations)

df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=[
            "sigma",
            "rer",
            "snr",
            "sharpness",
            "scale"
        ]
)
print(df)
df.to_csv(f'./{experiment_name}_regressor.csv')
scatter_plots(df, [['sigma','rer'],['sharpness','sigma'],['rer','snr'],['sharpness','rer'],['snr','snr_mean'],['sigma','scale'],['snr','scale']], True, "plots/")

print("Comparing Metrics with Regressed Quality Metrics")
df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=[
            "ssim",
            "psnr",
            "swd",
            "snr_mean",
            "fid",
            "rer_0",
            "rer_1",
            "rer_2"
            "sigma",
            "rer",
            "snr",
            "sharpness",
            "scale"
        ]
)
df.to_csv(f'./{experiment_name}.csv')
scatter_plots(df, [['sigma','rer_0'],['rer','rer_0'],['sharpness','rer_0'],['snr','snr_mean'],['snr','psnr'],['scale','rer'],['scale','snr']], True, "plots/")
