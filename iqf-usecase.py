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

from custom_iqf import DSModifierMSRN, DSModifierFSRCNN,  DSModifierLIIF
from custom_iqf import SimilarityMetrics

mic = ''
# Remove previous mlflow records of previous executions of the same experiment
try:
    mlflow.delete_experiment(ExperimentInfo("experimentA").experiment_id)
except:
    pass
shutil.rmtree("mlruns/.trash/",ignore_errors=True)
shutil.rmtree(f"./Data/test{mic}-ds/.ipynb_checkpoints",ignore_errors=True)


#Define name of IQF experiment
experiment_name = "experimentA"

#Define path of the original(reference) dataset
data_path = f"./Data/test{mic}-ds"

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#Define path of the training script
python_ml_script_path = 'custom_train.py'

#List of modifications that will be applied to the original dataset:

ds_modifiers_list = [
    DSModifierMSRN( params={
        'zoom':3,
        'model':"MSRN/MSRN_1to033_x3_blur/model_epoch_1500.pth"
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        'model':"LIIF_blur/epoch-best.pth" 
    } ),
    DSModifierFSRCNN( params={
        'config':"test_scale3.json",
        'model':"FSRCNN_1to033_x3_blur/best.pth"
    } )
]

# Task execution executes the training loop
task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    repetitions=1
)

#Execute the experiment
experiment.execute()

# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)

print('Calculating similarity metrics...')

# win = 128
# _ = experiment_info.apply_metric_per_run(
#     SimilarityMetrics(
#         experiment_info,
#         n_jobs               = 20,
#         ext                  = 'tif',
#         n_pyramids           = 2,
#         slice_size           = 7,
#         n_descriptors        = win*2,
#         n_repeat_projection  = win,
#         proj_per_repeat      = 4,
#         device               = 'cpu',
#         return_by_resolution = False,
#         pyramid_batchsize    = win
#     ),
#     ds_wrapper.json_annotations,
# )

for win in [64,128,300]:
    for n_pyramids in [2,3]:
        for slice_size in [7,9,12,15]:
            for proj_per_repeat in [4]:
                try:

                    _ = experiment_info.apply_metric_per_run(
                        SimilarityMetrics(
                            experiment_info,
                            n_jobs               = 20,
                            ext                  = 'tif',
                            n_pyramids           = n_pyramids,
                            slice_size           = slice_size,
                            n_descriptors        = win*2,
                            n_repeat_projection  = win,
                            proj_per_repeat      = proj_per_repeat,
                            device               = 'cpu',
                            return_by_resolution = False,
                            pyramid_batchsize    = win
                        ),
                        ds_wrapper.json_annotations,
                    )

                    print('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW',{
                        'win':win,
                        'n_pyramids':n_pyramids,
                        'slice_size':slice_size,
                        'proj_per_repeat':proj_per_repeat
                    })

                except:

                    continue


# print('Calculating RER Metric...')

# _ = experiment_info.apply_metric_per_run(
#     RERMetric(
#         experiment_info,
#         win=16,
#         stride=16,
#         ext="tif",
#         n_jobs=20
#     ),
#     ds_wrapper.json_annotations,
# )

# print('Calculating SNR Metric...')

# __ = experiment_info.apply_metric_per_run(
#      SNRMetric(
#          experiment_info,
#          n_jobs=20,
#          ext="tif",
#          patch_sizes=[30],
#          confidence_limit=50.0
#      ),
#      ds_wrapper.json_annotations,
#  )


