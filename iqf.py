import os
import shutil
import piq
import torch

from glob import glob
from scipy import ndimage
from typing import Any, Dict, Optional, Union, Tuple, List

import cv2
import numpy as np

from iq_tool_box.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iq_tool_box.experiments import ExperimentInfo, ExperimentSetup
from iq_tool_box.experiments.experiment_visual import ExperimentVisual
from iq_tool_box.experiments.task_execution import PythonScriptTaskExecution
from iq_tool_box.metrics import BBDetectionMetrics

from custom_iqf import SimilarityMetrics
from custom_iqf import DSModifierLIIF
from custom_iqf import DSModifierFSRCNN
from custom_iqf import DSModifierMSRN

mic = '-micro'
shutil.rmtree(f"./Data/test{mic}-ds/.ipynb_checkpoints",ignore_errors=True)
for el in os.listdir("./Data"):
    if el!="test" and "#" in el:
        shutil.rmtree(os.path.join("./Data",el),ignore_errors=True)
shutil.rmtree("./mlruns",ignore_errors=True)


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
        'model':"MSRN/SISR_MSRN_X2_BICUBIC.pth"
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        'model':"liif_UCMerced/epoch-best.pth"
    } ),
    DSModifierFSRCNN( params={
        'config':"test.json",
        'model':"FSRCNN_1to033_x3_noblur/best.pth"
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

experiment_info = ExperimentInfo(experiment_name)

_ = experiment_info.apply_metric_per_run(
    SimilarityMetrics( experiment_info ),
    ds_wrapper.json_annotations,
)


