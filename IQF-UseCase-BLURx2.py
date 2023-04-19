#!/usr/bin/env python
# coding: utf-8

# # SingleImageSR use case
# 
# The Single Image Super Resolution (SISR) use case is build to compare the image quality between different SiSR solutions. A SiSR algorithm inputs one frame and outputs an image with greater resolution.
# These are the methods that are being compared in the use case:
# 
# 1. Fast Super-Resolution Convolutional Neural Network (FSRCNN) [Ledig et al., 2016]
# 2. Single Image Super-Resolution Generative Adversarial Networks (SRGAN) [Dong et al., 2016]
# 3. Multi-scale Residual Network (MSRN) [Li et al., 2018]
# 4. Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) [Wang et al., 2018]
# 5. Content Adaptive Resampler (CAR) [Sun & Chen, 2019]
# 6. Local Implicit Image Function (LIIF) [Chen et al., 2021]
# 
# A use case in IQF usally involves wrapping a training within mlflow framework. In this case we estimate quality on the solutions offered by the different Dataset Modifiers which are the SISR algorithms. Similarity metrics against the Ground Truth are then compared, as well as predicted Quality Metrics.

# In[1]:


# SiSR Execution settings
plot_sne = False                         # t-SNE plot? (requires a bit of RAM)
plot_visual_comp = True                  # visual comparison?
plot_metrics_comp = True                 # metrics comparison?
use_fake_modifiers = True                # read existing sr output data files instead of modifying?
use_existing_metrics = True              # read existing metrics output data files instead of processing them?
savefig = True
compute_similarity_metrics = True       # compute these? 
compute_noise_metrics = True             # compute these?
compute_sharpness_metrics = True         # compute these?
compute_regressor_quality_metrics = True # compute these?
settings_lr_blur = True                  # blur right before modification?
settings_resize_preprocess = False       # resize right before modification?
settings_resize_postprocess = True       # resize right after modification?
settings_zoom = 2                        # scale?


# In[2]:


# load_ext autoreload
#autoreload 2
import os
import shutil
import mlflow
import pandas as pd
from glob import glob
from pdb import set_trace as debug

# display tables of max 50 columns
pd.set_option('display.max_columns', 50)

## update iquaflow with "pip3 install git+https://ACCESSTOKEN@github.com/satellogic/iquaflow.git"
from iquaflow.datasets import DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import SharpnessMetric, SNRMetric
from iquaflow.quality_metrics import ScoreMetrics, RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics

from custom_modifiers import DSModifierLR, DSModifierFake, DSModifierMSRN, DSModifierFSRCNN,  DSModifierLIIF, DSModifierESRGAN, DSModifierCAR, DSModifierSRGAN
from custom_metrics import SimilarityMetrics
from visual_comparison import visual_comp, metric_comp, plotSNE


# In[3]:


#Define path of the original(reference) dataset
data_path = f"./Data/test-ds"
images_folder = "test"
images_path = os.path.join(data_path, images_folder)
database_name = os.path.basename(data_path)
data_root = os.path.dirname(data_path)

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#Define name of IQF experiment
experiment_name = "SiSR"
experiment_name += f"_{database_name}"
experiment_name += f"_blur{settings_lr_blur}"+f"x{settings_zoom}"
experiment_name += f"_pre{settings_resize_preprocess}"+f"x{settings_zoom}"
experiment_name += f"_post{settings_resize_postprocess}"

#Output
plots_folder = "plots/"+experiment_name+"/"
comparison_folder = "comparison/"+experiment_name+"/"
results_folder = "results/"+experiment_name+"/"


# In[4]:



# Remove previous mlflow records of previous executions of the same experiment
try:
    # create output dirs
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(comparison_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    # rm_experiment
    mlflow.delete_experiment(ExperimentInfo(f"{experiment_name}").experiment_id)
    # Clean mlruns and __pycache__ folders
    shutil.rmtree("mlruns/",ignore_errors=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    shutil.rmtree(f"{data_path}/.ipynb_checkpoints",ignore_errors=True)
    [shutil.rmtree(x) for x in glob(os.path.join(os.getcwd(), "**", '__pycache__'), recursive=True)]
except:
    pass


# In[5]:


# plot SNE of existing images
if plot_sne:
    plotSNE(database_name, images_path, (232,232), 6e3, True, savefig, plots_folder)


# In[6]:


#List of modifications that will be applied to the original dataset:
ds_modifiers_list = [
    DSModifierLR( params={
        'zoom': settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    }),
        DSModifierFSRCNN( params={
        'config':"test.json",
        #'config':"test_scale3.json",
        'model':"FSRCNN_1to05_x2_noblur/best.pth",
        #'model':"FSRCNN_1to033_x3_noblur/best.pth",
        #'model':"FSRCNN_1to033_x3_blur/best.pth",
        'zoom': settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierSRGAN( params={
        "arch": "srgan_2x2",
        # "arch": "srgan",
        "model": "srgan/PSNR_inria_scale2.pth",
        #"model": "srgan/PSNR_inria_scale4.pth",
        "zoom": settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierMSRN( params={
        #'model':"MSRN/MSRN_1to033/model_epoch_1500.pth",
        'model':"MSRN_nonoise/MSRN_1to05/model_epoch_1500.pth",
        'compress': False,
        'add_noise': None,
        'zoom': settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierESRGAN( params={
        #'model':"ESRGAN_1to033_x3_blur/net_g_latest.pth",
        'model':"ESRGAN_1to05_x2_blur/net_g_latest.pth",
        'zoom':settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierCAR( params={
        "model": "car/2x/usn.pth",
        #"model": "car/4x/usn.pth",
        "zoom": settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        #'model':"LIIF_blur/epoch-best.pth",
        'model':"liif_UCMerced/epoch-best.pth",
        'zoom': settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
]

# adding fake modifier of original images (GT)
ds_modifiers_list.append(DSModifierFake(name="HR",images_dir=images_path))

# check existing modified images and replace already processed modifiers by DSModifierFake (only read images)
if use_fake_modifiers: 
    ds_modifiers_indexes_dict = {}
    for idx,ds_modifier in enumerate(ds_modifiers_list):
        ds_modifiers_indexes_dict[ds_modifier._get_name()]=idx
    ds_modifiers_found = [name for name in glob(os.path.join(data_root,database_name)+"#*")]
    for sr_folder in ds_modifiers_found:
        sr_name = os.path.basename(sr_folder).replace(database_name+"#","")
        sr_dir=os.path.join(sr_folder,images_folder)
        if len(os.listdir(sr_dir)) == len(os.listdir(images_path)) and sr_name in list(ds_modifiers_indexes_dict.keys()):
            index_modifier = ds_modifiers_indexes_dict[sr_name]
            ds_modifiers_list[index_modifier]=DSModifierFake(name=sr_name,images_dir = sr_dir,params = {"modifier": sr_name})
  


# In[7]:


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


# ## Execution
# 
# The number of runs are all the combinations between repetitions, modifiers list as well as hyper parameter changes.
# 
# (you can skip this step in demo pre-executed datasets)

# In[8]:


#Execute the experiment
experiment.execute()
# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)


# In[9]:


if plot_visual_comp:
    print('Visualizing examples')

    lst_folders_mod = [os.path.join(data_path+'#'+ds_modifier._get_name(),images_folder) for ds_modifier in ds_modifiers_list]
    lst_labels_mod = [ds_modifier._get_name().replace("sisr+","").split("_")[0] for ds_modifier in ds_modifiers_list] # authomatic readout from folders

    visual_comp(lst_folders_mod, lst_labels_mod, savefig, comparison_folder)


# ## Metrics
# 
# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis. Its instance can also be used to apply metrics per run. Some custum metrics are presented. They where build by inheriting Metric from iq_tool_box.metrics.
# 
# (you can skip this step in demo pre-executed datasets)

# In[10]:


df_results = []
similarity_metrics = ['ssim','psnr','swd','fid', 'ms_ssim','haarpsi','gmsd','mdsi']
noise_metrics = ['snr_median','snr_mean']
sharpness_metrics = ['RER', 'MTF', 'FWHM']
regressor_quality_metrics = ['sigma','snr','rer','sharpness','scale','score']


# In[11]:


print('Calculating similarity metrics...'+",".join(similarity_metrics))
path_similarity_metrics = f'./{results_folder}similarity_metrics.csv'
if use_existing_metrics and os.path.exists(path_similarity_metrics):
    df = pd.read_csv(path_similarity_metrics)
elif compute_similarity_metrics:
    win = 28
    _ = experiment_info.apply_metric_per_run(
        SimilarityMetrics(
            experiment_info,
            n_jobs               = 4,
            ext                  = 'tif',
            n_pyramids           = 1,
            slice_size           = 7,
            n_descriptors        = win*2,
            n_repeat_projection  = win,
            proj_per_repeat      = 4,
            device               = 'cuda:0', #'cpu',
            return_by_resolution = False,
            pyramid_batchsize    = win,
            use_liif_loader      = True,
            zoom                 = settings_zoom,
            blur                 = False, # settings_lr_blur
            resize_preprocess    = False, # settings_resize_preprocess
        ),
        ds_wrapper.json_annotations,
    )
    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=similarity_metrics,
        dropna=False
    )
    df.to_csv(path_similarity_metrics)
else:
    df = pd.DataFrame(0, index=[0], columns=['ds_modifier']+similarity_metrics); # empty df


# In[12]:


df_results.append(df)


# In[13]:


df


# In[14]:


if plot_metrics_comp:
    metric_comp(df,similarity_metrics,savefig,plots_folder)


# ## Noise and Sharpness (Blind) Metrics

# In[15]:


print('Calculating Noise Metrics...'+",".join(noise_metrics))
path_noise_metrics = f'./{results_folder}noise_metrics.csv'
if use_existing_metrics and os.path.exists(path_noise_metrics):
    df = pd.read_csv(path_noise_metrics)
elif compute_noise_metrics:
    _ = experiment_info.apply_metric_per_run(
         SNRMetric(
             experiment_info,
             ext="tif",
             method="HB",
             # patch_size=30, #patch_sizes=[30]
             #confidence_limit=50.0,
             #n_jobs=15
         ),
         ds_wrapper.json_annotations,
     )
    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=noise_metrics,
        dropna=False
    )
    df.to_csv(path_noise_metrics)
else:
    df = pd.DataFrame(0, index=[0], columns=['ds_modifier']+noise_metrics); # empty df


# In[16]:


df_results.append(df)


# In[17]:


df


# In[18]:


if plot_metrics_comp:
    metric_comp(df,noise_metrics,savefig,plots_folder)


# In[19]:


print('Calculating Sharpness Metrics...'+",".join(sharpness_metrics))
path_sharpness_metrics = f'./{results_folder}sharpness_metrics.csv'
# init from input sharpness metrics and replace list to output with directions (horizontal, vertical, other)
sharpness_metric = SharpnessMetric(
        experiment_info,
        stride=16,
        ext="tif",
        parallel=False,
        metrics=sharpness_metrics,
        njobs=1
    )
sharpness_metrics = sharpness_metric.metric_names #after initialization, update to (output) names
if use_existing_metrics and os.path.exists(path_sharpness_metrics):
    df = pd.read_csv(path_sharpness_metrics)
elif compute_sharpness_metrics:
    _ = experiment_info.apply_metric_per_run(
        sharpness_metric,
        ds_wrapper.json_annotations,
    )
    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=sharpness_metrics,
        dropna=False
    )
    df.to_csv(path_sharpness_metrics)
else:
    df = pd.DataFrame(0, index=[0], columns=['ds_modifier']+sharpness_metrics); # empty df


# In[20]:


df_results.append(df)


# In[21]:


df


# In[22]:


if plot_metrics_comp:
    metric_comp(df,sharpness_metrics,savefig,plots_folder)


# # Regressor Quality Metrics

# In[23]:

network_crop_size = 256
print('Calculating Regressor Quality Metrics...'+",".join(regressor_quality_metrics)) #default configurations
path_regressor_quality_metrics = f'./{results_folder}regressor_quality_metrics.csv'
if use_existing_metrics and os.path.exists(path_regressor_quality_metrics):
    df = pd.read_csv(path_regressor_quality_metrics)
elif compute_regressor_quality_metrics:
    _ = experiment_info.apply_metric_per_run(ScoreMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    _ = experiment_info.apply_metric_per_run(RERMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    _ = experiment_info.apply_metric_per_run(SNRMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    _ = experiment_info.apply_metric_per_run(GaussianBlurMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    _ = experiment_info.apply_metric_per_run(NoiseSharpnessMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    _ = experiment_info.apply_metric_per_run(GSDMetrics(input_size=network_crop_size), ds_wrapper.json_annotations)
    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=regressor_quality_metrics,
        dropna=False
    )
    df.to_csv(path_regressor_quality_metrics)
else:
    df = pd.DataFrame(0, index=[0], columns=['ds_modifier']+regressor_quality_metrics); # empty df



# In[24]:


df_results.append(df)


# In[25]:


df


# In[26]:


if plot_metrics_comp:
    metric_comp(df,regressor_quality_metrics,savefig,plots_folder)


# # All Metrics Comparison

# In[27]:


all_metrics = similarity_metrics+noise_metrics+sharpness_metrics+regressor_quality_metrics
print('Comparing all Metrics...'+",".join(all_metrics))
path_all_metrics = f'./{results_folder}all_metrics.csv'
if use_existing_metrics and os.path.exists(path_all_metrics):
    df = pd.read_csv(path_all_metrics)
elif compute_similarity_metrics and compute_noise_metrics and compute_sharpness_metrics and compute_regressor_quality_metrics:
    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=all_metrics,
        dropna=False
    )
    df.to_csv(path_all_metrics)
else:
    #df = pd.concat(df_results,axis=1)
    df = df_results[0]
    for df_result in df_results[1:]:
        df = pd.merge(df, df_result)
    # df = pd.DataFrame(0, index=[0], columns=['ds_modifier']+all_metrics); # empty df


# In[28]:


print(f"Removing Unnamed indexes and writing csv: {path_all_metrics}")
{df.drop(str(field),inplace=True,axis=1) for field in df if "Unnamed" in str(field)}
df.to_csv(path_all_metrics)


# In[29]:


df


# In[30]:


if plot_metrics_comp:
    metric_comp(df,all_metrics,savefig,plots_folder)

