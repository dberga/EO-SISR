import os
import shutil
import mlflow
from glob import glob

## update iquaflow with "pip3 install git+https://ACCESSTOKEN@github.com/satellogic/iquaflow.git"
from iquaflow.datasets import DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import SharpnessMetric, SNRMetric
from iquaflow.quality_metrics import ScoreMetrics, RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics

from custom_modifiers import DSModifierLR, DSModifierFake, DSModifierMSRN, DSModifierFSRCNN,  DSModifierLIIF, DSModifierESRGAN, DSModifierCAR, DSModifierSRGAN
from custom_metrics import SimilarityMetrics
from visual_comparison import visual_comp, scatter_plots, plotSNE

# SiSR Execution settings
plot_sne = False                     # t-SNE plot? (requires a bit of RAM)
use_fake_modifiers = False          # read existing sr output data files instead of modifying?
settings_lr_blur = False            # blur right before modification?
settings_resize_preprocess = True   # resize right before modification?
settings_resize_postprocess = False # resize right after modification?
settings_zoom = 3                   # scale?

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

# Remove previous mlflow records of previous executions of the same experiment
try: # rm_experiment
    mlflow.delete_experiment(ExperimentInfo(f"{experiment_name}").experiment_id)
    # Clean mlruns and __pycache__ folders
    shutil.rmtree("mlruns/",ignore_errors=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    shutil.rmtree(f"{data_path}/.ipynb_checkpoints",ignore_errors=True)
    [shutil.rmtree(x) for x in glob(os.path.join(os.getcwd(), "**", '__pycache__'), recursive=True)]
except:
    pass

# plot SNE of existing images
if plot_sne:
    plotSNE(database_name, images_path, (232,232), 6e4, True, True, "plots/")

#List of modifications that will be applied to the original dataset:
#List of modifications that will be applied to the original dataset:
ds_modifiers_list = [
    DSModifierLR( params={
        'zoom': settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    }),
    DSModifierFSRCNN( params={
        'config':"test_scale3.json",
        'model':"FSRCNN_1to033_x3_blur/best.pth",
        'blur': settings_lr_blur,
        'zoom': settings_zoom,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierSRGAN( params={
        #"arch": "srgan_2x2",
        #"model_path": "./models/srgan/weights/PSNR_inria_scale2.pth",
        "arch": "srgan",
        "model_path": "./models/srgan/weights/PSNR_inria_scale4.pth",
        "gpu": 0,
        "seed": 666,
        "zoom": settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierMSRN( params={
    'model':"MSRN_nonoise/MSRN_1to033/model_epoch_1500.pth",
    'compress': False,
    'add_noise': None,
    'zoom': settings_zoom,
    'blur': settings_lr_blur,
    'resize_preprocess': settings_resize_preprocess,
    'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierESRGAN( params={
        'model':"ESRGAN_1to033_x3_blur/net_g_latest.pth",
        'zoom':settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierCAR( params={
        "SCALE": 4,
        #"SCALE": 2,
        "model_dir": "./models/car/models",
        "gpu": 0,
        "zoom": settings_zoom,
        'blur': settings_lr_blur,
        'resize_preprocess': settings_resize_preprocess,
        'resize_postprocess': settings_resize_postprocess,
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        'model':"LIIF_blur/epoch-best.pth",
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

lst_folders_mod = [os.path.join(data_path+'#'+ds_modifier._get_name(),images_folder) for ds_modifier in ds_modifiers_list]
lst_labels_mod = [ds_modifier._get_name().replace("sisr+","").split("_")[0] for ds_modifier in ds_modifiers_list] # authomatic readout from folders

visual_comp(lst_folders_mod, lst_labels_mod, True, "comparison/")

print('Calculating similarity metrics...')

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
        device               = 'cpu',
        return_by_resolution = False,
        pyramid_batchsize    = win,
        use_liif_loader      = True,
        zoom                 = settings_zoom,
        blur                 = settings_lr_blur,
        resize_preprocess    = settings_resize_preprocess,
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

print('Calculating RER, MTF, FWHM Metrics...')

_ = experiment_info.apply_metric_per_run(
    SharpnessMetric(
        experiment_info,
        stride=16,
        ext="tif",
        parallel=True,
        metrics=["RER", "FWHM", "MTF"],
        njobs=4
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
    metrics=['ssim','psnr','swd','snr_median','snr_mean','fid','RER','FWHM','MTF'],
    dropna=False
)
print(df)
df.to_csv(f'./{experiment_name}_metrics.csv')
scatter_plots(df, [['ssim','psnr'],['fid','swd'],['RER','snr_mean'],['snr_mean','psnr'],['RER','MTF'],['RER','FWHM']], True, "plots/")

print('Calculating Regressor Quality Metrics...') #default configurations
_ = experiment_info.apply_metric_per_run(ScoreMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(RERMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(SNRMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(GaussianBlurMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(NoiseSharpnessMetrics(), ds_wrapper.json_annotations)
_ = experiment_info.apply_metric_per_run(GSDMetrics(), ds_wrapper.json_annotations)

df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=[
            "sigma",
            "snr",
            "rer",
            "sharpness",
            "scale",
            "score"
        ]
)
print(df)
df.to_csv(f'./{experiment_name}_regressor.csv')
scatter_plots(df, [['sigma','rer'],['sigma','sharpness'],['rer','snr'],['sharpness','rer'],['sigma','scale'],['snr','scale']], True, "plots/")

print("Comparing Metrics with Regressed Quality Metrics")
df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=[
            "ssim",
            "psnr",
            "swd",
            "snr_median",
            "snr_mean",
            "fid",
            "RER",
            "FWHM",
            "MTF",
            "sigma",
            "rer",
            "snr",
            "sharpness",
            "scale",
            "score"
        ]
)
print(df)
df.to_csv(f'./{experiment_name}.csv')
scatter_plots(df, [['sigma','RER'],['sigma','MTF'],['sigma','FWHM'],['rer','RER'],['rer','MTF'],['rer','FWHM'],['sharpness','RER'],['sharpness','MTF'],['sharpness','FWHM'],['snr','snr_mean'],['snr','psnr'],['scale','rer'],['scale','snr'],['score','psnr'],['score','ssim'],['score','swd'],['score','snr_median'],['score','RER'],['score','FWHM'],['score','MTF']], True, "plots/")

ev = ExperimentVisual(df, None)

ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="score",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="ssim",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="swd",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="fid",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="snr_mean",
    legend_var='snr_median',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="RER",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="FWHM",
    legend_var='RER',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="MTF",
    legend_var='RER',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="sigma",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="rer",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="snr",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="scale",
    legend_var='psnr',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="snr",
    legend_var='snr_mean',
    title=""
)
ev.visualize(
    plot_kind="bars",
    xvar="ds_modifier",
    yvar="rer",
    legend_var='RER',
    title=""
)
