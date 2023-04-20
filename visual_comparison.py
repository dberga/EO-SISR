import os
import shutil
import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image as pil_image

# metric_comp
from iquaflow.experiments.experiment_visual import ExperimentVisual
import itertools

# plotSNE
from iquaflow.quality_metrics.dataloader import Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

### Image differences

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def normalize(arr1, arr2):
    """
    Linear normalization
    """
    arr1 = arr1.astype('float')
    arr2 = arr2.astype('float')
    for i in range(3):
        arr1[...,i] -= arr2[...,i]
        arr1[...,i] = abs(arr1[...,i])

    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            avg = (arr1[i][j][0] + arr1[i][j][1] + arr1[i][j][2])/3
            for k in range(3):
                arr1[i][j][k] = avg
    diff_mn = 0
#    diff_mx = np.amax(arr1[...,...,0])
    diff_mx = 10
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            (arr1[i][j][0], arr1[i][j][1], arr1[i][j][2]) = rgb(diff_mn, diff_mx, arr1[i][j][0])
    return arr1 # astype('uint8')

#########################
# Visual comparison
#########################

def scatter_plots(df, 
    metrics = [
        ['ssim','psnr'],
        ['fid','swd'],
    ],
    savefig = False,
    plots_folder = "plots/"):
    
    print("plotting" + plots_folder)
    print(metrics)
    
    # short names
    df['ds_modifier'] = [name.split("_")[0] for name in df['ds_modifier'].values]
    
    for pair_metrics in metrics:

        met1, met2 = pair_metrics

        fig, ax = plt.subplots()
        marker_lst = ["X","x","o","s","8","P","s","d"]

        for i in df.index:
            print(len(df['ds_modifier'][i])>1)
            if len(df['ds_modifier'][i])>1:
                if "LR" in df['ds_modifier'][i]:
                    marker = "x"
                elif "HR" in df['ds_modifier'][i]:
                    marker = 'X'
                else:
                    marker = 'o'

            ax.scatter(
                df[met1][i],
                df[met2][i],
                s=250.,
                marker=marker,
                label=df['ds_modifier'][i],
                alpha=0.5,
                edgecolors='none'
            )

        ax.set_xlabel(met1)
        ax.set_ylabel(met2)
        ax.legend(title='Algorithms', bbox_to_anchor=(.45, 1.15), loc="upper center") #bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        if savefig is True:
            os.makedirs(plots_folder,exist_ok=True)
            plt.savefig(os.path.join(plots_folder,"scatter_"+met1+"_"+met2+".png"),facecolor=fig.get_facecolor())
        else:
            plt.show()

def visual_comp(
    lst_folders = [
    "./Data/test-ds/test/",
    "./Data/test-ds#GT(LR)/test/",
    "./Data/test-ds#sisr+MSRN_MSRN_nonoise-MSRN_1to033-model_epoch_1500/test/",
    "./Data/test-ds#sisr+ESRGAN_ESRGAN_1to033_x3_blur-net_g_latest/test/",
    "./Data/test-ds#sisr+FSRCNN_test_scale3_FSRCNN_1to033_x3_blur-best/test/",
    "./Data/test-ds#sisr+LIIF_LIIF_config_test_liif_LIIF_blur-epoch-best/test/"
    ], 
    lst_labels = [
    "GT_HR",
    "GT_LR",
    "MSRN",
    "ESRGAN",
    "FSRCNN",
    "LIIF"
    ],
    savefig = False,
    comparison_folder = "comparison/",
    max_img_size = 128,
    max_enu = 100,
    ):

    # short names
    lst_labels = [label.split("_")[0] for label in lst_labels]

    lst_lst = [glob(fr"{os.path.join(folder,'*')}") for folder in lst_folders]
    print(''.join([label+'\t     ' for label in lst_labels]))

    for enu,fn in enumerate(lst_lst[0]):
        if enu>max_enu:
            break

        n_alg = len(lst_lst)
        #import pdb; pdb.set_trace()
        #[ print(f) for f in lst_lst[i] if os.path.basename(f)==os.path.basename(fn) for i in range( n_alg ) ]

        arr_lst = [
            # cv2.imread( [ 
            #     f for f in lst_lst[i]
            #     if os.path.basename(f)==os.path.basename(fn)
            # ][0])
            # if i<2 else 
            cv2.imread( [ 
                f for f in lst_lst[i]
                if os.path.basename(f)==os.path.basename(fn)
            ][0] )[...,::-1]
            for i in range( n_alg ) 
        ]
        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i])
            ax[i].axis('off')
        
        if savefig is True:
            os.makedirs(comparison_folder,exist_ok=True)
            plt.savefig(os.path.join(comparison_folder,os.path.basename(fn)+'.png')) # added png conversion
        else:
            plt.show()
        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            height = arr_lst[i].shape[0]
            width = arr_lst[i].shape[1]
            if (height<max_img_size) or (width<max_img_size):
                ax[i].imshow( arr_lst[i])
            else:
                ax[i].imshow( arr_lst[i][int(height*0.5)-int(max_img_size*0.5):int(height*0.5)+int(max_img_size*0.5),int(width*0.5)-int(max_img_size*0.5):int(width*0.5)+int(max_img_size*0.5),:]) # zoom in
            ax[i].axis('off')
        if savefig is True:
            os.makedirs(comparison_folder,exist_ok=True)
            plt.savefig(os.path.join(comparison_folder,"zoomed_"+os.path.basename(fn)+'.png')) # added png conversion
        else:
            plt.show()

        #####################image differences here
        comparison_folder_diffs = comparison_folder+"_diffs"
        os.makedirs(comparison_folder_diffs,exist_ok=True)
        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            height = arr_lst[i].shape[0]
            width = arr_lst[i].shape[1]
            if i == 0:
                ax[i].imshow( arr_lst[i])
            else:
                diff_img = normalize(arr_lst[0],arr_lst[i]).astype('uint8')
                ax[i].imshow( diff_img)
            ax[i].axis('off')
        if savefig is True:
            os.makedirs(comparison_folder,exist_ok=True)
            plt.savefig(os.path.join(comparison_folder_diffs,os.path.basename(fn)+'.png')) # added png conversion
        else:
            plt.show()
        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            height = arr_lst[i].shape[0]
            width = arr_lst[i].shape[1]
            if i == 0:
                if (height<max_img_size) or (width<max_img_size):
                    ax[i].imshow( arr_lst[i],cmap='rainbow')
                else:
                    ax[i].imshow( arr_lst[i][int(height*0.5)-int(max_img_size*0.5):int(height*0.5)+int(max_img_size*0.5),int(width*0.5)-int(max_img_size*0.5):int(width*0.5)+int(max_img_size*0.5),:],cmap='rainbow') # zoom in
            else:
                diff_img = normalize(arr_lst[0],arr_lst[i]).astype('uint8')
                if (diff_img.shape[0]<max_img_size) or (diff_img.shape[1]<max_img_size):
                    ax[i].imshow( diff_img,cmap='rainbow')
                else:
                    ax[i].imshow( diff_img[int(height*0.5)-int(max_img_size*0.5):int(height*0.5)+int(max_img_size*0.5),int(width*0.5)-int(max_img_size*0.5):int(width*0.5)+int(max_img_size*0.5),:],cmap='rainbow') # zoom in
            ax[i].axis('off')
        if savefig is True:
            os.makedirs(comparison_folder,exist_ok=True)
            plt.savefig(os.path.join(comparison_folder_diffs,"zoomed_"+os.path.basename(fn)+'.png')) # added png conversion
        else:
            plt.show()


def metric_comp(df, selected_metrics,savefig = False,plots_folder = "plots/"):

    # short names
    df['ds_modifier'] = [name.split("_")[0] for name in df['ds_modifier'].values]

    ev = ExperimentVisual(df, None)
    reference_metric = selected_metrics[0]
    for metric in selected_metrics:
        try:
            ev.visualize(
                plot_kind="bars",
                xvar="ds_modifier",
                yvar=metric,
                legend_var=reference_metric,
                title=""
            )
        except:
            print("visual comparison not working for" + metric)
    metrics_comparison = list(itertools.combinations(selected_metrics, 2))
    scatter_plots(df, metrics_comparison, savefig, plots_folder)

def plotSNE(dataset_name="./Data/test-ds", images_folder="./Data/test-ds/test/", img_size=(232,232),shm_limit=6e4, crop = True, savefig = False,plots_folder = "plots/"):
    # create data loader
    dataset = Dataset(
        "whole", # split
        dataset_name, # dataset name
        images_folder, # dataset images path
        num_crops = 1,
        crop_size = img_size,
        split_percent = 1.0,
        img_size = img_size,
    )
    if crop is True:
        dataset.__crop__(False) # only if images are of distinct size

    # set maximum batch size according to shared memory available
    max_batch_size = int(shm_limit/np.mean(img_size))
    if max_batch_size > dataset.__len__():
        max_batch_size = dataset.__len__()

    dataloader= DataLoader(
        dataset=dataset,
        batch_size=max_batch_size,  
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # get one tensor/array of all images
    xbatches=[x for bix,(filename, param, x, y) in enumerate(dataloader)]
    for batch_idx,x_data in enumerate(xbatches):
        # reshape each image array to 1 dimension
        if len(x_data.shape)>=4: # RGB
            x_data = np.reshape(x_data, [x_data.shape[0], x_data.shape[1]*x_data.shape[2]*x_data.shape[3]])
        else: # GRAY
            x_data = np.reshape(x_data, [x_data.shape[0], x_data.shape[1]*x_data.shape[2]])

        # calculate TSNE
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x_data)
        tx = z[:, 0]
        ty = z[:, 1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        '''
        # Create simple 2d tSNE scatter plot
        fig = plt.scatter(tx,ty)
        plt.xlabel("tx")
        plt.ylabel("ty")
        plt.title("batch "+str(batch_idx))
        if savefig is True:
            plt.savefig(os.path.join(plots_folder,"tSNE"+"_"+str(batch_idx)+"_"+dataset_name+".png"))
        else:
            plt.show()
        plt.clf()
        '''
        # Create 2d tSNE scatter plot with visualization of images

        width = 4000
        height = 3000
        max_dim = 300

        # get images paths
        images = os.listdir(images_folder)
        for idx,img_name in enumerate(images):
            images[idx] = os.path.join(images_folder,img_name)

        # create full scatter plot
        full_image = pil_image.new('RGBA', (width, height))
        for img, x, y in zip(images, tx, ty):
            tile = pil_image.open(img)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), pil_image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        plt.figure(figsize = (16,12))
        plt.imshow(full_image)
        plt.xlabel("tx")
        plt.ylabel("ty")
        plt.title("batch "+str(batch_idx))
        if savefig is True:
            plt.savefig(os.path.join(plots_folder,"visual_tSNE"+"_"+str(batch_idx)+"_"+dataset_name+".png"))
        else:
            plt.show()
        plt.clf()
