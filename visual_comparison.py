import os
import shutil
import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image as pil_image

#########################
# Visual comparison
#########################

def scatter_plots(df, 
    metrics = [
        ['ssim','psnr'],
        ['fid','swd'],
        ['rer_0','snr_0'],
        ['rer_0','rer'],
        ['snr_mean','snr'],
        ['sigma','rer'],
        ['sharpness','snr'],
    ],
    savefig = False,
    plots_folder = "plots/"):
    
    for pair_metrics in metrics:

        met1, met2 = pair_metrics

        fig, ax = plt.subplots()

        marker_lst = []

        for i in df.index:

            if not '#' in df['ds_modifier'][i]:
                marker = 'X'
            else:
                marker = 'o'

            ax.scatter(
                (
                    df[met1][i]
                    if met1!='rer_0'
                    else np.nanmean([df['rer_0'][i],df['rer_1'][i],df['rer_2'][i]])
                ),
                df[met2][i],
                s=250.,
                marker=marker,
                label=df['ds_modifier'][i],
                alpha=0.5,
                edgecolors='none'
            )

        ax.set_xlabel(('rer' if met1=='rer_0' else met1))
        ax.set_ylabel(met2)
        ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        if savefig is True:
            os.makedirs(plots_folder,exist_ok=True)
            plt.savefig(os.path.join(plots_folder,"scatter_"+met1+"_"+met2+".png"))
        else:
            plt.show()

def visual_comp(
    lst_folders = [
    "./Data/test-ds/test/",
    "./Data/test-ds#sisr+MSRN_MSRN_nonoise-MSRN_1to033-model_epoch_1500/test/",
    "./Data/test-ds#sisr+ESRGAN_ESRGAN_1to033_x3_blur-net_g_latest/test/",
    "./Data/test-ds#sisr+FSRCNN_test_scale3_FSRCNN_1to033_x3_blur-best/test/",
    "./Data/test-ds#sisr+LIIF_LIIF_config_test_liif_LIIF_blur-epoch-best/test/"
    ], 
    lst_labels = [
    "GT",
    "MSRN",
    "ESRGAN",
    "FSRCNN",
    "LIIF"
    ],
    savefig = False,
    comparison_folder = "comparison/",
    ):
    lst_lst = [glob(fr"{os.path.join(folder,'*')}") for folder in lst_folders]
    print(''.join([label+'\t\t' for label in lst_labels]))
    
    for enu,fn in enumerate(lst_lst[0]):
        if enu>20:
            break

        n_alg = len(lst_lst)

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
            plt.savefig(os.path.join(comparison_folder,os.path.basename(fn)))
        else:
            plt.show()

        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i][75:-75:,75:-75:,:])
            ax[i].axis('off')
        if savefig is True:
            os.makedirs(comparison_folder,exist_ok=True)
            plt.savefig(os.path.join(comparison_folder,"zoomed_"+os.path.basename(fn)))
        else:
            plt.show()

from iq_tool_box.quality_metrics.dataloader import Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

def plotSNE(dataset_name="test-ds", images_folder="./Data/test-ds/test/", img_size=(232,232), savefig = False,plots_folder = "plots/"):
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
    dataset.__crop__(True) # only if images are of distinct size
    dataloader= DataLoader(
        dataset=dataset,
        batch_size=dataset.__len__(), # one batch only
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # get one tensor/array of all images
    xbatches=[x for bix,(filename, param, x, y) in enumerate(dataloader)]
    x_data=xbatches[0] # one batch only

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

    # Create 2d tSNE scatter plot
    fig = plt.scatter(tx,ty)
    plt.xlabel("tx")
    plt.xlabel("ty")
    if savefig is True:
        plt.savefig(os.path.join(plots_folder,"tSNE"+"_"+dataset_name+".png"))
    else:
        plt.show()
    plt.clf()

    # Create 2d tSNE scatter plot with visualization of images

    width = 4000
    height = 3000
    max_dim = 100

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
    plt.xlabel("ty")
    if savefig is True:
        plt.savefig(os.path.join(plots_folder,"visual_tSNE"+"_"+dataset_name+".png"))
    else:
        plt.show()
    plt.clf()
