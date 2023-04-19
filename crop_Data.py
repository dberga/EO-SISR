import os
import sys
from PIL import Image
from random import randrange


n = len(sys.argv)
argumentList = sys.argv[1:]
if len(argumentList)>0:
    dataset_dir = argumentList[0]
else:
    dataset_dir='./Data/xview_short/images'
    #dataset_dir='./Data/deepglobe_short/images'
    #dataset_dir='./Data/inria-aid_short/train_images'
    #dataset_dir='./Data/inria-aid_short/test_images'
    #dataset_dir='./Data/inria-aid_short/val_images'

crops_dir = dataset_dir+'_crops'
os.makedirs(crops_dir,exist_ok=True)

def autocrop(pil_img,matrix=250,sample=1): #pct_focus=0.3, matrix_HW_pct=0.3,
    """
    random crop from an input image
    Args:
        - pil_img
        - pct_focus(float): PCT of margins to remove based on image H/W
        - matrix_HW_pct(float): crop size in PCT based on image Height
        - sample(int): number of random crops to return
    returns:
        - crop_list(list): list of PIL cropped images
    """
    x, y = pil_img.size
    #img_focus = pil_img.crop((x*pct_focus, y*pct_focus, x*(1-pct_focus), y*(1-pct_focus)))
    #x_focus, y_focus = img_focus.size
    #matrix = round(matrix_HW_pct*y_focus)
    crop_list = []
    for i in range(sample):
        x1 = randrange(0, x - matrix)
        y1 = randrange(0, y - matrix)
        cropped_img = pil_img.crop((x1, y1, x1 + matrix, y1 + matrix))
        #display(cropped_img)
        crop_list.append(cropped_img)
    return crop_list

list_images=os.listdir(dataset_dir)
for idx,name in enumerate(list_images):
    img = Image.open(dataset_dir+'/'+name)
    print(name)
    crop_list=autocrop(img,256,10)
    print(len(crop_list))
    for c,crop in enumerate(crop_list):
        out_file = f'{crops_dir}/crop{c}_{name}'
        print(out_file)
        crop.save(out_file)



