from PIL import Image
import os

for filename in os.listdir(os.getcwd()):
   if filename.endswith('.tif'):
      im=Image.open(filename)
      w,h=im.size
      im=im.crop((250,259,w-200,h-250))
      im.save(filename[:-4]+".png")