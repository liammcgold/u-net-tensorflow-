import tifffile as tif
import numpy as np
import affinities as a
import sys
import os
import glob

raw_dir="spirou_data/RAW"
gt_dir="spirou_data/GT"


#load file lists
raw_files=[1]*(len(os.listdir(raw_dir))-1)
n=0
for file in os.listdir(raw_dir):
    if file.endswith(".tiff"):
        raw_files[n]=file
        n += 1

raw_files=np.sort(raw_files)


gt_files=[1]*(len(os.listdir(gt_dir)))
n=0
for file in os.listdir(gt_dir):
    if file.endswith(".tiff"):
        gt_files[n]=file
        n += 1

gt_files=np.sort(gt_files)



#save tifs as array


raw_tiff_array=[1]*len(raw_files)
n=0
for file in raw_files:
    raw_tiff_array[n]=np.asarray(tif.imread(raw_dir+"/"+file),dtype=np.float32)
    n+=1

raw_tiff_array=np.asarray(raw_tiff_array,dtype=np.float32)
raw_tiff_array=raw_tiff_array*(1/255)


np.save("spir_raw",raw_tiff_array)








gt_tiff_array=[1]*len(gt_files)
n=0
for file in gt_files:
    gt_tiff_array[n]=np.asarray(tif.imread(gt_dir+"/"+file),dtype=np.float32)
    n+=1

gt_tiff_array=np.asarray(gt_tiff_array,dtype=np.float32)
np.save("spir_gt",gt_tiff_array)


tif.imsave("tiffs_spirou/raw_test",raw_tiff_array[300])
tif.imsave("tiffs_spirou/gt_test",gt_tiff_array[300])



#save affinity file
affinities=a.get_affins(gt_tiff_array)
np.save("spir_aff",affinities)


