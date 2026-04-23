import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
#from tqdm import tqdm
#import torch
#import torchvision
#from torch.utils.data import Dataset, DataLoader
#import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
#import skimage
# MOI ajout des imports qui sont partout dans le doc ici : 
#import tarfile
import nibabel as nib
#import glob
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import seaborn as sns
#import pickle
#import pandas as pd # Ajouté pour les stats
from ipywidgets import interact
from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.lines as mlines

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])

def get_slice_index(sujet, scan_gt) : 
        
        # Obtient la seed random 
        subject_number = int(sujet.split("_")[1])
        rng = np.random.default_rng(seed=subject_number)
        
        # Obtient la slice de tumeur parmis l'ensemble des slices possibles 
        tumor_slices = np.where(scan_gt.any(axis=(0, 1)))[0]
        slice_index = rng.integers(tumor_slices.min(), tumor_slices.max())

        return slice_index

def get_slice_pair(layer, scan_data, gt_data):
    return scan_data[:, :, layer],gt_data[:, :, layer]

#re-format to match required shape of SAM input
def sam_imput_format(scan_2d) : 
    if scan_2d.shape[-1]>3 and len(scan_2d.shape)==3:
        scan_2d = scan_2d[:,:,:3]
    if len(scan_2d.shape)==2:
        scan_2d = np.repeat(scan_2d[:,:,None], 3, axis=-1)
    return scan_2d

def preprocess_scan(scan_2d):
    lower_bound, upper_bound = np.percentile(scan_2d, 0.5), np.percentile(scan_2d, 99.5)
    scan_2d_pre = np.clip(scan_2d, lower_bound, upper_bound)
    scan_2d_pre = (scan_2d_pre - np.min(scan_2d_pre))/(np.max(scan_2d_pre)-np.min(scan_2d_pre))*255.0
    scan_2d_pre[scan_2d==0] = 0
    scan_2d_pre = np.uint8(scan_2d_pre)
    return scan_2d_pre
