import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import skimage
# MOI ajout des imports qui sont partout dans le doc ici : 
import tarfile
import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import seaborn as sns
import pickle
import pandas as pd # Ajouté pour les stats
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

def get_slice_pair(layer, scan_data, gt_data):
    return scan_data[:, :, layer],gt_data[:, :, layer]

#re-format to match required shape of SAM input
def sam_imput_format(scan_2d) : 
    if scan_2d.shape[-1]>3 and len(scan_2d.shape)==3:
        scan_2d = scan_2d[:,:,:3]
    if len(scan_2d.shape)==2:
        scan_2d = np.repeat(scan_2d[:,:,None], 3, axis=-1)
    return scan_2d

def load_scan_2d(scan_path, gt_path, slice_index) : 
    scan_obj = nib.load(scan_path)
    gt_obj = nib.load(gt_path)

    scan_data = scan_obj.get_fdata()
    gt_data = gt_obj.get_fdata()

    scan_2d,gt_2d = get_slice_pair(slice_index, scan_data, gt_data)

    return scan_2d,gt_2d

def preprocess_scan(scan_2d):
    lower_bound, upper_bound = np.percentile(scan_2d, 0.5), np.percentile(scan_2d, 99.5)
    scan_2d_pre = np.clip(scan_2d, lower_bound, upper_bound)
    scan_2d_pre = (scan_2d_pre - np.min(scan_2d_pre))/(np.max(scan_2d_pre)-np.min(scan_2d_pre))*255.0
    scan_2d_pre[scan_2d==0] = 0
    scan_2d_pre = np.uint8(scan_2d_pre)
    return scan_2d_pre

def verification_slice_tumeur(subject_id, scan_path, gt_path, slice_index, scan_2d, gt_2d, max_attempts=10):

    if np.sum(gt_2d) > 0:
        print('Slice initiale fonctionne', slice_index)
        return slice_index, scan_2d, gt_2d

    # Seulement ici on charge le 3D
    print(f'Slice initiale sans tumeur ({slice_index}), chargement volume 3D...')
    scan_obj = nib.load(scan_path)
    gt_obj = nib.load(gt_path)
    image_3d = scan_obj.get_fdata()
    gt_3d = gt_obj.get_fdata()

    for _ in range(max_attempts):
        new_index = np.random.randint(10,140)
        new_image, new_gt = get_slice_pair(new_index, image_3d, gt_3d)
        if np.sum(new_gt) > 0:
            print(f'Slice trouvée: {new_index}')
            return new_index, new_image, new_gt

    print(f"Aucune slice valide pour {subject_id}, retour à la slice initiale")
    return slice_index, scan_2d, gt_2d
