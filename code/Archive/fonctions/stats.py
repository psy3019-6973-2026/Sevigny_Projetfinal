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

### STATISTIQUES
# Toujours gt, pred

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()  # |A| + |B| → nb de pixels dans chaque masque
    if volume_sum == 0:
        return 1.0                                 # cas spécial : les deux masques sont vides
    volume_intersect = (mask_gt & mask_pred).sum() # |A ∩ B| → pixels en commun
    return 2 * volume_intersect / volume_sum       # formule DICE

def get_tp_fp_fn(mask_gt, mask_pred):
    '''
    Plusieurs métriques utilisent ces chiffres ensuite 
    TP (True Positive): represents the number of FTU pixels that have been properly classified as FTU
    FP (False Positive): represents the number background pixels being misclassified as FTUs (due to misalignment)
    FN (False Negative): represents the number of FTU pixels being misclassified as background
    '''
    mask_gt   = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    tp = (mask_gt & mask_pred).sum()
    fp = (~mask_gt & mask_pred).sum() # Inverse du masque (donc pas tumeur) mais le modèle dit que non
    fn = (mask_gt & ~mask_pred).sum() # Inverse du prédit (pas tumeur) mais le modèle dit que oui

    return tp, fp, fn

def precision_recall(gt, pred) :
    tp, fp, fn = get_tp_fp_fn(pred, gt)
    denom_p = tp + fp
    denom_r = tp + fn

    precision = (tp / denom_p) if denom_p > 0 else 1.0
    recall = (tp / denom_r) if denom_r > 0 else 1.0
    return precision, recall
 
def compute_hd100(mask_gt, mask_pred, voxel_spacing=None):
    print(mask_gt.shape, mask_pred.shape)
    
    mask_gt   = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        return float("nan")

    sampling = voxel_spacing if voxel_spacing is not None else tuple([1.0] * mask_gt.ndim)

    surf_gt = mask_gt ^ binary_erosion(mask_gt)
    surf_pred = mask_pred ^ binary_erosion(mask_pred)

    dt_gt = distance_transform_edt(~mask_gt, sampling=sampling)
    dt_pred = distance_transform_edt(~mask_pred, sampling=sampling)

    dist_pred_to_gt = dt_gt[surf_pred]
    dist_gt_to_pred = dt_pred[surf_gt]

    return float(max(dist_pred_to_gt.max(), dist_gt_to_pred.max()))

def compute_avg_surf_dist(mask_gt, mask_pred, voxel_spacing=None):
    print(mask_gt.shape, mask_pred.shape)
    
    mask_gt   = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        return float("nan")

    sampling = voxel_spacing if voxel_spacing is not None else tuple([1.0] * mask_gt.ndim)

    # Gets only the surface voxels 
    surf_gt = mask_gt ^ binary_erosion(mask_gt)
    surf_pred = mask_pred ^ binary_erosion(mask_pred)

    # distance maps
    dt_gt = distance_transform_edt(~mask_gt, sampling=sampling)
    dt_pred = distance_transform_edt(~mask_pred, sampling=sampling)

    # distances surface → surface
    dist_pred_to_gt = dt_gt[surf_pred]
    dist_gt_to_pred = dt_pred[surf_gt]

    return dist_pred_to_gt.mean(), dist_gt_to_pred.mean()

