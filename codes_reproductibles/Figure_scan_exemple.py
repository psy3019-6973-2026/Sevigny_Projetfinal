'''

Va avoir besoin : 
data 

Option de sélectionner, sinon fait au hasard : 
le sujet, la slice 

Produit la figure de segmentation 

'''

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
import fonctions as fc
from ipywidgets import interact
import importlib.util, sys

# À AUTOMATISER 
slice_index = 80

# Read scans and ground-truth masks with .nii.gz format O
scan_path = "../data/BraTS2021_00495_t2.nii.gz"
gt_path =   "../data/BraTS2021_00495_seg.nii.gz"

# Load fichiers 
scan_2d_og,gt_2d = fc.load_scan_2d(scan_path, gt_path, slice_index)

# Good format 
scan_2d = fc.sam_imput_format(scan_2d_og)

# Figure partielle - jsp si je veux 
# TO DO faire mieux pour sauvegarder dans la fonction, comme visu finale 
figure = fc.visualisation_seg_gt(scan_2d_og, gt_2d)
plt.savefig(f'./figures/Segmentation_originale.png', dpi=150)

'''

# Load des modèles
sam_predictor, med_sam_predictor, med_sam_model = fc.initialisation_modeles()

# Get the box 
print("Shape gt:", gt_2d.shape)
gt_box = fc.get_bbox_from_mask(gt_2d)

# Preprocess to get in the format needed by models : 
scan_2d_pre = fc.preprocess_scan(scan_2d)

# Segmentation de SAM 
sam_seg = fc.get_sam_seg(scan_2d_pre, gt_box, sam_predictor)
medsam_seg = fc.get_medsam_seg(scan_2d_pre, gt_box, med_sam_model) 

sam_dsc = fc.compute_dice_coefficient(gt_2d, sam_seg)
medsam_dsc = fc.compute_dice_coefficient(gt_2d, medsam_seg)

fc.visualisation_resultats(scan_2d_og, gt_2d, sam_seg, medsam_seg, gt_box, sam_dsc, medsam_dsc, save_path='figure_finale.png')
'''