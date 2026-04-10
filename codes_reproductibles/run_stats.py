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

with open("results.pkl", "rb") as f:
    results_load = pickle.load(f)

print(list(results_load.keys()))  # les sujets
print(list(next(iter(results_load.values())).keys()))  # les variables

tableau_resultat = pd.DataFrame(index=results_load.keys())
tableau_resultat["slice_index"] = [results_load[s]["slice_index"] for s in results_load]
# print(tableau_resultat)

for sujet in results_load.keys() :
    
    sam_seg = results_load[sujet]['sam_seg']
    medsam_seg = results_load[sujet]['medsam_seg']
    gt = results_load[sujet]['gt']

    sam_dsc = fc.compute_dice_coefficient(gt>0, sam_seg>0)
    medsam_dsc = fc.compute_dice_coefficient(gt>0, medsam_seg>0)

    tableau_resultat.loc[sujet, "sam_dice"] = sam_dsc
    tableau_resultat.loc[sujet, "medsam_dice"] = medsam_dsc

print(tableau_resultat)

# TO DO 
'''
Faire une visualisation à partir d'un sujet du tableau 
'''

sujet = 'BraTS2021_00002'
ori_scan_2d = tableau_resultat.loc[sujet, "image"] 
gt_2d = tableau_resultat.loc[sujet, "gt"] 
bbox_raw = fc.get_bbox_from_mask(tableau_resultat.loc[sujet, "gt"])
sam_seg = tableau_resultat.loc[sujet, "sam_seg"] 
medsam_seg = tableau_resultat.loc[sujet, "medsam_seg"] 
sam_dsc = fc.compute_dice_coefficient(gt_2d, sam_seg)
medsam_dsc = fc.compute_dice_coefficient(gt_2d, medsam_dsc)

fc.visualisation_resultats(ori_scan_2d, gt_2d, sam_seg, medsam_seg, bbox_raw, sam_dsc, medsam_dsc, save_path='resultats_segmentation_tableau.png')