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

with open("./resultats/results_test.pkl", "rb") as f:
    results_load = pickle.load(f)

# Création du tableau initial 
tableau_resultat = pd.DataFrame(index=results_load.keys())
tableau_resultat["slice_index"] = [results_load[s]["slice_index"] for s in results_load]

for sujet in tableau_resultat.index:

    sam_seg = results_load[sujet]['sam_seg']
    medsam_seg = results_load[sujet]['medsam_seg']
    gt = results_load[sujet]['gt']

    # Enlève les dimmensions de 1 retournés par les modèles (nb images traités en mê temps)
    sam_seg = sam_seg.squeeze()
    medsam_seg = medsam_seg.squeeze()

    # Ajout colone dice score 
    sam_dsc = fc.compute_dice_coefficient(gt > 0, sam_seg > 0)
    medsam_dsc = fc.compute_dice_coefficient(gt > 0, medsam_seg > 0)

    # Ajout colone precision, recall 
    precision_sam, recall_sam = fc.precision_recall(gt > 0, sam_seg > 0)
    precision_medsam, recall_medsam = fc.precision_recall(gt > 0, medsam_seg > 0)
    
    #print(gt.shape, sam_seg.shape)

    # Ajout colone HD100 
    HD100_sam = fc.compute_hd100(gt > 0, sam_seg > 0)
    HD100_medsam = fc.compute_hd100(gt > 0, medsam_seg > 0)

    # Ajout colone Average Surface Distance

    tableau_resultat.loc[sujet, "sam_dice"] = sam_dsc
    tableau_resultat.loc[sujet, "sam_precision"] = precision_sam
    tableau_resultat.loc[sujet, "sam_recall"] = recall_sam
    tableau_resultat.loc[sujet, "sam_HD100"] = HD100_sam

    tableau_resultat.loc[sujet, "medsam_dice"] = medsam_dsc
    tableau_resultat.loc[sujet, "medsam_precision"] = precision_medsam
    tableau_resultat.loc[sujet, "medsam_recall"] = recall_medsam
    tableau_resultat.loc[sujet, "medsam_HD100"] = HD100_medsam

print(tableau_resultat.columns)
tableau_resultat.to_csv('./resultats/resultats_tableau.csv', index=True)

