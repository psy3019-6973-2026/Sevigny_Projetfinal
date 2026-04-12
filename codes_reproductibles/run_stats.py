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

def ajouter_stats_tableau(tableau, results_load) :
    for sujet in tableau.index:

        sam_seg = results_load[sujet]['sam_seg']
        medsam_seg = results_load[sujet]['medsam_seg']
        gt = results_load[sujet]['gt']

        # Ajout colone dice score 
        sam_dsc = fc.compute_dice_coefficient(gt > 0, sam_seg > 0)
        medsam_dsc = fc.compute_dice_coefficient(gt > 0, medsam_seg > 0)

        tableau.loc[sujet, "sam_dice"] = sam_dsc
        tableau.loc[sujet, "medsam_dice"] = medsam_dsc

    return tableau


def ajouter_colone_dice(tableau, results_load):

    
