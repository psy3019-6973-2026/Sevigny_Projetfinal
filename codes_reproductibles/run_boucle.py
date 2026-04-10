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

'''

Devrait faire la boucle qui passe à travers tous les participants et extrait le 
tableau de données statistiques 

'''

# Boucle avec la bonne structure, qui produit un dt : 

results = fc.get_statistic_socre_on_dataset("../data_test")

#with open("results.pkl", "wb") as f:
#    pickle.dump(results, f)
