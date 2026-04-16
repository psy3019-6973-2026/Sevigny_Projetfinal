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
from code import utiles
from code import modeles
from pathlib import Path

def get_statistic_socre_on_dataset(data_path):
    
    # Valeurs à stocker 
    results = {}
    sam_predictor, med_sam_predictor, med_sam_model = fc.modeles.initialisation_modeles()
    print('Initialisation terminée')

    # Boucle qui run trough les participants et extrait paths 
    for subject_folder in sorted(os.scandir(data_path), key=lambda e: e.name):
        
        subject_id = subject_folder.name
        print('Commence pour', subject_id) # Savoir je suis rendue ou 

        # Path des fichiers
        scan_path = os.path.join(subject_folder.path, f"{subject_id}_t2.nii.gz")
        gt_path = os.path.join(subject_folder.path, f"{subject_id}_seg.nii.gz")

        if not os.path.exists(scan_path) or not os.path.exists(gt_path):
            print(f"Fichiers manquants pour {subject_id}, skipping...")
            continue
        
        slice_index = np.random.randint(10,140)
        scan_2d_og, gt_2d = fc.utiles.load_scan_2d(scan_path, gt_path, slice_index)

        # Vérifie qu'il y a une tumeur sur la slice originale ou dans 10 autres essais suivant
        slice_index, scan_2d_og, gt_2d = fc.utiles.verification_slice_tumeur(subject_id, scan_path, gt_path, slice_index, scan_2d_og, gt_2d)
        
        scan_2d = fc.utiles.sam_imput_format(scan_2d_og)  # après la vérification, pas avant

        # Obtenir la boite autour de la segmentation 
        gt_box = fc.utiles.get_bbox_from_mask(gt_2d)

        # Mettre dans le format nécésaire pour les modèles
        scan_2d_pre = fc.utiles.preprocess_scan(scan_2d)

        # Segmentation des deux modèles 
        sam_seg, medsam_seg = fc.modeles.get_2_both_seg_scan(scan_2d_pre, gt_2d, sam_predictor, med_sam_model)
        print('Segmentation terminée', subject_id)

        results[subject_id] = {
        "slice_index": slice_index,
        "image": scan_2d_og,
        "gt": gt_2d,
        "sam_seg": sam_seg,
        "medsam_seg": medsam_seg,
        }
    
    print('boucle terminée :)')

    # Sauvegarde 
    output_path = os.path.join("./results/df_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Résultats sauvegardés dans : {output_path}")

    return results

   
def continue_statistic_score_on_dataset(data_path, output_path, modeles_path):
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "resultats.pkl"

    # Charger un tableau existant ou partir de zéro
    if results_file.exists():
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        print(f"Tableau existant chargé : {len(results)} sujets déjà traités")
    else:
        results = {}
        print("Aucun tableau existant, départ du début")

    # Vérifie que le dossier de données n'est pas vide 
    entries = list(os.scandir(data_path))
    if not entries:
        print(f"Le dossier {data_path} est vide, rien à traiter.")
        return results
    
    sam_predictor, med_sam_predictor, med_sam_model = modeles.initialisation_modeles(modeles_path)
    print('Initialisation terminée')

    for subject_folder in sorted(os.scandir(data_path), key=lambda e: e.name):
        
        subject_id = subject_folder.name

        # Skip si déjà traité
        if subject_id in results:
            print(f"Sujet {subject_id} déjà présent, skipping...")
            continue

        print('Commence pour', subject_id)

        scan_path = os.path.join(subject_folder.path, f"{subject_id}_t2.nii.gz")
        gt_path = os.path.join(subject_folder.path, f"{subject_id}_seg.nii.gz")

        if not os.path.exists(scan_path) or not os.path.exists(gt_path):
            print(f"Fichiers manquants pour {subject_id}, skipping...")
            continue
        
        slice_index = np.random.randint(10, 140)
        scan_2d_og, gt_2d = utiles.load_scan_2d(scan_path, gt_path, slice_index)

        slice_index, scan_2d_og, gt_2d = utiles.verification_slice_tumeur(subject_id, scan_path, gt_path, slice_index, scan_2d_og, gt_2d)
        
        scan_2d = utiles.sam_imput_format(scan_2d_og)
        #gt_box = fc.utiles.get_bbox_from_mask(gt_2d)
        scan_2d_pre = utiles.preprocess_scan(scan_2d)

        sam_seg, medsam_seg = modeles.get_2_both_seg_scan(scan_2d_pre, gt_2d, sam_predictor, med_sam_model)
        print('Segmentation terminée', subject_id)

        results[subject_id] = {
            "slice_index": slice_index,
            "image": scan_2d_og,
            "gt": gt_2d,
            "sam_seg": sam_seg,
            "medsam_seg": medsam_seg,
        }

        # Sauvegarde incrémentale après chaque sujet
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

    print(f"Boucle terminée : {len(results)} sujets au total")
    print(f"Résultats sauvegardés dans : {results_file}")

    return results
