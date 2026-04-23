import numpy as np
import os
join = os.path.join
import nibabel as nib
import numpy as np
from ipywidgets import interact
#import seaborn as sns
import pickle
from ipywidgets import interact
from code import utiles
from code import modeles
from pathlib import Path

def continue_statistic_score_on_dataset(data_path, output_path, modeles_path):

    MAX_NEW_SUBJECTS = 20
    
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

    all_subjects = sorted(os.scandir(data_path), key=lambda e: e.name)
    restant = [e for e in all_subjects if e.name not in results]
    
    print(f"Sujets restants à traiter : {len(restant)}")

    for subject_folder in restant[:MAX_NEW_SUBJECTS]:
        
        subject_id = subject_folder.name
        print('Commence pour', subject_id)

        # Chemins à partir du sujet 
        scan_path = os.path.join(subject_folder.path, f"{subject_id}_t2.nii.gz")
        gt_path = os.path.join(subject_folder.path, f"{subject_id}_seg.nii.gz")

        # Vérifie qu'il y a des données pour le sujet 
        if not os.path.exists(scan_path) or not os.path.exists(gt_path):
            print(f"Fichiers manquants pour {subject_id}, skipping...")
            continue

        # Load des fichiers
        scan_obj = nib.load(scan_path)
        gt_obj = nib.load(gt_path)
        scan_2d_og = scan_obj.get_fdata(dtype=np.float32)
        gt_data = gt_obj.get_fdata(dtype=np.float32)

        # Couche analysé, aléatoire selon le random seed du id 
        slice_index = utiles.get_slice_index(subject_id, gt_data) 

        '''
        # Obtient la seed random 
        subject_number = int(subject_id.split("_")[1])
        rng = np.random.default_rng(seed=subject_number)
        
        # Obtient la slice de tumeur parmis l'ensemble des slices possibles 
        tumor_slices = np.where(gt_data.any(axis=(0, 1)))[0]
        slice_index = rng.integers(tumor_slices.min(), tumor_slices.max())
        '''

        # Preprocess et mettre dans le bon format 
        scan_2d_og,gt_2d = utiles.get_slice_pair(slice_index, scan_2d_og, gt_data)
        scan_2d = utiles.sam_imput_format(scan_2d_og)
        scan_2d_pre = utiles.preprocess_scan(scan_2d)

        # Segmentation
        sam_seg, medsam_seg = modeles.get_2_both_seg_scan(scan_2d_pre, gt_2d, sam_predictor, med_sam_model)
        print('Segmentation terminée', subject_id)

        # Store les résultats 
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

    restant = len(restant) - MAX_NEW_SUBJECTS
    print(f"Sujets restants après cette session : {max(0, restant)}")
    print(f"Boucle terminée : {len(results)} sujets au total")
    print(f"Résultats sauvegardés dans : {results_file}")

    return results
