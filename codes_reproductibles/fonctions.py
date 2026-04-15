# Tous les imports du notebook, à faire le ménage 

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


# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

####### Essentiels

#Return a bounding box from ground-truth, this bounding box will be used as a prompt when inferencing
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

'''
def verification_slice_tumeur(subject_id, slice_index, image_3d, gt_3d, max_attempts=10):
    
    # Slice initiale
    original_index = slice_index
    image, gt = get_slice_pair(slice_index, image_3d, gt_3d)

    # Si pas de tumeur = chercher autre slice
    if np.sum(gt) == 0:
        print('Slice initiale sans tumeur', slice_index)
        print("Shape gt:", gt.shape)

        for _ in range(max_attempts):
            new_index = np.random.randint(10, 140)
            new_image, new_gt = get_slice_pair(new_index, image_3d, gt_3d)

            if np.sum(new_gt) > 0:
                print(f'Slice trouvée: {new_index}')
                print("Shape gt:", gt.shape)
                return new_index, new_image, new_gt

        # Si aucune slice valide retourner l'originale
        print(f"Aucune slice valide pour {subject_id}, retour à la slice initiale")
        return original_index, image, gt

    # Si la slice initiale est déjà valide
    print("Shape gt:", gt.shape)
    return slice_index, image, gt
'''

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

####### Modèles 

def initialisation_modeles() : 
    model_type = 'vit_b'
    sam_model_checkpoint = '../models/sam_vit_b_01ec64.pth'
    med_sam_model_checkpoint = '../models/sam_model_best.pth' # Not the original but matches the link 

    device = torch.device('cpu')

    sam_model = sam_model_registry[model_type](checkpoint=None)
    sam_model.load_state_dict(torch.load(sam_model_checkpoint, map_location=torch.device('cpu')))
    sam_model.to(device)

    med_sam_model = sam_model_registry[model_type](checkpoint=None)
    med_sam_model.load_state_dict(torch.load(med_sam_model_checkpoint, map_location=torch.device('cpu')))
    med_sam_model.to(device)

    sam_predictor = SamPredictor(sam_model)
    med_sam_predictor = SamPredictor(med_sam_model)

    print('Initialisation modèles terminée')
    return sam_predictor, med_sam_predictor, med_sam_model

# Needed input to both models 
def preprocess_scan(scan_2d):
    lower_bound, upper_bound = np.percentile(scan_2d, 0.5), np.percentile(scan_2d, 99.5)
    scan_2d_pre = np.clip(scan_2d, lower_bound, upper_bound)
    scan_2d_pre = (scan_2d_pre - np.min(scan_2d_pre))/(np.max(scan_2d_pre)-np.min(scan_2d_pre))*255.0
    scan_2d_pre[scan_2d==0] = 0
    scan_2d_pre = np.uint8(scan_2d_pre)
    return scan_2d_pre

def get_sam_seg(scan_2d, bbox_raw, sam_predictor) :
    # predict the segmentation mask using the original SAM model
    sam_predictor.set_image(scan_2d)
    sam_seg, _, _ = sam_predictor.predict(point_coords=None, box=bbox_raw, multimask_output=False)
    #print(sam_seg.shape) 
    print("Segmentation SAM terminée")
    return sam_seg

def get_medsam_seg(scan_2d, bbox_raw, med_sam_model):

    device = 'cpu'

    med_sam_transform = ResizeLongestSide(med_sam_model.image_encoder.img_size)
    resize_img = med_sam_transform.apply_image(scan_2d)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = med_sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, med_sam_model.image_encoder.img_size, med_sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

    with torch.no_grad():
        
        H, W, _ = scan_2d.shape
        # pre-compute the image embedding
        ts_img_embedding = med_sam_model.image_encoder(input_image)
        # convert box to 1024x1024 grid
        bbox = med_sam_transform.apply_boxes(bbox_raw, (H, W))
        #print(f'{bbox_raw=} -> {bbox=}')

        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = med_sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = med_sam_model.mask_decoder(
            image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
            image_pe=med_sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        medsam_seg = skimage.transform.resize(medsam_seg,(240,240))
        medsam_seg = skimage.util.img_as_ubyte(medsam_seg)
        #print(medsam_seg.shape)
        #print(medsam_seg)

        print("Segmentation MEDSAM terminée! :)")
        return medsam_seg

def get_2_both_seg_scan(scan_2d, gt, modele_sam, modele_medsam) : 

    gt_box = get_bbox_from_mask(gt)
    sam_seg = get_sam_seg(scan_2d, gt_box, modele_sam)
    medsam_seg = get_medsam_seg(scan_2d, gt_box, modele_medsam)

    return sam_seg, medsam_seg 

##### Boucle statistiques 

def get_statistic_socre_on_dataset(data_path):
    
    # Valeurs à stocker 
    results = {}
    sam_predictor, med_sam_predictor, med_sam_model = initialisation_modeles()
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
        scan_2d_og, gt_2d = load_scan_2d(scan_path, gt_path, slice_index)

        # Vérifie qu'il y a une tumeur sur la slice originale ou dans 10 autres essais suivant
        slice_index, scan_2d_og, gt_2d = verification_slice_tumeur(subject_id, scan_path, gt_path, slice_index, scan_2d_og, gt_2d)
        
        scan_2d = sam_imput_format(scan_2d_og)  # après la vérification, pas avant

        # Obtenir la boite autour de la segmentation 
        gt_box = get_bbox_from_mask(gt_2d)

        # Mettre dans le format nécésaire pour les modèles
        scan_2d_pre = preprocess_scan(scan_2d)

        # Segmentation des deux modèles 
        sam_seg, medsam_seg = get_2_both_seg_scan(scan_2d_pre, gt_2d, sam_predictor, med_sam_model)
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

def continue_statistic_score_on_dataset(data_path, existing_results_path=None):
    
    # Charger un tableau existant ou partir de zéro
    if existing_results_path is not None and os.path.exists(existing_results_path):
        with open(existing_results_path, "rb") as f:
            results = pickle.load(f)
        print(f"Tableau existant chargé : {len(results)} sujets déjà traités")
    else:
        results = {}
        print("Aucun tableau existant, départ du début")

    sam_predictor, med_sam_predictor, med_sam_model = initialisation_modeles()
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
        scan_2d_og, gt_2d = load_scan_2d(scan_path, gt_path, slice_index)

        slice_index, scan_2d_og, gt_2d = verification_slice_tumeur(subject_id, scan_path, gt_path, slice_index, scan_2d_og, gt_2d)
        
        scan_2d = sam_imput_format(scan_2d_og)
        gt_box = get_bbox_from_mask(gt_2d)
        scan_2d_pre = preprocess_scan(scan_2d)

        sam_seg, medsam_seg = get_2_both_seg_scan(scan_2d_pre, gt_2d, sam_predictor, med_sam_model)
        print('Segmentation terminée', subject_id)

        results[subject_id] = {
            "slice_index": slice_index,
            "image": scan_2d_og,
            "gt": gt_2d,
            "sam_seg": sam_seg,
            "medsam_seg": medsam_seg,
        }

        # Sauvegarde incrémentale après chaque sujet
        output_path = existing_results_path if existing_results_path else "./results/df_results.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

    print(f"Boucle terminée : {len(results)} sujets au total")
    print(f"Résultats sauvegardés dans : {output_path}")

    return results


###### Visualisations 

# Visualisations initiales 

def visualize_3d_scan(layer, scan_data):
    plt.figure(figsize=(10, 5))
    plt.imshow(scan_data[:, :, layer], cmap='gray')
    plt.axis('off')
    return layer

def visualize_3d_gt(layer, gt_data):
    plt.figure(figsize=(10, 5))
    plt.imshow(gt_data[:, :, layer], cmap='gray')
    plt.axis('off')
    return layer

def visualisation_seg_gt(scan_2d, gt_2d):
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    
    axs[0].imshow(scan_2d, cmap='gray')
    axs[0].set_title('Scan')
    axs[0].axis('off')
    
    axs[1].imshow(gt_2d, cmap='gray')
    axs[1].set_title('GT')
    axs[1].axis('off')
    
    fig.tight_layout()
    
    return fig

def visualisation_resultats(ori_scan_2d, gt_2d, sam_seg, medsam_seg, bbox_raw, sam_dsc, medsam_dsc, save_path='resultats_segmentation.png'):
    
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

    _, axs = plt.subplots(1, 3, figsize=(25, 25))

    axs[0].imshow(ori_scan_2d)
    show_mask(gt_2d>0, axs[0])
    axs[0].axis('off')
    axs[0].text(0.5, 0.5, 'Ground Truth', fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')

    axs[1].imshow(ori_scan_2d)
    show_mask(sam_seg, axs[1])
    show_box(bbox_raw, axs[1])
    axs[1].text(0.5, 0.5, 'SAM DSC: {:.4f}'.format(sam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
    axs[1].axis('off')

    axs[2].imshow(ori_scan_2d)
    show_mask(medsam_seg, axs[2])
    show_box(bbox_raw, axs[2])
    axs[2].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
    axs[2].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # libère la mémoire sans afficher

    print('Figure finale sauvergardée :)')

def save_figure_gt(results_load, sujet, save_dir="."):

    if sujet not in results_load:
        raise ValueError(f"Sujet {sujet} non trouvé dans results")

    image = results_load[sujet]["image"]
    gt = results_load[sujet]["gt"]

    fig = visualisation_seg_gt(image, gt)

    filepath = os.path.join(save_dir, f"figure_gt_{sujet}.png")

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure sauvegardée : {filepath}")

def save_figure_resultats(results_load, tableau_metriques, sujet, save_dir="."):

    if sujet not in results_load:
        raise ValueError(f"Sujet {sujet} non trouvé dans results")

    data = results_load[sujet]

    image     = data["image"]
    gt        = data["gt"]
    sam_seg   = data["sam_seg"]
    medsam_seg = data["medsam_seg"]

    # Recalcule les métriques depuis les segmentations stockées
    bbox_raw  = get_bbox_from_mask(gt)
    sam_dsc = tableau_metriques.loc[sujet, 'sam_dice']
    medsam_dsc = tableau_metriques.loc[sujet, 'medsam_dice']

    filepath = os.path.join(save_dir, f"figure_resultats_{sujet}.png")

    visualisation_resultats(
        ori_scan_2d=image,
        gt_2d=gt,
        sam_seg=sam_seg,
        medsam_seg=medsam_seg,
        bbox_raw=bbox_raw,
        sam_dsc=sam_dsc,
        medsam_dsc=medsam_dsc,
        save_path=filepath
    )

    print(f"Figure sauvegardée : {filepath}")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def figure_complete(scan, gt, medsam_seg, sam_seg, bbox_raw, sam_dsc, medsam_dsc):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Scan original
    axs[0, 0].imshow(scan, cmap='gray')
    axs[0, 0].set_title('Scan')
    axs[0, 0].axis('off')

    # Scan + GT
    axs[0, 1].imshow(scan)
    show_mask(gt > 0, axs[0, 1])          # ← axs[0, 1] au lieu de axs[0]
    axs[0, 1].axis('off')
    axs[0, 1].set_title('GT')
    axs[0, 1].text(0.5, 0.5, 'Ground Truth', fontsize=30,
                   horizontalalignment='left', verticalalignment='top', color='yellow')

    # SAM seg
    axs[1, 0].imshow(scan)                # ← axs[1, 0] pour le 3e panneau
    show_mask(sam_seg, axs[1, 0])         # ← axs[1, 0] au lieu de axs[1]
    show_box(bbox_raw, axs[1, 0])
    axs[1, 0].text(0.5, 0.5, 'SAM DSC: {:.4f}'.format(sam_dsc), fontsize=30,
                   horizontalalignment='left', verticalalignment='top', color='yellow')
    axs[1, 0].axis('off')

    # MedSAM seg
    axs[1, 1].imshow(scan)                # ← axs[1, 1] au lieu de axs[1, 2]
    show_mask(medsam_seg, axs[1, 1])      # ← axs[1, 1] au lieu de axs[2]
    show_box(bbox_raw, axs[1, 1])
    axs[1, 1].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc), fontsize=30,
                   horizontalalignment='left', verticalalignment='top', color='yellow')
    axs[1, 1].axis('off')

    fig.tight_layout()

    return fig

def save_figure_complete(results_load, tableau_metriques, sujet, save_dir="."):

    if sujet not in results_load:
        raise ValueError(f"Sujet {sujet} non trouvé dans results")

    data = results_load[sujet]

    image     = data["image"]
    gt        = data["gt"]
    sam_seg   = data["sam_seg"]
    medsam_seg = data["medsam_seg"]

    # Recalcule les métriques depuis les segmentations stockées
    bbox_raw  = get_bbox_from_mask(gt)
    sam_dsc = tableau_metriques.loc[sujet, 'sam_dice']
    medsam_dsc = tableau_metriques.loc[sujet, 'medsam_dice']

    fig = figure_complete(image, gt, medsam_seg, sam_seg, bbox_raw, sam_dsc, medsam_dsc)

    filepath = os.path.join(save_dir, f"figure_resultats_{sujet}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure sauvegardée : {filepath}")

def figure_comparaison_dice(tableau_resultat, save_dir="./visualisations"):
    
    couleurs = {'SAM': '#378ADD', 'MedSAM': '#D85A30'}

    diff = tableau_resultat['medsam_dice'] - tableau_resultat['sam_dice']
    subjects = tableau_resultat.index.tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Parallel coordinates ──────────────────────────────────────────────────
    for subj in subjects:
        sam_val    = tableau_resultat.loc[subj, 'sam_dice']
        medsam_val = tableau_resultat.loc[subj, 'medsam_dice']
        d = medsam_val - sam_val
        color  = '#D85A30' if d > 0 else '#378ADD'   # MedSAM better → coral, SAM better → blue
        alpha  = 0.3 + 0.6 * abs(d)                  # bigger difference = more opaque
        ax1.plot([0, 1], [sam_val, medsam_val],
                 color=color, alpha=alpha, linewidth=1.2, zorder=2)

    # Means
    mu_sam    = tableau_resultat['sam_dice'].mean()
    mu_medsam = tableau_resultat['medsam_dice'].mean()
    ax1.plot([0, 1], [mu_sam, mu_medsam],
             color='black', linewidth=2.5, linestyle='--', zorder=5, label='Moyenne')
    ax1.scatter([0, 1], [mu_sam, mu_medsam],
                color='black', s=60, zorder=6)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['SAM', 'MedSAM'], fontsize=13)
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Parallel coordinates — Dice par sujet', fontsize=12,
                  fontweight='bold', pad=12)

    legend_elems = [
        mlines.Line2D([], [], color='#D85A30', linewidth=1.5, label='MedSAM meilleur'),
        mlines.Line2D([], [], color='#378ADD', linewidth=1.5, label='SAM meilleur'),
        mlines.Line2D([], [], color='black',   linewidth=2,   linestyle='--', label='Moyenne'),
    ]
    ax1.legend(handles=legend_elems, fontsize=9, frameon=False, loc='lower right')

    # ── Lollipop — MedSAM − SAM ───────────────────────────────────────────────
    order   = diff.sort_values().index          # sort by difference
    y_pos   = np.arange(len(order))
    diff_sorted = diff[order]

    for i, (subj, val) in enumerate(diff_sorted.items()):
        color = '#D85A30' if val > 0 else '#378ADD'
        ax2.plot([0, val], [i, i], color=color, linewidth=1.4, zorder=2)
        ax2.scatter(val, i, color=color, s=45, zorder=3)

    ax2.axvline(0, color='black', linewidth=0.8, linestyle='-', zorder=1)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(order, fontsize=8)
    ax2.set_xlabel('Δ Dice  (MedSAM − SAM)', fontsize=11)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Lollipop — différence Dice par sujet', fontsize=12,
                  fontweight='bold', pad=12)

    legend_elems2 = [
        mlines.Line2D([], [], color='#D85A30', linewidth=1.5,
                      marker='o', markersize=5, label='MedSAM meilleur (Δ > 0)'),
        mlines.Line2D([], [], color='#378ADD', linewidth=1.5,
                      marker='o', markersize=5, label='SAM meilleur (Δ < 0)'),
    ]
    ax2.legend(handles=legend_elems2, fontsize=9, frameon=False, loc='lower right')

    fig.tight_layout()

    filepath = os.path.join(save_dir, "figure_comparaison_dice.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {filepath}")

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

    dist_pred_to_gt = distance_transform_edt(~mask_gt,   sampling=sampling)[mask_pred]
    dist_gt_to_pred = distance_transform_edt(~mask_pred, sampling=sampling)[mask_gt]

    return dist_pred_to_gt.mean(), dist_gt_to_pred.mean()

#mean( d(p, S_gt) ) pour p ∈ S_pred + d(q, S_pred) pour q ∈ S_gt