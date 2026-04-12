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
from scipy.ndimage import distance_transform_edt


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
        
        # slice_index = np.random.randint(10,140)
        slice_index = 80 
        scan_2d_og,gt_2d = load_scan_2d(scan_path, gt_path, slice_index)
        scan_2d = sam_imput_format(scan_2d_og)

        gt_box = get_bbox_from_mask(gt_2d)
        
        # Vérifie qu'il y a une tumeur sur la slide originale ou dans 10 autres essais suivant
        # slice_index, image, gt = verification_slice_tumeur(subject_id, slice_index, image_3d, gt_3d)
        
        if np.sum(gt_2d) == 0:
            print(f"Attention: pas de tumeur pour {subject_id}")

        # Boite qui entoure la segmentation
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
    output_path = os.path.join("./results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

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

### STATISTIQUES

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


def precision_recall(pred, gt) :
    tp, fp, fn = get_tp_fp_fn(pred, gt)
    denom_p = tp + fp
    denom_r = tp + fn

    precision = (tp / denom_p) if denom_p > 0 else 1.0
    recall = (tp / denom_r) if denom_r > 0 else 1.0
    return precision, recall
 
def compute_hd100(mask_gt, mask_pred, voxel_spacing=None):
    mask_gt   = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        return float("nan")

    sampling = voxel_spacing if voxel_spacing is not None else tuple([1.0] * mask_gt.ndim)

    dist_pred_to_gt = distance_transform_edt(~mask_gt,   sampling=sampling)[mask_pred]
    dist_gt_to_pred = distance_transform_edt(~mask_pred, sampling=sampling)[mask_gt]

    return float(max(dist_pred_to_gt.max(), dist_gt_to_pred.max()))
