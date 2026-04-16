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
from pathlib import Path


####### Modèles 

def initialisation_modeles(modeles_path) : 
    model_type = 'vit_b'
    #sam_model_checkpoint = '../models/sam_vit_b_01ec64.pth'
    #med_sam_model_checkpoint = '../models/sam_model_best.pth' # Not the original but matches the link

    sam_model_checkpoint = Path(modeles_path) / 'sam_vit_b_01ec64.pth'
    med_sam_model_checkpoint = Path(modeles_path) / 'sam_model_best.pth' 

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

    gt_box = utiles.get_bbox_from_mask(gt)
    sam_seg = get_sam_seg(scan_2d, gt_box, modele_sam)
    medsam_seg = get_medsam_seg(scan_2d, gt_box, modele_medsam)

    return sam_seg, medsam_seg 
