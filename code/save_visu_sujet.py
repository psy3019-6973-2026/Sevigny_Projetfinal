import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import os
from code import utiles
from code import modeles
from pathlib import Path
import pickle

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

def plot_segmentation(sujet, figures_dir, scan_2d_pre, gt_2d, sam_seg, medsam_seg, gt_box,
                      sam_dsc, precision_sam, recall_sam,
                      medsam_dsc, precision_medsam, recall_medsam):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Scan original
    axs[0, 0].imshow(scan_2d_pre, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].text(0.5, 0.5, 'Scan original', fontsize=35, fontweight='bold', horizontalalignment='left', verticalalignment='top', color='white')

    # Scan + GT
    axs[0, 1].imshow(scan_2d_pre, cmap='gray')
    show_mask(gt_2d > 0, axs[0, 1])        
    axs[0, 1].axis('off')
    axs[0, 1].text(0.5, 0.5, 'Segmentation réelle', fontsize=35, fontweight='bold', horizontalalignment='left', verticalalignment='top', color='white')

    # SAM seg
    axs[1, 0].imshow(scan_2d_pre, cmap='gray')             
    show_mask(sam_seg, axs[1, 0])     
    show_box(gt_box, axs[1, 0])
    axs[1, 0].text(0.02, 0.98, 'Segmentation SAM', fontsize=30, fontweight='bold',
                   horizontalalignment='left', verticalalignment='top', color='white', transform=axs[1, 0].transAxes)
    axs[1, 0].text(0.02, 0.92, f'Dice Score = {sam_dsc:.4f}\nPrecision  = {precision_sam:.4f}\nRecall     = {recall_sam:.4f}',
                   fontsize=20, horizontalalignment='left', verticalalignment='top', color='white', transform=axs[1, 0].transAxes)
    axs[1, 0].axis('off')

    # MedSAM seg
    axs[1, 1].imshow(scan_2d_pre, cmap='gray')            
    show_mask(medsam_seg, axs[1, 1])     
    show_box(gt_box, axs[1, 1])
    axs[1, 1].text(0.02, 0.98, 'Segmentation MED-SAM', fontsize=30, fontweight='bold',
                   horizontalalignment='left', verticalalignment='top', color='white', transform=axs[1, 1].transAxes)
    axs[1, 1].text(0.02, 0.92, f'Dice Score = {medsam_dsc:.4f}\nPrecision  = {precision_medsam:.4f}\nRecall     = {recall_medsam:.4f}',
                   fontsize=20, horizontalalignment='left', verticalalignment='top', color='white', transform=axs[1, 1].transAxes)
    axs[1, 1].axis('off')

    fig.tight_layout()

    # Sauvegarde
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / f"{sujet}_segmentation.png"
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Figure sauvegardée : {save_path}")

    return fig

def visu(sujet, output_data, output_figure) :
    
    # Chemins 
    tableau = output_data / 'resultats.csv'
    scans_tableau = output_data / 'resultats.pkl'

    # Load résultats 
    with open(scans_tableau, "rb") as f: 
        scans_tableau = pickle.load(f)

    tableau_resultat = pd.read_csv(tableau, index_col=0)

    # Vérifications 
    if sujet not in scans_tableau:
        print(f"{sujet} n'a pas de segmentation valide dans le tableau")
        return
    
    if sujet not in tableau_resultat.index:
        print(f"{sujet} n'est pas dans le tableau de résultats statistique")
        return

    # Obtenir métriques 
    data = scans_tableau[sujet]

    # Get les images
    scan_2d_pre     = data["image"]
    gt_2d        = data["gt"]
    sam_seg   = data["sam_seg"]
    medsam_seg = data["medsam_seg"]
    gt_box = utiles.get_bbox_from_mask(gt_2d)

    # Get les métriques 
    medsam_dsc = tableau_resultat.loc[sujet, 'medsam_dice']
    precision_medsam = tableau_resultat.loc[sujet, 'medsam_precision']
    recall_medsam = tableau_resultat.loc[sujet, 'medsam_recall']
    sam_dsc = tableau_resultat.loc[sujet, 'sam_dice']
    precision_sam = tableau_resultat.loc[sujet, 'sam_precision']
    recall_sam = tableau_resultat.loc[sujet, 'sam_recall']

    plot_segmentation(sujet, output_figure, scan_2d_pre, gt_2d, sam_seg, medsam_seg, gt_box,
                  sam_dsc, precision_sam, recall_sam,
                  medsam_dsc, precision_medsam, recall_medsam)


