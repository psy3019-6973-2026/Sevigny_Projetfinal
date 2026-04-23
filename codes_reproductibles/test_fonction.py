import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

tableau_resultat = pd.read_csv('./resultats/resultats_tableau.csv', index_col=0)
#print(tableau_resultat)

# Mettre dans le bon format
df_sam = tableau_resultat[['sam_dice', 'sam_precision', 'sam_recall', 'sam_HD100']].copy()
df_sam.columns = ['dice', 'precision', 'recall', 'HD100']
df_sam['modele'] = 'SAM'

df_medsam = tableau_resultat[['medsam_dice', 'medsam_precision', 'medsam_recall', 'medsam_HD100']].copy()
df_medsam.columns = ['dice', 'precision', 'recall', 'HD100']
df_medsam['modele'] = 'MedSAM'

df_long = pd.concat([df_sam, df_medsam])

print(tableau_resultat.shape)

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

figure_comparaison_dice(tableau_resultat, save_dir='./visualisations')