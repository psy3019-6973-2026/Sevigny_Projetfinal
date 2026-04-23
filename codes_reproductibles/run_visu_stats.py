import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import fonctions as fc

tableau_resultat = pd.read_csv('./resultats/resultats_tableau.csv', index_col=0)
print(tableau_resultat.columns)

# Mettre dans le bon format
df_sam = tableau_resultat[['sam_dice', 'sam_precision', 'sam_recall', 'sam_HD100', 'sam_AVG_seg2gt', 'sam_AVG_gt2seg']].copy()
df_sam.columns = ['dice', 'precision', 'recall', 'HD100', 'AVG_seg2gt', 'AVG_gt2seg']
df_sam['modele'] = 'SAM'

df_medsam = tableau_resultat[['medsam_dice', 'medsam_precision', 'medsam_recall', 'medsam_HD100', 'medsam_AVG_seg2gt', 'medsam_AVG_gt2seg']].copy()
df_medsam.columns = ['dice', 'precision', 'recall', 'HD100', 'AVG_seg2gt', 'AVG_gt2seg']
df_medsam['modele'] = 'MedSAM'

df_long = pd.concat([df_sam, df_medsam])

print(df_long.columns)

# Figure 1 : Dice 
#fc.figure_dice(df_long)

# Figure 2 : Precision, recall 
#fc.figure_2_variables(df_long, 'precision', 'recall')

# Figure 4 : comparaison entre modèles 
#fc.figure_comparaison_dice(tableau_resultat, save_dir='./figures')

# Figure 4 : AVG surface dist 
fc.figure_2_variables(df_long, 'AVG_seg2gt', 'AVG_gt2seg')


'''
# Figure 2 : 

fig, ax = plt.subplots(figsize=(6, 5))

df_pr = df_long.melt(id_vars='modele', value_vars=['precision', 'recall'],
                     var_name='metrique', value_name='valeur')

sns.boxplot(data=df_pr, x='metrique', y='valeur', hue='modele',
            palette={'SAM': '#378ADD', 'MedSAM': '#D85A30'},
            fill=False, linewidth=1.5, fliersize=0, ax=ax)

sns.stripplot(data=df_pr, x='metrique', y='valeur', hue='modele',
              palette={'SAM': '#378ADD', 'MedSAM': '#D85A30'},
              alpha=0.7, jitter=True, dodge=True, legend=False,
              size=5, ax=ax)

ax.set_xlabel('')
ax.set_ylabel('Score', fontsize=12)
ax.set_xticklabels(['Precision', 'Recall'], fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('Mesure de Précision et Recall pour Med-sam et sam', fontsize=12, loc='center', pad=20, fontweight='bold')
ax.yaxis.set_tick_params(labelsize=8)

# Grille discrète
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True)

# Bordures
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Légende
ax.legend(title='Modèles :', fontsize=11, frameon=False)
legend = ax.legend(title='Modèles :', fontsize=11, frameon=False)
legend.get_title().set_horizontalalignment('left')

plt.savefig('./figures/boxplot_precision_recall.png', dpi=300, bbox_inches='tight')
print('precision_recall sauvegardé')
#plt.show()
plt.close()

# Figure 3 : HD100 

fig, ax = plt.subplots(figsize=(5, 5))

modeles = ['SAM', 'MedSAM']
couleurs = {'SAM': '#378ADD', 'MedSAM': '#D85A30'}

# --- Boxplot avec transparence via boxprops ---
for i, modele in enumerate(modeles):
    vals = df_long[df_long['modele'] == modele][['modele', 'HD100']]
    bp = ax.boxplot(
        vals['HD100'],
        positions=[i],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color=couleurs[modele], linewidth=1.5),
        capprops=dict(color=couleurs[modele], linewidth=1.5),
        boxprops=dict(facecolor=couleurs[modele] + '40',   # '40' ≈ 25% opacité
                      edgecolor=couleurs[modele], linewidth=1.5),
        flierprops=dict(marker=''),
        showfliers=False
    )

    # Ligne de moyenne tiretée
    mu = vals['HD100'].mean()
    ax.hlines(mu, i - 0.25, i + 0.25,
              colors=couleurs[modele], linewidths=1.5,
              linestyles='--', zorder=5)

# --- Points décalés à droite ---
for i, modele in enumerate(modeles):
    vals = df_long[df_long['modele'] == modele]['HD100']
    x = np.random.normal(i, 0.04, size=len(vals)) # i + 0,32 pour tasser à droite 
    ax.scatter(x, vals, color=couleurs[modele], alpha=0.6, s=18, zorder=3)

# --- Annotations μ ± σ ---
for i, modele in enumerate(modeles):
    vals = df_long[df_long['modele'] == modele]['HD100']
    mu, sigma = vals.mean(), vals.std()
    ax.annotate(
        f'M={mu:.2f} ± {sigma:.2f}',
        xy=(i, 0.02), ha='center', va='bottom',
        fontsize=9, color=couleurs[modele], fontstyle='italic'
    )

ax.set_xticks([0, 1])
ax.set_xticklabels(['SAM', 'MedSAM'], fontsize=12)
ax.set_ylabel('HD100', fontsize=12)
ax.set_xlabel('')
#ax.set_ylim(-0.07, 1)
ax.yaxis.set_tick_params(labelsize=11)

ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.title('HD100 score — SAM vs MedSAM', fontsize=12, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('./figures/HD100.png', dpi=300, bbox_inches='tight')
print('HD100 sauvegardé')
#plt.show()
plt.close()

# Figure 4 : comparaison entre modèles 

fc.figure_comparaison_dice(tableau_resultat, save_dir='./figures')

# Figure 5 : AVSD 

'''