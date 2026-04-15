import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import fonctions as fc

# Sélectionner le sujet à visualiser 
sujet = "BraTS2021_00003"
save_dir = "./visualisations"

# Charger ces images 
with open("results.pkl", "rb") as f:
    results_load = pickle.load(f)

# Charger ces résultats 
tableau_resultat = pd.read_csv('./resultats/resultats_tableau.csv', index_col=0)

# Visualisations 
fc.save_figure_resultats(results_load, tableau_resultat, sujet, save_dir="visualisations/")
fc.save_figure_gt(results_load, sujet, save_dir)

