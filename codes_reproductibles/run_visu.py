import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import fonctions as fc

# Sélectionner le sujet à visualiser 
sujet_liste = ['BraTS2021_00009'] 
#sujet_liste = ["BraTS2021_00068", 'BraTS2021_00058', 'BraTS2021_00064', 'BraTS2021_00051', 'BraTS2021_00025']

save_dir = "./visualisations"

# Charger ces images 
with open("resultats/results_test.pkl", "rb") as f:
    results_load = pickle.load(f)

# Charger ces résultats 
tableau_resultat = pd.read_csv('./resultats/resultats_tableau.csv', index_col=0)

# Visualisations 
for sujet in sujet_liste : 
    #fc.save_figure_resultats(results_load, tableau_resultat, sujet, save_dir="visualisations/")
    #fc.save_figure_gt(results_load, sujet, save_dir)
    fc.save_figure_complete(results_load, tableau_resultat, sujet, save_dir="visualisations/")

