import os
import matplotlib.pyplot as plt
import pickle
import fonctions as fc

with open("results.pkl", "rb") as f:
    results_load = pickle.load(f)

sujet = "BraTS2021_00002"
save_dir = "./figures"

fc.save_figure_gt(results_load, sujet, save_dir)