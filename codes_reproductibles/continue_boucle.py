import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from torch.utils.data import Dataset, DataLoader
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
from ipywidgets import interact
import seaborn as sns
import pickle
import pandas as pd # Ajouté pour les stats
import fonctions as fc

results = fc.continue_statistic_score_on_dataset("../data_test3", existing_results_path="./resultats/results_test.pkl")
