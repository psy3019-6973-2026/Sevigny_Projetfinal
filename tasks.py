import os
from pathlib import Path
from invoke import task

# Du faire ça pour que ca marche (env_projet_repro) cassa2@DESKTOP-R035LH9:~/psy3019/projet_repro$ touch code/__init__.py

@task(
    help={
        "name": "Nom logique du fichier, tel que défini dans la section 'files' de invoke.yaml."
    }
)

@task
def run_boucle(c):
    """
    Run la boucle sur les sujets dans source_data 
    """
    input_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir"))
    models_dir = Path(c.config.get("models_dir"))

    from code.boucle import continue_statistic_score_on_dataset
    continue_statistic_score_on_dataset(input_dir, output_dir, models_dir)

# Enlevé run boucle avant car sinon la roule à chaque fois idk
@task
def run_stats(c):
    """
    Run toutes les métriques de comparaison sur le tableau résultats 
    """
    resultats_dir = Path(c.config.get("output_data_dir"))

    from code.stats import run_stats
    run_stats(resultats_dir) 

@task
def run_notebook_explicatif(c, sujet="BraTS2021_00002"):
    from airoh.utils import run_figures
    print("sujet reçu :", repr(sujet))  # <-- ajoute ça
    c.config["sujet"] = sujet
    notebooks_dir = Path(c.config.get("notebooks_dir")) / "explicatif"
    run_figures(c, notebooks_dir, keys=["source_data_dir", "output_data_dir", "models_dir", "sujet"])


@task
def save_visu_sujet(c, sujet):
    from code.save_visu_sujet import visu
    import pandas as pd

    output_dir = Path(c.config.get("output_data_dir"))
    output_figure = Path(c.config.get("figures_dir"))

    # Validation avant d'appeler visu
    tableau = output_dir / "resultats.csv"
    if not tableau.exists():
        raise FileNotFoundError(f"Fichier de résultats introuvable : {tableau}")
    
    tableau_resultat = pd.read_csv(tableau, index_col=0)
    if sujet not in tableau_resultat.index:
        raise ValueError(f"Sujet '{sujet}' introuvable dans le tableau de résultats.")

    visu(sujet, output_dir, output_figure)

