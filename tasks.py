import os
from pathlib import Path
from invoke import task

# Du faire ça pour que ca marche (env_projet_repro) cassa2@DESKTOP-R035LH9:~/psy3019/projet_repro$ touch code/__init__.py

@task(
    help={
        "name": "Nom logique du fichier, tel que défini dans la section 'files' de invoke.yaml."
    }
)
def import_file(c, name):
    """🌐 Download a single file from a URL using urllib."""
    from urllib.request import Request, urlopen
    
    files = c.config.get("files", {})
    if name not in files:
        raise ValueError(f"❌ No file config found for '{name}' in invoke.yaml.")

    entry = files[name]
    url = entry.get("url")
    output_file = entry.get("output_file")

    if not url or not output_file:
        raise ValueError(
            f"❌ Entry for '{name}' must define both 'url' and 'output_file'."
        )

    output_path = Path(output_file)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"🫧 Skipping {name}: {output_file} already exists.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)

    print(f"📥 Downloading '{name}' from {url}")
    print(f"📁 Target: {output_file}")

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )

    try:
        with urlopen(req, timeout=60) as response, tmp_path.open("wb") as f:
            total = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)

        if total == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"❌ Downloaded 0 bytes for '{name}'.")

        tmp_path.replace(output_path)

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"❌ Failed to download '{name}' from {url}: {e}") from e

    print(f"✅ Downloaded {name} to {output_file} ({output_path.stat().st_size} bytes)")

@task
def fetch(c):
    """
    Retrieve all data assets.
    """
    import_file(c, "papers")

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

@task
def run_stats(c):
    """
    Run toutes les métriques de comparaison sur le tableau résultats 
    """
    resultats_dir = Path(c.config.get("output_data_dir"))

    from code.stats import run_stats
    run_stats(resultats_dir) 

'''

À changer : 

@task
def run_simulation(c):
    """
    Run a small simulation.
    """
    output_dir = Path(c.config.get("output_data_dir"))
    from code.simulation import simulation
    simulation(output_dir)

@task(pre=[run_simulation])
def run_figure(c):
    """
    Generate figures from the simulation output using a notebook.
    """
    from airoh.utils import run_figures, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    source_dir = Path(c.config.get("source_data_dir")).resolve()

    ensure_dir_exist(c, "output_data_dir")
    run_figures(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])

@task(pre=[run_simulation, run_figure])
def run(c):
    print("all analyses completed")
  

@task
def clean(c):
    """
    Clean the output folder.
    """
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.png")
    clean_folder(c, "output_data_dir", "*.csv")
'''