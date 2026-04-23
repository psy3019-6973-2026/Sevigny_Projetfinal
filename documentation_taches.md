Documentation détaillée sur les tâches 

# Tâche 1 

## Difficulté à trouver les données 
Sur le github de l’élève, ça pointe vers le site de concours Brats, mais vu que c’est 2021 tout est archivé et il n’y a plus les données directement 
Finit par trouver en cherchant directement sur kaggle 
kaggle datasets download dschettler8845/brats-2021-task1
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1 

Permet d’importer seulement 2 participants comme test, j’ai commencé avec ça le temps de faire de l’espace dans mon PC 
Trouver les modèles 
Recherche du checkpoint de SAM : 
Trouvé ici https://github.com/facebookresearch/segment-anything/blob/main/README.md
Dans le read me section modelcheckpoints, celui pour le vit-b based on the code

Celui de MED-SAM (sam fine tuné par l’élève), trouvé dans le lien commenté dans le Notebook, mais le nom est différent 
Du code original : 
#Load pre-trained model checkpoint and do simple pre-processing
model_type = 'vit_b'
sam_model_checkpoint = './sam_vit_b_01ec64.pth'
med_sam_model_checkpoint = './med_sam_model_best.pth' #you can find my pre-traind checkpoint file from: https://drive.google.com/drive/folders/1MbHo0qBfkQYARUhB-DAhbD5a4lhmYNqs?usp=drive_link

## Tentative de rouler le notebook initial : 

### Issue : Pas accès à compute_dice_coefficient
from utils.SurfaceDice import compute_dice_coefficient
Ne fonctionne pas (dans venv initialement, mais n’est simplement pas une fonction, devait être un fichier téléchargé par l’élève)  
Solution : créé ma propre fonction Dice score directement dans le code, basé sur la formule en ligne 
Issue : Packages installables seulement en pip 
Mon venv crashait, en essayant de les installer manuellement ça ne marchait pas non plus 
Sauf quand j’ai essayé avec pip 

Donc trouvé une manière de mettre ça dans le venv, quand je lui spécifie de les installer en pip ça fonctionne : 
  - pip:
      - segment-anything
      - scikit-image

### Issue : load du GPU innéxistant 
Le code original utilise toarch pour load un GPU si j’ai bien compris, mais moi vu que j’en avais pas ça mettait une erreur 
Avec des essais erreurs j’ai reussi à avoir un code qui fonctionne, en lui indiquant de ne pas charger de gpu mais plutôt de rouler sur ‘cpu’ 

device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=sam_model_checkpoint).to(device)
med_sam_model = sam_model_registry[model_type](checkpoint=med_sam_model_checkpoint).to(device)
sam_predictor = SamPredictor(sam_model)
med_sam_predictor = SamPredictor(med_sam_model)

Changé pour : 
sam_model = sam_model_registry[model_type](checkpoint=None)
sam_model.load_state_dict(torch.load(sam_model_checkpoint, map_location=torch.device('cpu')))
sam_model.to(device)

med_sam_model = sam_model_registry[model_type](checkpoint=None)
med_sam_model.load_state_dict(torch.load(med_sam_model_checkpoint, map_location=torch.device('cpu')))
med_sam_model.to(device)

sam_predictor = SamPredictor(sam_model)
med_sam_predictor = SamPredictor(med_sam_model)

### Issue : boucle 
Après avoir reussi à load les modèles, le code roulait pour des scans individuels (production de la segmentation pour l’image) 
La boucle qui le fait pour tous les participants utilise le format 

data_test
├── BraTS2021_00000
│   ├── BraTS2021_00000_seg.nii.gz
│   └── BraTS2021_00000_t2.nii.gz
└── BraTS2021_00002
    ├── BraTS2021_00002_seg.nii.gz
    └── BraTS2021_00002_t2.nii.gz

Mais en les téléchargeant individuellement moi j’avais fait : 

data
├── BraTS2021_00495_seg.nii.gz
├── BraTS2021_00495_t2.nii.gz
├── BraTS2021_00621_seg.nii.gz
└── BraTS2021_00621_t2.nii.gz

Donc du changer le format de la boucle temporairement 

Original : 
(data_path = "./brain_images")
scan_path_list = glob.glob(data_path+"/*")
    for scan_path in scan_path_list:
    scan_data_list = glob.glob(scan_path+"/*")

Moi : 
scan_path_list = glob.glob(data_path+"/*")
scan_path_list = [file for file in scan_path_list if file.endswith("t2.nii.gz")]
for scan_path in scan_path_list:

## Téléchargement de tous les sujets 
Le contenu du kaggle est : 
(env_initial) cassa2@DESKTOP-R035LH9:~/psy3019/projet$ kaggle datasets files dschettler8845/brats-2021-task1
name                                size  creationDate
---------------------------  -----------  --------------------------
BraTS2021_00495.tar             10321920  2021-08-19 21:59:03.089000
BraTS2021_00621.tar             10772480  2021-08-19 21:59:03.067000
BraTS2021_Training_Data.tar  13396408320  2021-08-19 22:05:19.995000

BraTS2021_00495.tar        → archive d'UN seul cas patient (4 modalités + masque)
BraTS2021_00621.tar        → archive d'UN seul cas patient (4 modalités + masque)
BraTS2021_Training_Data.tar → archive de TOUS les 1666 cas patients

Pour vérifier que le notebook roulait j’avais commencé avec seulement les deux sujets, maintenant je télécharge la vraie base de données : 
kaggle datasets download dschettler8845/brats-2021-task1 -p source_data

Par souci d’espace dans mon ordi, j’ai seulement gardé la moitié (participants 00000 – 00099) 
Les analyses finales sont donc faites sur : 
(env_projet_repro) cassa2@DESKTOP-R035LH9:~/psy3019/projet_repro$ ls source_data/
BraTS2021_00000  BraTS2021_00017  BraTS2021_00031  BraTS2021_00051  BraTS2021_00064  BraTS2021_00085
BraTS2021_00002  BraTS2021_00018  BraTS2021_00032  BraTS2021_00052  BraTS2021_00066  BraTS2021_00087
BraTS2021_00003  BraTS2021_00019  BraTS2021_00033  BraTS2021_00053  BraTS2021_00068  BraTS2021_00088
BraTS2021_00005  BraTS2021_00020  BraTS2021_00035  BraTS2021_00054  BraTS2021_00070  BraTS2021_00089
BraTS2021_00006  BraTS2021_00021  BraTS2021_00036  BraTS2021_00056  BraTS2021_00071  BraTS2021_00090
BraTS2021_00008  BraTS2021_00022  BraTS2021_00043  BraTS2021_00058  BraTS2021_00072  BraTS2021_00094
BraTS2021_00009  BraTS2021_00024  BraTS2021_00044  BraTS2021_00059  BraTS2021_00074  BraTS2021_00095
BraTS2021_00011  BraTS2021_00025  BraTS2021_00045  BraTS2021_00060  BraTS2021_00077  BraTS2021_00096
BraTS2021_00012  BraTS2021_00026  BraTS2021_00046  BraTS2021_00061  BraTS2021_00078  BraTS2021_00097
BraTS2021_00014  BraTS2021_00028  BraTS2021_00048  BraTS2021_00062  BraTS2021_00081  BraTS2021_00098
BraTS2021_00016  BraTS2021_00030  BraTS2021_00049  BraTS2021_00063  BraTS2021_00084  BraTS2021_00099

## Amélioration de la boucle pour tous les sujets 
Prend maintenant le format : 
├── source_data
│   ├── BraTS2021_00000
│   │   ├── BraTS2021_00000_seg.nii.gz
│   │   └── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00002
│   │   ├── BraTS2021_00002_seg.nii.gz
│   │   └── BraTS2021_00002_t2.nii.gz

Comme ils sont téléchargés quand la base totale de données est téléchargée 

### Autres changements : 
-	Ajouté beaucoup de fonctions qui rendent la boucle plus lisible 
        o	Mettre dans le bon format, preprocessing
-	Changements dans les fonctions qui ne marchaient pas totalement, voir les issues 
-	Fait en sorte que la slice sélectionnée aie tout le temps un tumeur 
        o	Le code initial prend une slice entre 40 et 120 sur z 
        o	Initialement fait la fonction verification_slice_tumeur qui si la slice initiale ne marche pas, essaye d’en trouver une autre pour 10 essais 
        o	Finalement fait que le code choisisse parmis toutes les couches présentant une tumeur 
-	Métriques calculés après 
        o	Décision que de calculer les métriques à même la boucle était lourd pour rien, car je dois run la boucle à chaque fois pour les tester 
        o	Donc la boucle produit un .pkl qui sauvegarde par sujet : 
            -   Image (2d seulement), slice analysée, ground truth, segmentation MED-sam, segmentation SAM 
            -	Beaucoup plus facile à charger ensuite 
        o	Ensuite, une fonction stats roule à travers les sujets et produit les métriques voulues (voir tâche 2) 
-	Sauvegarde incrémentale 
        o	Chaque sujet sauvegarde le .pkl, parfait car la fonction crash parfois à cause du RAM de mon ordi 
        o	Pour éviter le problème je lui dis de faire max 20 sujet à la fois et je reroule la boucle plusieurs fois 

### Reproductibilité des seed random 
Ajout de seeds à partir du ID du sujet comme recommadé : 
def get_slice_index(sujet, scan_gt) : 
        
        # Obtient la seed random 
        subject_number = int(sujet.split("_")[1])
        rng = np.random.default_rng(seed=subject_number)
        
        # Obtient la slice de tumeur parmis l'ensemble des slices possibles 
        tumor_slices = np.where(scan_gt.any(axis=(0, 1)))[0]
        slice_index = rng.integers(tumor_slices.min(), tumor_slices.max())

        return slice_index

## Autres changements à travers le code : 

### Issue : fonction get_slice_pair
image = nib.load(scan_data_list[4]).get_fdata() #t2 for example
gt =  nib.load(scan_data_list[1]).get_fdata()
slice_index = np.random.randint(10,140)
image,gt = get_slice_pair(slice_index)

Mais la fonction get_slice_pair ne prend pas gt ni slice_index donc m’apparait sketch : 
def get_slice_pair(layer):
    return scan_data[:, :, layer],gt_data[:, :, layer]

Donc modifié pour : 
def get_slice_pair(layer, scan_data, gt_data):
    return scan_data[:, :, layer],gt_data[:, :, layer]

### Issue : ground thruth box 
La “boite définie une fois au début, et ensuite bbox_raw réutilisé en appelant les modèles 
bbox_raw = get_bbox_from_mask(gt_2d)

sam_seg, _, _ = sam_predictor.predict(point_coords=None, box=bbox_raw, multimask_output=False)

Mais devrait être recalculée à chaque fois : get_bbox_from_mask(gt_2d)
Du faire plein de modifications 

### Issue : chemins de données pas optimaux 

Chemins vers les fichiers .nii 
scan_path = "./brain_images/BraTS2021_00002/BraTS2021_00002_t2.nii.gz" # image médicale (IRM)
gt_path = "./brain_images/BraTS2021_00002/BraTS2021_00002_seg.nii.gz"  # gt pour ground truth = segmentation correspondante

= l'organisation de la structure doit être : 
projet/
│── brain_images/
│── Notebook.ipynb

Je crois que à la place je devrais avoir un : 
"Path vers les images = """ et ensuite ca serait plus reproductible

Il y a aussi ça dans les boucles à la fin : get_statistic_socre_on_dataset("./data")

Issue fixée dans la tâche 3 ou tous est via les chemins de invoke! 

### Issue : l’image sélectionnée peut être vide 
Sélectionne une slice aléatoire 
Avantage : parfois il faut voir si le modèle segmente bel et bien rien 

 if np.sum(gt) == 0:
            print(f"Slice {slice_index} vide pour {subject_id}, skipping...")
            continue

Long fix : fait une fonction qui cherche jusqu’à en trouver une qui marche jusqu’à un nb d’essai max 

Mais : a besoin d’un gt, donc live je fais qu’il en cherche une autre jusqu’à trouver une tumeur 
Faire qu’il sélectionne parmis les slices ou il y a une tumeur
Fonction archviée verification_slide_tumeur 

J’ai fini par sélectionner parmis les slices ou il y a une tumeur 

### changements mineurs du code 
•	Le path pour le modèle ne correspond pas à celui disponible sur google collab à partir du github
med_sam_model_checkpoint = './med_sam_model_best.pth' #you can find my pre-traind checkpoint file from: https://drive.google.com/drive/folders/1MbHo0qBfkQYARUhB-DAhbD5a4lhmYNqs?usp=drive_link
Changé pour : 
med_sam_model_checkpoint = './models/sam_model_best.pth' # Not the original but matches the link

•	Mis tous les packages au début du code à importer, et pour faciliter l’écriture du venv 
•	Ajouté beaucoup de titres et de sous sections 

# Tâche 2 : mettre airoh 

Résumé du travail : 

-	Séparé le notebook en 2 parties indépendantes : 
		a. La partie tutorielle sur un sujet dans un notebook (NOTEBOOK), qui explique chaque étapes avec plusieurs visualisation 
		b. La segmentatation automatique pour tous les sujets, ainsi que les mesures qui compare chaque métrique obtenue pour les deux modèles 

-	Rendu l'ensemble des analyses séparé en scripts exécutables appelés à partir de invoke : 
		- Codes explicatifs / ajoutés pour la section a. : 
			•	run-notebook-explicatif : roule le notebook de partie a au complet 
			•	run-visu-sujet : sauvegarde la visualisation finale pour un sujet sélectionné 

		- Codes pour l'analyse automatique de la section b. : 
			•	Run-boucle 
			•	Run-stats 
			•	Run-figures 

## Rendre plus concis les boucles du notebook 

Séparé en 3 scripts pour l'analyse : 
- Boucle 
- Stats (roule à travers les segmentations pour produire les métriques)
- Visualisation 

J'avais déjà mis beaucoup de fonctions, mais séparer en 3 scripts a demandé du travail additionnel 

### Ajout de tests si il manque des info 

Parfois crashait si par exemple il n'y avait pas encore resultat.pkl produit 
Mis des return qui print un message pour la clarté, aussi invoke donne les instructions pour que ce soit dans le bon ordre 

Je voulais initiallement faire un invoke run-all mais la boucle fait vraiment surchauffer mon ordi, et ne fonctionne que si je lui met un max donc je préfère garder le contrôle pour le moment, et je crois que run all demanderait trop de RAM donc idée abandonnée 

### Compréhension de l'exécutabilité des scripts 

Il m'a fallu beaucoup de neurones pour comprendre comment appeler les fonctions avec invoke, en fait surtout pour comprendre comment accéder au Paths 
Avec les fonctions disponibles et des essais erreurs j'ai fini par comprendre et reussir en partie : 

- Je n'arrivais pas a run un notebook à la fois fix pas super propre que chaque notebook aie son folder (car run-figure, que j'utilise pour run le notebook roule le fichier au complet)
    - Possibilité future : regarder l'écriture de run-figure pour avoir une fonction run-notebook et rendre ça plus propre, mais je n'ai pas eu le temps 

- Je n'ai pas tenté de faire invoke fetch par manque de temps, car télécharger les données me prend à chaque fois +20 minutes 
Même chose pour setup abandonné pour manque de temps, je voulais prioriser que les analyses roulent 
    - La partie "invokable" commence quand l'environnement est activé et les données dans le bon format dans source_data 
    - Mis des instructions dans le readme pour maximiser quand même la reproductibilité 

### Finalisation 

- Ajouté les descriptions des tasks 
- Vérifier que re-run à partir d'une page vierge 
- Push sur github 

# Tâche 3 : Métriques et visualisation  

Résumé de la tâche : 
Implémentation d'un code qui produit des figues d'analyse pour tous les sujets, cad 
Compare les deux modèles (SAM et MEDSAM) sur 5 métriques différentes 

Étapes : 

### Métrique calculés à l'extérieur de la boucle 

Décision : je crois que au lieu de faire mes métriques directement dans la boucle, je vais produire un df avec plusieurs éléments de tous les participants 
Avantage : run une fois cette grosse boucle, après pour les métriques je pourrai explorer à partir du df qui sera stocké  
À chaque itération, le scan 3d complet est chargé, mais ensuite une seule slide n’est pas très volumineux 
-	Image 2D slice : ~512×512 float32 ≈ 1 MB
-	2 masques (SAM + MedSAM) : ~0.5 MB chacun
-	Ground truth : ~0.5 MB
-	Métriques : négligeable
Confirmé avec bcp de sujets : 1 MB par sujet 
Test de ça avec 2 sujets 

### Recherche pour trouver les métriques 
https://link.springer.com/article/10.1186/s12880-015-0068-x 
https://www.sciencedirect.com/science/article/pii/S0167814021062289 

Overlap mesures : 
-	Dice 
o	JAC : Un similaire, That means that both of the metrics measure the same aspects and provide the same system ranking. Therefore, it does not provide additional information to select both of them together as validation metrics as done in [15–17].
-	True Positive Rate (TPR), also called Sensitivity and Recall
Spatial distance based metrics :
-	Hausdorff Distance (HD)
-	The Average Distance, or the Average Hausdorff Distance (AVD), is the HD averaged over all points. The AVD is known to be stable and less sensitive to outliers than the HD.
Fait un script qui produit toutes les figures qui comparent les métriques 
Fait un script qui prend un sujet et sort le scan initial, le gt et les deux segmentations pour aller voir les métriques 
-	Chaque figure est 68K, je pourrais quasiment le faire automatiquement pour tous les sujets 

### Produire les graphiques 

- Amélioration du graphique de Dice score 
    - Ajouté les points de mesure individuels, la barre indiquand un résultat significatif, un titre et des titres d'axes, une ligne pour la médiane 
    - Amélioré l'esthétique 
- Ajouté la mesure de 4 autres métriques, 2 de chevauchement (Precision, Recall) et deux de périmètres (HD100, distance moyenne)
    - Explications des métriques et justification des choix dans les Notebooks de Figure et celui explicatif 
- Graphiques comparant les 4 métriques, conjointement pour le type (chevauchement et périmètre)
- Graphique permettant une meilleure compréhension du Dice score (permet de voir l'évolution pour chaque sujet)

### Rendre en notebook 

Pour qu'il soit exécutable par Airoh, rendu en 1 notebook, permet de voir les figures 
Et il les sauvegarde aussi dans output_data/Figures 

Mis des explications sur les métriques dans le notebook pour améliorer la compréhension et l'analyse 

# Retour sur les tâches originales : 

Ce qui a été présenté à la présentation de mi-étape : 

## Tâche 1 : Reproduction du Notebook
Le projet comporte 2 notebook, l'entraînement du modèle et son application (lui feed des données tests, analyser les sorties)

Je veux reproduire le nootebook d'application des deux modèles entraînés

X Configurer l'environnement (tout download les packages)
X Reproduire l'organisation des données / fichiers nécéssaires (selon les chemins spécifiés)
X Download les modèles entrainés (possiblement devoir ajuster car le modèle est disponible sur google drive)
X Documenter les étapes de reproductions (bogues rencontrés, solutions)
X Ajouter des instructions de reproduction claires dans le README. (basé sur mes commentaires / notes de reproduction)

Bonification possible : 
X Résumer les étapes de download des packages en un
X Environment.ylm
~ Requirements.txt

Ajouts : 
- Refactorisation de plusieurs fonctions

# Tâche 2 : Améliorer la comparaison entre les modèles

FAIT mais écrit comme tâche 3 pour l'ordre soit plus logique et lisible 

Bonifier la figure de visualisation
Documenter les étapes, choix méthodologiques et artistiques
Ajouter une métrique de comparaison (en ce moment les modèles sont uniquement comparés selon leur performance au DICE score) ex : Kappa, Hausdorff Distance (HD = distance max entre les contours), Average Symmetric Surface Distance (ASSD = Distance moyenne entre les surfaces des deux segmentations)
Bonification possible : Ajouter des graphiques diagnostiques spécifiquement reliés aux modèles

# Tâche 3 : Notebook pédagogique
L’objectif de cette tâche est de produire un notebook destiné à un public débutant qui explique de manière progressive le processus de fine-tuning du modèle SAM pour la segmentation de tumeurs cérébrales. C'est la partie du travail que je ne touche pas à date (entraînement du modèle, notebook 1) et je suis interessée à comprendre théoriquement comment le modèle fonctionne et faire un travail de vulgarisation pour comprendre ce qui se passe derrière les commandes clés du notebook d'entraînement.

Présenter SAM et MedSAM : objectifs et différences
Expliquer le principe de modèle de fondation et de fine-tuning dans le contexte médical
Éléments mathématiques principaux

Tâche remplacée par une : 

Tâche supplémentaire potentielle
Si une des tâche ne s'avère pas possible / trop simple, je prévois refactoriser le notebook en pipeline réutilisable (fonctions).

Car j'avais l'impression d'avoir encore beaucoup de travail potentiel pour rendre les analyses reproductibles après avoir completé ma tâche 1 et déjà trop travaillé dessus pour l'intégrer. J'ai beaucoup aimé le cours de scripts donc c'est quelque chose qui m'interessait de mieux comprendre et implémenter des scripts pour les analyses, donc j'avais un intêret pour continuer la tâche dans la même direction 

Aussi, une partie de la tâche 2 s'est transformée en Notebook explicatif par rapport aux métriques sélectionnées, donc j'avais moins envie de faire de la recherche mais plutôt coder. 

Ce qui a été fait : 
## Tâche changée : Reproductiblité en scripts appelables à partir de airoh 

Une fois l'environnement installé et les données downloadés selon les instructions du readme, mon analyse est totalement reproductible à partir de commandes invoke 

- Séparaison en 2 : notebook explicatif et analyse pour tous les sujets 
- Écriture de tasks.py 
