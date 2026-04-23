Bonjour! Folder pour reproduire les analyses faites dans le cadre de mon projet de reproduction à 
partir du projet : 

Brain Tumor Segmentation via SAM-based fine-tuning on structural MRI images

# Informations générales projet 

Auteur : Alex Peng

Repo github : https://github.com/AlexPeng517/BHS2023_Project_SAM_MRI/tree/main?tab=readme-ov-file

URL brainhack : https://school-brainhack.github.io/project/

brain_tumor_segmentation_via_sam-based_fine-tuning_on_structural_mri_images/

## Description du projet 
Le projet choisi a pour but de faire la segmentation de tumeur cérébrale sur des sur des images d’IRM structurelle du cerveau. L’objectif principal est de comparer les performances de deux modèles de segmentation : le modèle MedSAM et une version du modèle fine-tunée par un étudiant.

Parmmis les plus récents développements dans le domaine du "computer vision" se trouve un modèle développé par Méta Segments Anything Model (SAM) qui a pour but de segmenter une grande variété d’objets dans des images naturelles. Par la suite, des chercheurs de l’Université de Toronto ont adapté ce modèle au domaine médical en le spécialisant sur des images biomédicales, ce qui a donné MedSAM, qui montre de meilleures performances pour ce type de données que le SAM original. Le projet reprend cette idée, en fine-tunant le modèle SAM sur des images d’IRM de tumeurs cérébrales, afin d’améliorer ses performances pour cette tâche.

Les deux modèles (SAM original et MED-SAM (fined tuné)) sont ensuite comparés avec une métrique de Dice Score, et une visualisation de la segmentation par les deux modèles sont offertes. 

### Base de données 

Le projet utilise deux bases de données ouvertes qui contiennent des images IRM anatomiques ainsi que la segmentation ("ground-truth"), c'est à dire ce qui est utilisé comme référence de la délimitation de la tumeur. 

La première base, utilisée pour le fine-tuning du modèle, provient du jeu de données Brain Tumor Segmentation disponible sur Kaggle (source originale : Jun Cheng). Elle contient 3064 images IRM T1 avec injection de contraste, issues de 233 patients et réparties en trois types de tumeurs : méningiomes (708 coupes), gliomes (1426 coupes) et adénomes hypophysaires (930 coupes).

La deuxième, utilisée pour comparer les performances du modèle original et du modèle fine-tuné (évaluation du modèle), est la base BRaTS 2021 (RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge). Elle comprend 1666 examens avec des séquences T1, T2 et T2-FLAIR au format NIfTI (.nii.gz), accompagnées de leurs segmentations de référence. C'est la deuxième qui est pertinante pour moi, car je compte reproduire le notebook d'évaluation et ajouter des visualisations qui seront produites à partir de ces images. 

### Résultats 
La comparaison entre les modèles est faite à partir des Dice score. 
Les résultats montrent que le modèle fine-tune performe mieux.

## Choix du projet 
J'ai choisi ce projet car une des techniques d'analyse de données en neuroscience cognitive avec laquelle je suis le moins à l'aise est l'utilisation de modèles d'intelligence artificielle. Ce projet est une bonne façon pour moi de me familiariser avec ceux-ci, sans avoir à entraîner un modèle dans son ensemble.

Pour plus de détails concernant le projet, voir Achive/Présentation_initiale.md


# Résumé du contenu des codes : 

a. Notebook explicatif qui prend un sujet et passe à travers toutes les étapes et produit les visualisations 

b. Fonctions à rouler pour faire l'analyse sur tous les sujets : 

	invoke run-boucle 

	invoke run-stats

	invoke run-figures

# Reproductibilité

Pour reproduire les mêmes analyses et figures que moi, j'ai roulé sur un sous-ensemble de participants par souci d'espace de stockage et de RAM, c'est à dire : 
Participants 00000 à 00099 

# Étapes : 

## 1. Initialiser l'environnement virtuel : 
conda env create -f environment.yml 
conda activate env_projet_final 

## 2. Télécharger les données 
Les données proviennent du dataset BraTS 2021 Task 1 sur Kaggle. Pour les télécharger, effectuer ces étapes : 

a. Prérequis : configurer l'API Kaggle

- Créer un compte sur kaggle.com

- Suivre les instructions pour créer un token : https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#api-credentials

b. Télécharger les données :

    kaggle datasets download dschettler8845/brats-2021-task1 -p source_data

c. Dézipper les données :

    tar -xf source_data/BraTS2021_Training_Data.tar -C source_data

Les données doivent être dans ce format : 

├── source_data

│   ├── BraTS2021_00000

│   │   ├── BraTS2021_00000_seg.nii.gz

│   │   └── BraTS2021_00000_t2.nii.gz

│   ├── BraTS2021_00002

│   │   ├── BraTS2021_00002_seg.nii.gz

│   │   └── BraTS2021_00002_t2.nii.gz
...

d. Vérifier que les modèles sont présents 
Les modèles checkpoints des modèles sont inclus dans modèles/ 
Les modèles ont été trouvés sur : 
	- MED-SAM : https://drive.google.com/drive/folders/1MbHo0qBfkQYARUhB-DAhbD5a4lhmYNqs 
	- SAM : https://github.com/facebookresearch/segment-anything/blob/main/README.md 

## Exécuter le code et les analyses avec invoke! 

### **Étape 1**: Boucle pour segmenter chaque sujet 
Passe à travers tous les sujets de source data, et segmente la tumeur d'une slice aléatoire avec les deux modèles analysés 
```
bash
run-boucle
```

### **Étape 3**: Boucle pour extraire les métriques 
Pour chaque segmentation, quantifie la qualité avec plusieurs métriques (expliqués dans le notebook explicatif)
```
bash
invoke stats
```

### **Étape 4** : Notebook avec toutes les visualisations 
```
bash
invoke run-figures
```

## Éléments optionnels : 

### Visualisation pour un sujet
Si un sujet veut être plus analysé, produit la visualisation du scan original, la tumeur et les segmentations 
```
bash
invoke save-visu-sujet --sujet BraTS2021_XXXXX
```

### Notebook explicatif 
Passe à travers toutes les étapes pour le sujet BraTS2021_00002 par défault, avec des explications 
```
bash
invoke run-notebook-explicatif 
```

# Tâches 

Pour voir les tâches initiales, présentés à la présentation de mis session, voir Presentation_initiale.md 

Pour une documentation plus précise sur la timeline et les issues rencontrés de chaque tâche, voir documentation_taches.md 
Pour mes notes (très personnelles et brouillones, plus comme un journal de bord que je tenais) à travers le temps, voir notes.txt 

## 1. Tâche 1 : Reproduction du Notebook 

But initial : Je veux reproduire le nootebook d'application des deux modèles entraînés
J'ai combiné avec la partie éducative de la tâche initiale 3 

Ce qui a été fait : 
-	Reproduction du Notebook initial 
		- Documentation de issues 
			- Trouver les données, non disponible sur le lien avec le github. Et trouver comment mettre dans le bon format 
			- Difficultés avec certains packages et une fonction écrite par l’élève manquante 
			- Ajustements car le code était fait pour avoir un GPU 
			- Problématiques dans certaines fonctions (utilisation de variables globales innapropriés)
			- Changement pour une adaptation si la slice sélectionnée n’a pas de tumeur afin de maximiser tous les participants 
			- Changements mineurs dans le code, comme ajout de titres, commentaires explicatifs et organisation  
-   Refactorisation de fonctions 
-	Environnement.yml 
		- Certains packages ont du être installés en pip 

J'ai travaillé sur le Notebook_initial_reproduit.ipynb 
Il était initialement dans un repo forked du projet original
Je l'ai merge (pour l'historique) à la fin, voir Note repo git 

## 2. Tâche 2 : Script et reproductibilité 

Tâche ajoutée après le cours sur les scripts 
La dernière tâche était initialement sur un Notebook explicatif du modèle, mais les analyses étaient 
toutes regroupés dans un seul notebook, avec des chemins de fichiers hardcodés après ma première tâche et je considérais qu'il restait du travail considérable pour rendre les analyses reproductibles. 
J'ai voulu d'abord séparer les grandes boucles d'analyses initiales en petites fonctions, puis scripts ce qui a grandement aidé la lisibilité, puis j'ai voulu accomplir le défi de mettre en format airoh. 

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

Ces commandes produisent un analyse statistique complète sur des couches aléatoires de segmentation pour chaque sujet

Une fois l'environnement installé et les données downloadés selon les instructions du readme, mon analyse est totalement reproductible à partir de commandes invoke 

## 3. Tâche 3 : Visualisation et métriques 

Tâche originale (tâche 2 initiale) : 
•	Bonifier la figure de visualisation
		- Documenter les étapes, choix méthodologiques et artistiques
•	Ajouter une métrique de comparaison 

Ce qui a été fait : 
	- Recherche sur des métriques à utiliser et sélection 
	- Amélioration du graphique de Dice score 
		- Ajouté les points de mesure individuels, la barre indiquand un résultat significatif, un titre et des titres d'axes, une ligne pour la médiane 
		- Amélioré l'esthétique 
	- Ajouté la mesure de 4 autres métriques, 2 de chevauchement (Precision, Recall) et deux de périmètres (HD100, distance moyenne)
		- Explications des métriques et justification des choix dans les Notebooks de Figure et celui explicatif 
	- Graphiques comparant les 4 métriques, conjointement pour le type (chevauchement et périmètre)
	- Graphique permettant une meilleure compréhension du Dice score (permet de voir l'évolution pour chaque sujet)

Toutes les figures finales sont dans output_data/Figures 

# Utilisation d'IA 
L'intelligence artificielle a été utilisée dans ce projet, surtout pour l'aide à la compréhension de certains concepts, la paufination de certains éléments de code et l'interprétation de messages d'erreurs. 

# Note repo git 

Ce git repo a été commencé lors de la tâche 3, car j'avais besoin d'un nouveau folder ou tester les composantes à partir de invoke, et dans l'original j'étais rendu avec trop de codes / scripts et fonctions non néttoyés 

Ce repo comprend quand même plusieurs commits lié à la tâche 3 (visualisation) qui a été faite en parrallèle 
À la fin, j'ai merge les historiques : 
git remote add ancien-repo /home/cassa2/psy3019/projet
git fetch ancien-repo
git merge ancien-repo/main --allow-unrelated-histories -m "Fusion historique ancien projet"

Les éléments importants du premier repo sont dans Archive : 
- Le notebook initial avec ma documentation de reproduction (initiale)
- La présentation d'étape avec les tâches originales, le choix et la description du projet 

L'historique documente aussi l'évolution des premiers scripts exécutables dans le dossier code reproductible : git log --all -- codes_reproductibles/

