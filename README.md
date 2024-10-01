Exercices OpenCV

Ce dépôt contient une série d'exercices utilisant OpenCV pour le traitement d'images et la vision par ordinateur.

Table des matières

1. Prérequis
2. Installation
3. Exercices
4. Utilisation
5. Contribution

Prérequis

- Python 3.7+
- pip (gestionnaire de paquets Python)

Installation

1. Clonez ce dépôt :
   git clone https://github.com/votre-nom-utilisateur/exercices-opencv.git
   cd exercices-opencv

2. Installez les dépendances :
   pip install -r requirements.txt

3. Téléchargez les modèles pré-entraînés nécessaires (pour l'exercice 3) :
   - deploy.prototxt
   - res10_300x300_ssd_iter_140000.caffemodel
   - age_net.caffemodel
   - age_deploy.prototxt
   - gender_net.caffemodel
   - gender_deploy.prototxt

   Placez ces fichiers dans le répertoire racine du projet.

Exercices

1. exo1.py : Traitement d'image basique
   - Conversion en niveaux de gris
   - Application du filtre Sobel
   - Transformation de Fourier
   - Segmentation par seuillage adaptatif

2. exo2.py : Manipulation de Région d'Intérêt (ROI)
   - Sélection manuelle de ROI
   - Modification de la luminosité dans la ROI
   - Application de flou gaussien en dehors de la ROI

3. exo3.py : Détection de visages et estimation d'âge/genre
   - Utilisation de modèles pré-entraînés pour la détection de visages
   - Estimation de l'âge et du genre

4. exo4.py : Suivi d'objets en temps réel
   - Sélection manuelle de l'objet à suivre
   - Utilisation du tracker CSRT d'OpenCV

5. bonus.py : Reconnaissance de gestes de la main
   - Utilisation de MediaPipe pour la détection des mains
   - Reconnaissance de gestes simples (paume ouverte/poing fermé)

Utilisation

Pour exécuter un exercice, utilisez la commande suivante :

python exoX.py

Remplacez X par le numéro de l'exercice que vous souhaitez exécuter.

Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

Imports nécessaires

Voici la liste complète des imports utilisés dans les exercices :

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

Assurez-vous d'avoir installé toutes ces dépendances avant d'exécuter les exercices.
