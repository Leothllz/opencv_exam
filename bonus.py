# Import de OpenCV pour le traitement d'images et la capture vidéo
import cv2

# Import de MediaPipe pour la détection et le suivi des mains
# - mediapipe fournit des modèles pré-entraînés pour la détection des mains
import mediapipe as mp

# Import de NumPy pour les calculs mathématiques efficaces
import numpy as np



# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
# Création d'une instance de Hands avec des paramètres spécifiques
# - static_image_mode=False : optimisé pour le traitement vidéo
# - max_num_hands=1 : détecte une seule main
# - min_detection_confidence=0.5 : seuil de confiance pour la détection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# Utilitaire de dessin pour visualiser les points de repère des mains
mp_drawing = mp.solutions.drawing_utils

# Fonction pour calculer la distance euclidienne entre deux points
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Fonction pour reconnaître le geste basé sur la position des doigts
def recognize_gesture(landmarks):
    # Récupération des positions du bout du pouce et de l'index
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calcul de la distance entre le pouce et l'index
    distance = calculate_distance(thumb_tip, index_tip)
    
    # Classification du geste basée sur la distance
    # Note: Le seuil de 0.1 est arbitraire et peut nécessiter des ajustements
    if distance > 0.1:
        return "Paume ouverte"
    else:
        return "Poing fermé"

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)  # 0 indique la webcam par défaut

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Impossible de lire la vidéo")
        break

    # Conversion de l'image BGR (OpenCV) en RGB (MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Traitement de l'image pour détecter les mains
    results = hands.process(image_rgb)

    # Si des mains sont détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessin des points de repère de la main
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Reconnaissance du geste
            gesture = recognize_gesture(hand_landmarks.landmark)
            
            # Affichage du geste détecté sur l'image
            cv2.putText(image, f"Geste: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de l'image traitée
    cv2.imshow('Hand Tracking', image)

    # Sortie de la boucle si 'q' est pressé
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()