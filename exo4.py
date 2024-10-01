import cv2

def select_roi(frame):
    # Fonction pour permettre à l'utilisateur de sélectionner manuellement l'objet à suivre
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")
    return bbox

def main():
    # Initialisation de la capture vidéo (utilise la webcam par défaut)
    cap = cv2.VideoCapture(0)

    # Lecture de la première frame
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo")
        return

    # Sélection de l'objet à suivre par l'utilisateur
    bbox = select_roi(frame)

    # Initialisation du tracker CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
    # CSRT offre une bonne précision mais est plus lent que d'autres trackers
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    while True:
        # Lecture d'une nouvelle frame
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo")
            break

        # Mise à jour du tracker avec la nouvelle frame
        success, bbox = tracker.update(frame)

        if success:
            # Si le suivi est réussi, dessiner le rectangle autour de l'objet suivi
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Si l'objet est perdu, afficher un message
            cv2.putText(frame, "Objet perdu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Affichage de la frame
        cv2.imshow