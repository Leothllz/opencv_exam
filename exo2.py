import cv2
import numpy as np

# Chargement de l'image
image = cv2.imread('@propImageMat4_r4.jpg')

# Sélection de la région d'intérêt (ROI) par l'utilisateur
# La fonction selectROI ouvre une fenêtre permettant à l'utilisateur de sélectionner une zone de l'image
r = cv2.selectROI("Sélectionnez la ROI", image, False)
x, y, w, h = r  # Coordonnées et dimensions de la ROI sélectionnée

# Création d'un masque noir de la même taille que l'image
# Ce masque sera utilisé pour appliquer les modifications uniquement à la ROI
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Remplissage de la ROI dans le masque avec du blanc
# Cela permet d'identifier la zone à modifier
mask[y:y+h, x:x+w] = 255

# Modification de la luminosité dans la ROI
roi = image[y:y+h, x:x+w].copy()
roi = cv2.add(roi, np.array([50.0]))  # Augmentation de la luminosité de 50 unités

# Application du masque pour modifier uniquement la ROI
image_modifiee = image.copy()
image_modifiee[y:y+h, x:x+w] = roi

# Application du flou gaussien en dehors de la ROI
# Cela crée un effet de mise au point sur la ROI
image_floue = cv2.GaussianBlur(image, (15, 15), 0)
image_finale = np.where(mask[:,:,np.newaxis] == 255, image_modifiee, image_floue)

# Affichage et sauvegarde de l'image finale
cv2.imshow("Image finale", image_finale)
cv2.imwrite("image_roi_modifiee.jpg", image_finale)

cv2.waitKey(0)
cv2.destroyAllWindows()