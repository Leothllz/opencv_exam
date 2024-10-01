import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chargement de l'image
image = cv2.imread('@propImageMat4_r4.jpg')

# Conversion de l'image en niveaux de gris
# Cela simplifie le traitement en réduisant l'image à une seule dimension de couleur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Application du filtre Sobel pour la détection de contours
# Sobel calcule les gradients dans les directions x et y
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient horizontal
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient vertical
# Combinaison des gradients pour obtenir la magnitude totale
sobel = np.sqrt(sobelx**2 + sobely**2)

# Normalisation de l'image Sobel pour l'affichage
# Conversion des valeurs en entiers 8 bits (0-255)
sobel = np.uint8(255 * sobel / np.max(sobel))

# Transformation de Fourier
# Permet d'analyser l'image dans le domaine fréquentiel
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)  # Déplace les basses fréquences au centre
magnitude_spectrum = 20 * np.log(np.abs(fshift))  # Calcul du spectre de magnitude

# Manipulation du spectre (suppression des basses fréquences)
# Cela peut être utilisé pour supprimer le bruit ou mettre en évidence certaines caractéristiques
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0  # Supprime une région 60x60 au centre

# Transformation inverse de Fourier
# Retour dans le domaine spatial après modification du spectre
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Segmentation par seuillage adaptatif
# Utile pour séparer les objets du fond dans des conditions d'éclairage variables
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Affichage des résultats
cv2.imshow('Original', image)
cv2.imshow('Sobel', sobel)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()
cv2.imshow('Inverse Fourier', img_back.astype(np.uint8))
cv2.imshow('Segmentation', thresh)

# Sauvegarde des images traitées
cv2.imwrite('sobel.jpg', sobel)
cv2.imwrite('inverse_fourier.jpg', img_back.astype(np.uint8))
cv2.imwrite('segmentation.jpg', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()