import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local(img, tamaño_ventana) -> np.ndarray:
    """
    Realiza la ecualización local de una imagen en escala de grises utilizando un tamaño de ventana específico.
    Args:
        img (np.ndarray): Imagen de entrada en escala de grises.
        tamaño_ventana (tuple): Tamaño de la ventana para la ecualización local (m, n).
    Returns:
        np.ndarray: Imagen ecualizada.
    """

    m,n = tamaño_ventana
    bordes = max(m,n)//2

    filas, columnas = img.shape

    img_copia = img.copy()

    img_bordes = cv2.copyMakeBorder(img_copia, bordes, bordes, bordes, bordes, cv2.BORDER_REPLICATE)

    for fila in range(filas):
        for columna in range(columnas):
            ventana = img_bordes[fila:fila+m, columna:columna+n]
            img_histogrma = cv2.equalizeHist(ventana)
            img_copia[fila, columna] = img_histogrma[m//2, n//2]
    
    return img_copia

tamaño = (33,33)

img = cv2.imread('TP 1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img_ecualizada = ecualizacion_local(img, tamaño)
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Imagen Original')
plt.axis('off')
plt.subplot(122), plt.imshow(img_ecualizada, cmap='gray'), plt.title(f'Imagen Ecualizada de tamaño {tamaño}')
plt.axis('off')
plt.show()