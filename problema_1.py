import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')
plt.show()

# Ecualizacion global
img_ecualizada_global = cv2.equalizeHist(img)
plt.imshow(img_ecualizada_global, cmap='gray')
plt.title('Imagen con ecualización global')
plt.show()

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
    top, bottom, left, right = bordes, bordes, bordes, bordes

    filas, columnas = img.shape

    img_copia = img.copy()

    img_bordes = cv2.copyMakeBorder(img_copia, top,bottom,left,right, cv2.BORDER_REPLICATE)

    for fila in range(filas):
        for columna in range(columnas):
            ventana = img_bordes[fila:fila+m, columna:columna+n]
            img_histograma = cv2.equalizeHist(ventana)
            img_copia[fila, columna] = img_histograma[m//2, n//2]
    
    return img_copia


tamaño1 = (3,3)
tamaño2 = (20, 20)
tamaño3 = (33,33)
tamaño4 = (50,50)
tamaño5 = (15, 45)
tamaño6 = (45, 15)

# Ecualizacion local con tamaño (3, 3)
img_ecualizada_local_t3 = ecualizacion_local(img, tamaño1)
# Ecualizacion local con tamaño (20, 20)
img_ecualizada_local_t15 =  ecualizacion_local(img, tamaño2)
# Ecualizacion local con tamaño (33, 33)
img_ecualizada_local_t33 =  ecualizacion_local(img, tamaño3)
# Ecualizacion local con tamaño (50, 50)
img_ecualizada_local_t50 =  ecualizacion_local(img, tamaño4)
# Ecualizacion local con tamaño (15, 45)
img_ecualizada_local_r15x45 = ecualizacion_local(img, (15, 45))
# Ecualizacion local con tamaño (45, 15)
img_ecualizada_local_r45x15 = ecualizacion_local(img, (45, 15))

# Visualizamos
plt.figure(figsize=(16, 8))

# Imagen original
plt.subplot(241)
plt.imshow(img, cmap='gray')
plt.title('Original')

# Ecualización global
plt.subplot(242)
plt.imshow(img_ecualizada_global, cmap='gray')
plt.title('Ecualización Global')

# Ecualizaciones locales
plt.subplot(243)
plt.imshow(img_ecualizada_local_t3, cmap='gray')
plt.title('Ecualización Local (3x3)')

plt.subplot(244)
plt.imshow(img_ecualizada_local_t15, cmap='gray')
plt.title('Ecualización Local (15x15)')

plt.subplot(245)
plt.imshow(img_ecualizada_local_t33, cmap='gray')
plt.title('Ecualización Local (33x33)')

plt.subplot(246)
plt.imshow(img_ecualizada_local_t50, cmap='gray')
plt.title('Ecualización Local (50x50)')

plt.subplot(247)
plt.imshow(img_ecualizada_local_r15x45, cmap='gray')
plt.title('Ecualización Local (15x45)')

plt.subplot(248)
plt.imshow(img_ecualizada_local_r45x15, cmap='gray')
plt.title('Ecualización Local (45x15)')

plt.tight_layout()
plt.show()

# Visualizamos la imágen con mejor resultado:
plt.imshow(img_ecualizada_local_t15, cmap='gray')
plt.title('Imagen con ecualización local (15x15)')
plt.show()
