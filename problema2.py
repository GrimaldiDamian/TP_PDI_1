import cv2
import numpy as np
import matplotlib.pyplot as plt

def deteccion_renglones(img):
    """
    Detecta los renglones en una imagen utilizando la transformada de Hough.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        numpy.ndarray: Imagen con los renglones detectados.
    """
    img_copy = img.copy()
    img_copy[img_copy <= 110] = 0
    img_copy[img_copy > 110] = 255

    img_zeros = img_copy == 0

    img_row_zeros = img_zeros.any(axis=1)

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x)
    renglones_indxs = renglones_indxs[[0, 3]]
    len(renglones_indxs)

    fila_inicio = renglones_indxs[0, 0] + 1
    fila_fin = renglones_indxs[1, 0] + 1

    renglon_img = img_copy[fila_inicio:fila_fin, :]

    return renglon_img

img = cv2.imread('TP 1/examen_1.png', cv2.IMREAD_GRAYSCALE)

renglon = deteccion_renglones(img)