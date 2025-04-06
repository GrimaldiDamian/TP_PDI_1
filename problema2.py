import cv2
import numpy as np
import matplotlib.pyplot as plt

def deteccion_renglones(img):
    """
    Detecta los renglones en una imagen.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        numpy.ndarray: Imagen con los renglones detectados.
    """
    img_copy = img.copy()
    img_copy[img_copy <= 130] = 0
    img_copy[img_copy > 130] = 255

    img_zeros = img_copy == 0

    img_row_zeros = img_zeros.any(axis=1)

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x).flatten()

    fila_inicio_encabezado = renglones_indxs[2] + 1
    fila_fin_encabezado = renglones_indxs[3] + 1
    encabezado = img_copy[fila_inicio_encabezado:fila_fin_encabezado, :]

    preguntas = []
    for i in range(4, len(renglones_indxs)-1, 2):  # Avanzamos de 2 en 2 para pares (inicio, fin)
        y1 = renglones_indxs[i] + 1
        y2 = renglones_indxs[i + 1] + 1

        if y2 - y1 > 10:  # Aseguramos que sea un bloque con contenido
            pregunta = img_copy[y1:y2, :]
            preguntas.append(pregunta)

    return encabezado, preguntas

img = cv2.imread('TP 1/multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)

respuestas_correctas = [
    "A", "A", "B", "A", "D", "B", "B", "C", "B", "A",
    "D", "A", "C", "C", "D", "B", "A", "C", "C", "D",
    "B", "A", "C", "C", "C"
]

encabezado, preguntas = deteccion_renglones(img)