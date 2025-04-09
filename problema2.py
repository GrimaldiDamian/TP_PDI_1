import cv2
import numpy as np
import matplotlib.pyplot as plt

def deteccion_renglones(img : np.ndarray) -> tuple:
    """
    Detecta los renglones en una imagen.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        (numpy.ndarray, [numpy.ndarray]): Tupla de imágenes de encabezado y lista con las preguntas.
    """
    img_copy = img.copy()

    img_zeros = img_copy <= 130

    img_row_zeros = img_zeros.any(axis=1)

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x).flatten()

    fila_inicio_encabezado = renglones_indxs[2]
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

def agrupar_columnas(columnas : np.ndarray, min_dist : int =5) -> list:
    """
    Agrupa columnas contiguas en una lista de posiciones.
    Args:
        columnas (numpy.ndarray): Array de posiciones de columnas.
        min_dist (int): Distancia mínima para considerar columnas contiguas.
    Returns:
        list: Lista de posiciones agrupadas.
    """
    agrupadas = []
    inicio = columnas[0]
    
    for i in range(1, len(columnas)):
        if columnas[i] - columnas[i - 1] > min_dist:
            fin = columnas[i - 1]
            agrupadas.append((inicio + fin) // 2)
            inicio = columnas[i]
    
    agrupadas.append((inicio + columnas[-1]) // 2)
    return agrupadas

def segmentar_encabezado(encabezado: np.ndarray) -> list:
    """
    Segmenta el encabezado en celdas usando detección de bordes verticales.
    
    Args:
        encabezado (numpy.ndarray): Imagen del encabezado.
    
    Returns:
        list: Lista de subimágenes (celdas) extraídas del encabezado.
    """
    encabezado_copia = encabezado.copy()
    encabezado_copia[encabezado_copia < 128] = 0
    encabezado_copia[encabezado_copia >= 128] = 255

    w = np.array([[-1, 2, -1],
                  [-1, 2, -1],
                  [-1, 2, -1]])
    f_fil = cv2.filter2D(encabezado_copia, cv2.CV_64F, w)
    f_fil_abs = np.abs(f_fil)
    f_det = f_fil_abs >= f_fil_abs.max() * 0.5

    vertical_projection = np.sum(f_det, axis=0)
    threshold = vertical_projection.max() * 0.7
    columnas_con_linea = np.where(vertical_projection > threshold)[0]

    columnas_segmentadas = agrupar_columnas(columnas_con_linea)

    celdas = []
    for i in range(len(columnas_segmentadas) - 1):
        x1 = columnas_segmentadas[i]
        x2 = columnas_segmentadas[i + 1]
        celda = encabezado[:, x1:x2]
        celdas.append(celda)
    
    return celdas

img = cv2.imread('TP 1/multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)

respuestas_correctas = [
    "A", "A", "B", "A", "D", "B", "B", "C", "B", "A",
    "D", "A", "C", "C", "D", "B", "A", "C", "C", "D",
    "B", "A", "C", "C", "C"
]

encabezado, preguntas = deteccion_renglones(img)
celdas = segmentar_encabezado(encabezado)
nombre, id_valor, code, fecha = celdas[1], celdas[3], celdas[5], celdas[7]