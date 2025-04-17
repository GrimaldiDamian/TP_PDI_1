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
    img_row_zeros = (img <= 130).any(axis=1)

    x = np.diff(img_row_zeros)
    renglones_indxs = np.argwhere(x).flatten()

    encabezado = img[renglones_indxs[2]+2:renglones_indxs[3], :]

    preguntas = [
        img[renglones_indxs[i] + 1:renglones_indxs[i + 1] +1, :] 
        for i in range(4, len(renglones_indxs)-1, 2)  # Avanzamos de 2 en 2 para pares (inicio, fin)
        if renglones_indxs[i + 1] - renglones_indxs[i] > 10  # Aseguramos que sea un bloque con contenido
    ]

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
            inicio = columnas[i] + 1
    
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

    celdas = [
        encabezado[:, columnas_segmentadas[i]+1:columnas_segmentadas[i + 1]]
        for i in range(len(columnas_segmentadas) - 1)
    ]

    return celdas

def detectar_caracteres(img: np.ndarray, th_area: int = 2, umbral_espacio = 5) -> list:
    """
    Detecta caracteres y espacios en una imagen binaria.

    Args:
        img (np.ndarray): Imagen en escala de grises o binaria.
        umbral_espacio (int): Distancia mínima entre caracteres para considerar un espacio.

    Returns:
        list: Lista con los bounding boxes de cada carácter y " " si hay un espacio.
    """
    _, celda_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, connectivity=8, ltype=cv2.CV_32S)

    # Filtrar componentes con área demasiado pequeña (ruido)
    ix_area = stats[:, cv2.CC_STAT_AREA] > th_area
    stats_filtradas = stats[ix_area]

    # Ignorar la primera componente (fondo)
    caracteres = []
    for i in range(1, len(stats_filtradas)):
        x, y, w, h, area = stats_filtradas[i]
        caracteres.append((x, y, w, h))

    # Ordenar caracteres de izquierda a derecha
    caracteres = sorted(caracteres, key=lambda b: b[0])

    espacios = []
    for i in range(len(caracteres) - 1):
        _, _, w, _ = caracteres[i]
        x_siguiente, _, _, _ = caracteres[i + 1]
        
        # Calcular espacio entre caracteres consecutivos
        espacio_entre = x_siguiente - (caracteres[i][0] + w)

        if espacio_entre >= umbral_espacio:
            espacios.append(i)

    return caracteres, espacios

def validacion_encabezado (caracteres : list, espacios : list, tipo_encabezado : str):
    """
    """
    total_caracteres = len(caracteres)
    total_espacios = len(espacios)
    reglas = {
        "name":  total_espacios >= 1 and total_caracteres <= 25,
        "id":    total_espacios == 0 and total_caracteres == 8,
        "code":  total_espacios == 0 and total_caracteres == 1,
        "date":  total_espacios == 0 and total_caracteres == 8,
    }

    estado = "OK" if reglas.get(tipo_encabezado.lower(), False) else "MAL"
    return f"{tipo_encabezado} : {estado}"

respuestas_correctas = [
    "A", "A", "B", "A", "D", "B", "B", "C", "B", "A",
    "D", "A", "C", "C", "D", "B", "A", "C", "C", "D",
    "B", "A", "C", "C", "C"
]

for i in range(1,6):
    archivo = f'multiple_choice_{i}.png'
    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    encabezado, preguntas = deteccion_renglones(img)
    celdas = segmentar_encabezado(encabezado)
    tipos_encabezado = {
        1: "name",
        3: "id",
        5: "code",
        7: "date"
    }
    print(f"{archivo} :")
    for i in range(1,8,2):
        caracteres, espacio = detectar_caracteres(celdas[i])
        print(validacion_encabezado(caracteres,espacio,tipos_encabezado[i]))