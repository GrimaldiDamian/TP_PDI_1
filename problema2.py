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
        th_area (int): Umbral de área mínima para considerar un componente como carácter.
        umbral_espacio (int): Distancia mínima entre caracteres para considerar un espacio.

    Returns:
        list: Lista con los bounding boxes de cada carácter y " " si hay un espacio.
    """
    _, celda_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, connectivity=8, ltype=cv2.CV_32S)

    ix_area = stats[:, cv2.CC_STAT_AREA] > th_area
    stats_filtradas = stats[ix_area]

    caracteres = []
    for i in range(1, len(stats_filtradas)):
        x, y, w, h, area = stats_filtradas[i]
        caracteres.append((x, y, w, h))

    caracteres = sorted(caracteres, key=lambda b: b[0])

    espacios = []
    for i in range(len(caracteres) - 1):
        _, _, w, _ = caracteres[i]
        x_siguiente, _, _, _ = caracteres[i + 1]
        
        espacio_entre = x_siguiente - (caracteres[i][0] + w)

        if espacio_entre >= umbral_espacio:
            espacios.append(i)

    return caracteres, espacios

def validacion_encabezado (caracteres : list, espacios : list, tipo_encabezado : str):
    """
    Valida el encabezado según el tipo especificado.
    Args:
        caracteres (list): Lista de bounding boxes de caracteres.
        espacios (list): Lista de índices de espacios entre caracteres.
        tipo_encabezado (str): Tipo de encabezado a validar ("name", "id", "code", "date").
    Returns:
        str: Mensaje de validación del encabezado.
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

def detectar_respuesta_marcada(img: np.ndarray) -> list:
    """
    A partir de la seccion de la pregunta, detecta cuál burbuja está marcada (de A a E)
    y retorna la/s letra/s correspondiente.
    Args:
        img (numpy.ndarray): Sección de la pregunta
    Returns:
        list: Lista de letras de respuestas marcadas o no
    """
    _, thresh = cv2.threshold(img, 94, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    burbujas = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 100 < area < 1000:
            x, y, w, h = cv2.boundingRect(c)
            aspecto = w / float(h)
            if 0.8 < aspecto < 1.2:
                burbujas.append((x, y, w, h, c))

    # Ordenar burbujas de izquierda a derecha
    burbujas = sorted(burbujas, key=lambda x: x[0])

    # Evaluamos el nivel del relleno (más claro = más relleno)
    niveles = []
    for x, y, w, h, c in burbujas:
        roi = thresh[y:y+h, x:x+w]
        nivel = cv2.countNonZero(roi)
        niveles.append(nivel)

    # Creamos un umbral, para detectar si hay mas de una burbuja marcada
    max_nivel = max(niveles)
    umbral = 0.85 * max_nivel

    # Detectamos las burbujas marcadas
    indices_marcados = [i for i, nivel in enumerate(niveles) if nivel >= umbral]

    opciones = ["A", "B", "C", "D", "E"]
    respuestas = [opciones[i] for i in indices_marcados if i < len(opciones)]

    return respuestas

def validacion_respuestas(respuestas: list, respuestas_correctas: list) -> int:
    """
    Valida las respuestas marcadas contra las correctas.
    Args:
        respuestas (list): Lista de respuestas detectadas.
        respuestas_correctas (list): Lista de respuestas correctas.
    Returns:
        int: Cantidad de respuestas correctas.
    """
    cont_correctas = 0
    for k, respuesta in enumerate(respuestas):
        if len(respuesta) == 1:
            if respuesta[0] == respuestas_correctas[k]:
                print(f"Pregunta {k+1}: OK")
                cont_correctas += 1
            else:
                print(f"Pregunta {k+1}: MAL")
        else:
            print(f"Pregunta {k+1}: MAL")
    return cont_correctas

def informe_final(aprobados: list, desaprobados: list) -> None:
    """
    Genera una imagen con el informe final de los alumnos aprobados y desaprobados.
    Se guardara como "informe_final.png".
    Se mostrara el nombre de las personas aprobadas con una carita sonriente ( :) )
    y las desaprobadas con una carita triste ( :( ).
    Args:
        aprobados (list): Lista de crop de nombres de los aprobados.
        desaprobados (list): Lista de crop de nombres de los desaprobados.
    Returns:
        None
    """
    salida = "informe_final.png"
    imagenes = []
    for nombre in aprobados:
        imagen = nombre.copy()
        cv2.putText(imagen, ':)', (100, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        imagenes.append(imagen)

    for nombre in desaprobados:
        imagen = nombre.copy()
        cv2.putText(imagen, ':(', (100, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        imagenes.append(imagen)

    altura_total = sum([img.shape[0] for img in imagenes])
    ancho_max = max([img.shape[1] for img in imagenes])

    # Creamos una imagen con fondo negro para el informe
    informe = np.array([[255]*ancho_max]*altura_total,dtype="uint8")

    y = 0
    for img in imagenes:
        informe[y:y+img.shape[0], 0:img.shape[1]] = img
        y += img.shape[0]

    cv2.imwrite(salida, informe)

respuestas_correctas = [
    "A", "A", "B", "A", "D", "B", "B", "C", "B", "A",
    "D", "A", "C", "C", "D", "B", "A", "C", "C", "D",
    "B", "A", "C", "C", "C"
]

aprobados = []
desaprobados = []

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
    for j in range(1,8,2):
        caracteres, espacio = detectar_caracteres(celdas[j])
        print(validacion_encabezado(caracteres,espacio,tipos_encabezado[j]))
    respuestas = []
    for pregunta in preguntas:
        respuesta = detectar_respuesta_marcada(pregunta)
        respuestas.append(respuesta)
    cantidad_corectas = validacion_respuestas(respuestas, respuestas_correctas)
    print(f"Cantidad de respuestas correctas: {cantidad_corectas}")
    if cantidad_corectas >= 20: aprobados.append(celdas[1])
    else: desaprobados.append(celdas[1])
informe_final(aprobados, desaprobados)

informe = cv2.imread("informe_final.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(informe, cmap='gray')
plt.show()