# TP 1 PDI

Este proyecto requiere Python para su ejecución. A continuación, se detallan las instrucciones para configurar el entorno y ejecutar el proyecto correctamente.

## Requisitos

- Tener instalado [Python](https://www.python.org/downloads/) (se recomienda la versión 3.9 o superior).
- Tener instalado [pip](https://pip.pypa.io/en/stable/) para gestionar las dependencias de Python.

## Instalación

### 1. Crear un entorno virtual

Es recomendable utilizar un entorno virtual para gestionar las dependencias del proyecto y evitar conflictos con otras aplicaciones.

1. Abre Visual Studio Code.
2. Abre la terminal con `Ctrl + J` y escribe `cmd` para abrir la línea de comandos.
3. Para crear un entorno virtual, ejecuta el siguiente comando (reemplaza `"nombre_del_entorno"` por el nombre que desees para el entorno):
   python -m venv nombre_del_entorno 

4. Y para activarlo:
nombre_del_entorno\Scripts\activate -> Windows
source nombre_del_entorno/bin/activate -> Linux/MacOS

### 2. Instalar las librerias
Una vez ejecutado el entorno debes instalar las librerias, para ello usa el siguiente comando:

pip install -r requirements.txt

Otra opcion puede ser, ejecutar desde CMD con el entorno virtual activado:
pip install opencv-python numpy matplotlib


### 3. Ejecucion
Para ejecutar el proyecto, simplemente utiliza el siguiente comando de Python, reemplazando "nombre_del_archivo.py" por el nombre del archivo que deseas ejecutar:

python nombre_del_archivo.py


