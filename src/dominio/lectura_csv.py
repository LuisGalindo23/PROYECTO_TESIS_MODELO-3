"""
Lectura resiliente de los CSV de dataset del proyecto. En vez de exigir que el archivo se reconvierta cada vez (lo que lo dejaria
dificil de seguir editando en Excel), se intenta leer en el formato esperado del proyecto primero y, si falla, se reintenta con el
formato que exporta Excel.
"""

import pandas as pd #Módulo para la lectura de archivos csv


def leer_csv_dataset(ruta: str) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta)  #Formato esperado: UTF-8, separador ","
    except UnicodeDecodeError:
        return pd.read_csv(ruta, sep=";", encoding="cp1252")  #Formato que exporta Excel en región español
