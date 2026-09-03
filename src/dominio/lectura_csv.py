"""
Lectura resiliente de los CSV de dataset del proyecto. Estos archivos se
editan a mano en Excel (agregar/corregir correos), y Excel en
configuracion regional en espanol guarda por defecto en cp1252 con ";"
como separador, no UTF-8 con ",". En vez de exigir que el archivo se
reconvierta cada vez (lo que lo dejaria dificil de seguir editando en
Excel), se intenta leer en el formato esperado del proyecto primero y,
si falla, se reintenta con el formato que exporta Excel -- el archivo en
disco nunca se modifica, solo cambia como se interpreta al leerlo.
"""
import pandas as pd


def leer_csv_dataset(ruta: str) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta)  #Formato esperado: UTF-8, separador ","
    except UnicodeDecodeError:
        return pd.read_csv(ruta, sep=";", encoding="cp1252")  #Formato que exporta Excel en regional espanol
