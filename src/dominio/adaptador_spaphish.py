"""
Adapta el esquema del dataset publico SpaPhish (columnas subject, body,
Label, urls, y otras 43 columnas tecnicas/de anotacion psicologica no
usadas por este proyecto; separador ',' con BOM UTF-8, pese a que la
documentacion de Mendeley dice ';') al esquema del proyecto (asunto,
cuerpo, remitente, etiqueta). No hace validacion/limpieza -- eso lo hace
limpiar_dataset (src/dominio/limpieza_dataset.py) despues.

SpaPhish tampoco trae remitente (queda vacio, mismo limite que
SpearPhishMX). A diferencia de SpearPhishMX, sus URLs (columna 'urls') NO
vienen defangeadas -- se extraen y se reinsertan en el cuerpo tal cual,
sin necesidad de revertir ningun defanging.
"""
import re

import pandas as pd

MAPA_ETIQUETA = {1: "phishing", 0: "benigno"}

#Cada URL en la columna "urls" viene envuelta en sus propios corchetes: "[url1],[url2],..."
_PATRON_URL_ENTRE_CORCHETES = re.compile(r"\[(.*?)\]")


def _extraer_urls(campo_urls) -> list:
    """
    Parsea la columna 'urls' de SpaPhish. A diferencia de SpearPhishMX, las
    URLs aca no vienen defangeadas (esquema http/https real), asi que no
    hace falta revertir ningun defanging -- solo extraerlas de su formato
    "[url1],[url2],...". Sin URLs (url_count=0), el valor es NaN.
    """
    if not isinstance(campo_urls, str):
        return []
    return _PATRON_URL_ENTRE_CORCHETES.findall(campo_urls)


def _sin_saltos_de_linea(texto):
    """
    Reemplaza saltos de linea internos por un espacio. Un \\n dentro del
    texto, mezclado con el \\r\\n que usa el propio CSV como separador de
    fila, confunde a Excel al mostrarlo (una linea en blanco del correo
    original se ve como una fila vacia). No afecta a TF-IDF ni al regex de
    URLs de features.py (\\s ya trata \\n igual que un espacio) -- es
    puramente para que el CSV se pueda inspeccionar bien en Excel.
    """
    if not isinstance(texto, str):
        return texto
    return re.sub(r"[\r\n]+", " ", texto)


def adaptar_spaphish(df: pd.DataFrame) -> pd.DataFrame:
    urls_por_fila = df["urls"].apply(_extraer_urls) #Lista de URLs reales por fila
    cuerpos = [
        _sin_saltos_de_linea(cuerpo if not urls else f"{cuerpo} {' '.join(urls)}") #Si hay URLs, se agregan al final para que features.py las detecte
        for cuerpo, urls in zip(df["body"], urls_por_fila)
    ]

    return pd.DataFrame({
        "asunto": df["subject"].apply(_sin_saltos_de_linea),
        "cuerpo": cuerpos,
        "remitente": "", #No disponible en el dataset de origen
        "etiqueta": df["Label"].map(MAPA_ETIQUETA),
    })


RUTA_CSV_ORIGEN_DEFECTO = "dataset/publico/spaphish/SpaPhish_dataset.csv"
RUTA_CSV_ADAPTADO_DEFECTO = "dataset/spaphish_adaptado.csv"


if __name__ == "__main__":
    #encoding="utf-8-sig" quita el BOM inicial; el separador real es ',' pese a lo que dice la documentacion
    df_origen = pd.read_csv(RUTA_CSV_ORIGEN_DEFECTO, sep=",", encoding="utf-8-sig")
    df_adaptado = adaptar_spaphish(df_origen)
    df_adaptado.to_csv(RUTA_CSV_ADAPTADO_DEFECTO, index=False)
    print(f"{len(df_adaptado)} correos adaptados. Guardado en: {RUTA_CSV_ADAPTADO_DEFECTO}")
