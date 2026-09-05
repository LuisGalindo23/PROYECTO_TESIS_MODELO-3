"""
Limpieza de un DataFrame de correos (columnas: asunto, cuerpo, remitente, etiqueta) antes de construir features.
Se ejecuta antes de construir_matriz_features, tanto en entrenamiento (entrenar_modelo.py) como en evaluacion (evaluar_modelo.py),
para que ambos apliquen exactamente la misma limpieza al leer sus respectivos CSV.
"""
import pandas as pd  #Modulo de lectura/manipulacion del CSV como tabla (DataFrame)

COLUMNAS_REQUERIDAS = ["asunto", "cuerpo", "remitente", "etiqueta"]  #Columnas que debe tener si o si el CSV
COLUMNAS_DE_TEXTO = ["asunto", "cuerpo", "remitente"]  #Subconjunto de texto libre (sin la etiqueta)
ETIQUETAS_VALIDAS = {"benigno", "phishing"}  #Unicos valores aceptados en la columna 'etiqueta'


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    #Lista las columnas requeridas que no estan presentes en el CSV
    columnas_faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in df.columns]
    if columnas_faltantes:  #Si falta alguna, no tiene sentido seguir limpiando
        raise ValueError(f"Faltan columnas requeridas en el dataset: {columnas_faltantes}")

    df = df.copy()  #Copia para no mutar el DataFrame original que recibio la funcion
    df[COLUMNAS_DE_TEXTO] = df[COLUMNAS_DE_TEXTO].fillna("")  #Nulos (NaN) en asunto/cuerpo/remitente -> string vacio
    for columna in COLUMNAS_DE_TEXTO:
        df[columna] = df[columna].str.strip()  #Quita espacios sobrantes al inicio/final (no toca mayusculas ni el contenido interno)

    if df["etiqueta"].isna().any():  #Una etiqueta nula no se puede usar para entrenar/evaluar
        raise ValueError("La columna 'etiqueta' tiene valores nulos")

    df["etiqueta"] = df["etiqueta"].astype(str).str.strip().str.lower()  #Normaliza " Phishing " -> "phishing"

    #Valores que quedaron fuera de {"benigno", "phishing"} tras normalizar (typos, otras clases, etc.)
    etiquetas_invalidas = set(df["etiqueta"]) - ETIQUETAS_VALIDAS
    if etiquetas_invalidas:
        raise ValueError(
            f"La columna 'etiqueta' tiene valores fuera de {ETIQUETAS_VALIDAS}: {etiquetas_invalidas}"
        )

    # Se conserva el indice original (no reset_index) para que, en el dataset de evaluacion, evaluar_modelo.py pueda seguir
    # reportando el indice de cada correo tal como aparece en su CSV de origen.
    df = df.drop_duplicates(subset=COLUMNAS_DE_TEXTO, keep="first")  #Mismo asunto+cuerpo+remitente repetido -> se conserva solo la primera fila
    return df
