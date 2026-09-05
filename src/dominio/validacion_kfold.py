"""
Validacion cruzada (k-fold) del SVM sobre el propio dataset de entrenamiento: mide que tan bien generaliza el modelo dentro de su misma
distribucion de datos, a diferencia de evaluar_modelo.py. Se usa como indicador honesto al final de entrenar_y_guardar (ver entrenar_modelo.py)
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold #Módulo necesario para la partición del dataset en 5 agrupaciones con la misma proporción de Phishing/Benigno
from sklearn.svm import SVC

from src.dominio.features import crear_vectorizador_tfidf, construir_matriz_features

ETIQUETA_POSITIVA = "phishing"


def validar_con_kfold(df, n_splits: int = 5, semilla: int = 42) -> dict:
    """
    Entrena y evalua un SVC en n_splits particiones estratificadas de `df`(dataFrame, CSV) (columnas asunto/cuerpo/remitente/etiqueta, ya
    limpio). En cada particion se ajusta un vectorizador TF-IDF nuevo solo con el fold de entrenamiento (igual invariante que
    construir_matriz_features en el resto del proyecto: el fold de validacion nunca participa del ajuste), para no filtrar
    informacion del fold de validacion hacia el de entrenamiento.
    """

    particionador = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    f1_por_fold = []

    for indices_train, indices_val in particionador.split(df, df["etiqueta"]):
        df_train = df.iloc[indices_train]
        df_val = df.iloc[indices_val]

        vectorizador_fold = crear_vectorizador_tfidf()
        X_train = construir_matriz_features(df_train, vectorizador_fold, ajustar=True)
        X_val = construir_matriz_features(df_val, vectorizador_fold, ajustar=False)

        modelo_fold = SVC(kernel="linear", random_state=semilla)
        modelo_fold.fit(X_train, df_train["etiqueta"])
        predicciones = modelo_fold.predict(X_val)

        f1_por_fold.append(f1_score(df_val["etiqueta"], predicciones, pos_label=ETIQUETA_POSITIVA))

    return {
        "f1_promedio": float(np.mean(f1_por_fold)),
        "f1_desviacion": float(np.std(f1_por_fold)),
        "f1_por_fold": f1_por_fold,
    }
