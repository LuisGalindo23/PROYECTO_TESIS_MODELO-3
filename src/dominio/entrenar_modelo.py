"""
Entrena el modelo SVM de detección de phishing sobre todo el dataset en
español y guarda el modelo entrenado junto con el vectorizador TF-IDF
ajustado, para que el monitor de Outlook (src/monitor_outlook.py) pueda
reutilizarlos sin volver a entrenar.

La evaluación del modelo (src/dominio/evaluar_modelo.py) usa un dataset
independiente, no un split de este archivo.
"""

import pandas as pd #Módulo lectura de archivos CSV como tabla
import joblib #Módulo para guardar en disco el modelo y vectorizador

from sklearn.svm import SVC #Módulo de SVM, específicamente la clase SVC (Support Vector Classification)
from sklearn.calibration import CalibratedClassifierCV #Reemplaza a SVC(probability=True), deprecado desde sklearn 1.9 (se elimina en 1.11)

from src.dominio.features import crear_vectorizador_tfidf, construir_matriz_features #Se importan estas clases, ya que por arquitectura el modelo no tiene conocimiento de la existencia de Outlook
from src.dominio.limpieza_dataset import limpiar_dataset
from src.dominio.validacion_kfold import validar_con_kfold

RUTA_CSV_DEFECTO = "dataset/correos_dataset.csv"
RUTA_MODELO_DEFECTO = "modelo/svm_phishing.pkl"
RUTA_VECTORIZADOR_DEFECTO = "modelo/vectorizador_tfidf.pkl"

# Número sin repercusión en caso de cambio, funciona como base para la reproducibilidad
# Reproducibilidad (capacidad de obtener los mismos resultados en un experimento)
SEMILLA_ALEATORIA = 42

def entrenar_y_guardar(
    ruta_csv: str = RUTA_CSV_DEFECTO,
    ruta_modelo: str = RUTA_MODELO_DEFECTO,
    ruta_vectorizador: str = RUTA_VECTORIZADOR_DEFECTO,
) -> dict: #Retorna un diccionario, que guarda pares clave-valor = {"clave": valor}
    ds_correos = pd.read_csv(ruta_csv) #Lectura del dataset
    ds_correos = limpiar_dataset(ds_correos) #Nulos, espacios, etiquetas y duplicados exactos

    vectorizador = crear_vectorizador_tfidf()
    X_train = construir_matriz_features(ds_correos, vectorizador, ajustar=True)
    y_train = ds_correos["etiqueta"]

    # Kernel lineal: buen desempeño en espacios de alta dimensión como texto
    # Kernel lineal: Separa las clases (Benigno y Phishing) en un hiperplano lineal
    # (TF-IDF genera miles de columnas) y es interpretable (los pesos de cada
    # término indican su influencia en la clasificación)
    # CalibratedClassifierCV(ensemble=False) agrega predict_proba calibrando el SVC lineal
    # por fuera (el SVM en si no produce probabilidades) -- reemplaza a SVC(probability=True),
    # que hacia lo mismo mismo internamente pero esta deprecado. Con ensemble=False el SVC final
    # se ajusta una sola vez sobre todo X_train (la validacion cruzada solo se usa para calibrar),
    # asi que el coef_ que necesita explicar_clasificacion (features.py) sigue siendo el de un
    # unico modelo lineal, solo que anidado un nivel mas adentro (ver _coef_del_modelo).
    modelo = CalibratedClassifierCV(SVC(kernel="linear", random_state=SEMILLA_ALEATORIA), ensemble=False)
    modelo.fit(X_train, y_train) #Modelo entrena con la matriz y etiquetas ingresadas como argumentos

    joblib.dump(modelo, ruta_modelo) #Crea o actualiza el modelo svm
    joblib.dump(vectorizador, ruta_vectorizador) #Crea o actualiza el vectorizador tfidf

    # Reemplaza a modelo.score(X_train, y_train): esa exactitud se mide sobre
    # los mismos datos de entrenamiento, asi que no dice nada sobre
    # generalizacion (un SVM con miles de columnas TF-IDF sobre ~1000 filas
    # tiende a memorizarlas -- daba 100% sin que eso significara nada bueno).
    # El k-fold entrena/evalua en particiones separadas del mismo dataset,
    # dando una medida honesta de que tan bien generaliza el modelo dentro
    # de su propia distribucion de datos (ver validacion_kfold.py).
    resultado_kfold = validar_con_kfold(ds_correos)
    return {
        "f1_promedio_kfold": resultado_kfold["f1_promedio"],
        "f1_desviacion_kfold": resultado_kfold["f1_desviacion"],
    }


if __name__ == "__main__":
    resultado = entrenar_y_guardar()
    print(
        f"Modelo entrenado. F1 promedio (5-fold): {resultado['f1_promedio_kfold']:.2%} "
        f"+/- {resultado['f1_desviacion_kfold']:.2%}"
    )
    print(f"Modelo guardado en: {RUTA_MODELO_DEFECTO}")
    print(f"Vectorizador guardado en: {RUTA_VECTORIZADOR_DEFECTO}")
