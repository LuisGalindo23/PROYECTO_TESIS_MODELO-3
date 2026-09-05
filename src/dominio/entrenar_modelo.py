"""
Entrena el modelo SVM de detección de phishing haciendo uso del dataset "Dataset_entrenamiento_correos" y guarda el modelo entrenado
junto con el vectorizador TF-IDF ajustado, para que la Azure Function (function_app.py) pueda reutilizarlos sin volver a entrenar.
"""

import joblib #Módulo para guardar en disco el modelo y vectorizador

from sklearn.svm import SVC #Módulo de SVM, específicamente la clase SVC (Support Vector Classification)
from sklearn.calibration import CalibratedClassifierCV #Módulo para declarar probabilidades en la clase de SVM

from src.dominio.features import crear_vectorizador_tfidf, construir_matriz_features
from src.dominio.limpieza_dataset import limpiar_dataset
from src.dominio.validacion_kfold import validar_con_kfold #Módulo para evaluar el entrenamiento del modelo con el dataset en diferentes participaciones estratificadas manteniendo la proporción de categorías (Phishing/Benigno)
from src.dominio.lectura_csv import leer_csv_dataset

RUTA_CSV_DEFECTO = "dataset/Dataset_entrenamiento_correos.csv"
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
    ds_correos = leer_csv_dataset(ruta_csv) #Lectura del dataset.
    ds_correos = limpiar_dataset(ds_correos) #Nulos, espacios, etiquetas y duplicados exactos.

    vectorizador = crear_vectorizador_tfidf()
    X_train = construir_matriz_features(ds_correos, vectorizador, ajustar=True)
    y_train = ds_correos["etiqueta"]

    # Kernel lineal: Separa las clases (Benigno y Phishing) en un hiperplano lineal.
    # (TF-IDF genera miles de columnas) y es interpretable (los pesos de cada término indican su influencia en la clasificación)
    # se ajusta una sola vez sobre todo X_train (la validacion cruzada solo se usa para calibrar),
    # ensemble = false, declara que se usará un único modelo de SVM en las calibraciones/estratificaciones.
    modelo = CalibratedClassifierCV(SVC(kernel="linear", random_state=SEMILLA_ALEATORIA), ensemble=False)
    modelo.fit(X_train, y_train) #Modelo entrena con la matriz y etiquetas ingresadas como argumentos

    joblib.dump(modelo, ruta_modelo) #Registra el modelo svm entrenado en la ruta
    joblib.dump(vectorizador, ruta_vectorizador) #Registra el vectorizador tfidf entrenado en la ruta

    # El k-fold entrena/evalua en particiones separadas del mismo dataset, dando una medida honesta de que tan bien generaliza el modelo dentro
    # de su propia distribucion de datos.
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
