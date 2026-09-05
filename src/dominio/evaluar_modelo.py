"""
Escenario de prueba: mide la efectividad real del SVM sobre el dataset de evaluación "Dataset_evaluacion_correos",
se usa el dataset de correos de phishing recolectados de la empresa.

"""

import joblib #Módulo para guardar en disco el modelo y vectorizador
from sklearn.metrics import confusion_matrix #Matriz de confusion: base de los indicadores VP%/FP%

from src.dominio.features import (
    construir_matriz_features,
    extraer_features_numericas,
    NOMBRES_FEATURES,
)
from src.dominio.entrenar_modelo import (
    RUTA_MODELO_DEFECTO,
    RUTA_VECTORIZADOR_DEFECTO,
)
from src.dominio.limpieza_dataset import limpiar_dataset
from src.dominio.graficos_evaluacion import graficar_resultados
from src.dominio.lectura_csv import leer_csv_dataset

ETIQUETA_POSITIVA = "phishing"
RUTA_DATASET = "dataset/Dataset_evaluacion_correos.csv"
RUTA_GRAFICO_DEFECTO = "grafica_indicadores/evaluacion.png"

def evaluar(
    ruta_dataset: str = RUTA_DATASET,
    ruta_modelo: str = RUTA_MODELO_DEFECTO,
    ruta_vectorizador: str = RUTA_VECTORIZADOR_DEFECTO,
) -> dict:
    df_eval = leer_csv_dataset(ruta_dataset) ##Lectura del dataset.
    df_eval = limpiar_dataset(df_eval) #Nulos, espacios, etiquetas y duplicados exactos.
    modelo = joblib.load(ruta_modelo) #Carga el modelo SVM
    vectorizador = joblib.load(ruta_vectorizador) #Carga el vectorizador

    # ajustar=False: reutiliza el vectorizador ya entrenado,
    X_eval = construir_matriz_features(df_eval, vectorizador, ajustar=False) #Construye la matriz de evaluación
    y_real = df_eval["etiqueta"] #Etiquetas de Benigno y Phishing
    y_predicho = modelo.predict(X_eval) #Evalua los correos del dataset
    y_probabilidades = modelo.predict_proba(X_eval) #Establece la probabilidad de clasificación del correo 
    remitentes = df_eval["remitente"].fillna("")

    ejemplos_mal_clasificados = []
    resultados_detallados = []

    #Se guardan en listas los correos bien o mal clasificados
    for idx, (real, predicho) in enumerate(zip(y_real, y_predicho)): 
        fila = df_eval.iloc[idx]
        vector_features = extraer_features_numericas(fila["asunto"], fila["cuerpo"], remitentes.iloc[idx])
        resultados_detallados.append({
            "indice": df_eval.index[idx],
            "asunto": fila["asunto"],
            "etiqueta_real": real,
            "etiqueta_predicha": predicho,
            "correcto": real == predicho,
            "features_ingenieradas": dict(zip(NOMBRES_FEATURES, vector_features)),
            "probabilidades": dict(zip(modelo.classes_, y_probabilidades[idx])),
        })
        if real != predicho:
            ejemplos_mal_clasificados.append({
                "asunto": fila["asunto"],
                "etiqueta_real": real,
                "etiqueta_predicha": predicho,
            })

    # Matriz de confusion con "phishing" como clase positiva: VN, FP, FN, VP
    # (orden fijo de confusion_matrix cuando labels=[negativa, positiva]).
    vn, fp, fn, vp = confusion_matrix(y_real, y_predicho, labels=["benigno", ETIQUETA_POSITIVA]).ravel()

    # Porcentaje de verdaderos positivos = VP / (VP + FN) x 100
    # Mide que proporcion de los correos que REALMENTE son phishing fue detectada como tal.
    # (VP+FN) = 0 solo si el dataset de evaluacion no trae ningun correo phishing real.
    porcentaje_verdaderos_positivos = (vp / (vp + fn) * 100) if (vp + fn) > 0 else 0.0
    # Porcentaje de falsos positivos = FP / (FP + VN) x 100
    # Mide que proporcion de los correos REALMENTE legitimos fue clasificada incorrectamente como phishing.
    # (FP+VN) = 0 solo si el dataset de evaluacion no trae ningun correo legitimo.
    porcentaje_falsos_positivos = (fp / (fp + vn) * 100) if (fp + vn) > 0 else 0.0
    # Precision = VP / (VP + FP) x 100
    # Mide que proporcion de los correos clasificados como phishing REALMENTE lo eran.
    # (VP+FP) = 0 solo si el modelo no predijo ningun correo como phishing.
    precision = (vp / (vp + fp) * 100) if (vp + fp) > 0 else 0.0
    # Exactitud (Accuracy) = (VP + VN) / (VP + VN + FP + FN) x 100
    # Mide que proporcion del total de correos fue clasificada correctamente, sin distinguir clase.
    exactitud = (vp + vn) / (vp + vn + fp + fn) * 100
    # F1-Score = 2 x (Precision x Recall) / (Precision + Recall)
    # Media armonica entre Precision y Recall (VP%): resume ambos en un solo valor,
    # util cuando hay desbalance entre correos phishing y legitimos.
    f1_score = (
        2 * (precision * porcentaje_verdaderos_positivos) / (precision + porcentaje_verdaderos_positivos)
        if (precision + porcentaje_verdaderos_positivos) > 0
        else 0.0
    )

    return {
        "porcentaje_verdaderos_positivos": porcentaje_verdaderos_positivos,
        "porcentaje_falsos_positivos": porcentaje_falsos_positivos,
        "precision": precision,
        "exactitud": exactitud,
        "f1_score": f1_score,
        "vp": vp,
        "vn": vn,
        "fp": fp,
        "fn": fn,
        "ejemplos_mal_clasificados": ejemplos_mal_clasificados,
        "resultados_detallados": resultados_detallados,
    }


# Features cuyo valor es 0.0/1.0 y se leen mejor como si/no que como numero.
_FEATURES_BOOLEANAS = {"url_sospechosa", "datos_sensibles", "saludo_generico"}


def _formatear_features(features: dict) -> str:
    partes = []
    for nombre, valor in features.items():
        if nombre in _FEATURES_BOOLEANAS:
            partes.append(f"{nombre}={'si' if valor else 'no'}")
        elif nombre == "mayusculas":
            partes.append(f"{nombre}={valor:.0%}")
        else:
            partes.append(f"{nombre}={valor:.0f}")
    return ", ".join(partes)


if __name__ == "__main__":
    resultado = evaluar()
    print("=== Metricas del escenario de prueba (dataset de evaluacion independiente) ===")
    print(f"Porcentaje de verdaderos positivos: {resultado['porcentaje_verdaderos_positivos']:.2f}%")
    print(f"Porcentaje de falsos positivos: {resultado['porcentaje_falsos_positivos']:.2f}%")
    print(f"Precision: {resultado['precision']:.2f}%")
    print(f"Exactitud: {resultado['exactitud']:.2f}%")
    print(f"F1-Score: {resultado['f1_score']:.2f}%")
    print(f"VP={resultado['vp']}, VN={resultado['vn']}, FP={resultado['fp']}, FN={resultado['fn']}")
    graficar_resultados(resultado, RUTA_GRAFICO_DEFECTO)
    print(f"Grafico guardado en: {RUTA_GRAFICO_DEFECTO}")
    print() #Imprime un salto de linea
    if resultado["ejemplos_mal_clasificados"]:
        print("=== Correos mal clasificados ===")
        for ejemplo in resultado["ejemplos_mal_clasificados"]:
            print(f"- '{ejemplo['asunto']}': real={ejemplo['etiqueta_real']}, predicho={ejemplo['etiqueta_predicha']}")
        print()

    # Listado completo por índice, para contrastar contra el etiquetado
    # original en el CSV de evaluacion (ver RUTA_DATASET).
    for etiqueta in ("benigno", "phishing"):
        correos_categoria = [
            r for r in resultado["resultados_detallados"] if r["etiqueta_real"] == etiqueta
        ]
        print(f"=== Correos reales '{etiqueta}' (indice: predicho | ok/error) ===")
        total_ok = 0
        for r in correos_categoria:
            estado = "ok" if r["correcto"] else "ERROR"
            total_ok += r["correcto"]
            probabilidades_texto = ", ".join(
                f"{clase}={prob:.2%}" for clase, prob in r["probabilidades"].items()
            )
            print(f"[{r['indice']}] '{r['asunto']}' -> predicho={r['etiqueta_predicha']} ({estado})")
            print(f"    features: {_formatear_features(r['features_ingenieradas'])}")
            print(f"    SVM (probabilidad por clase): {probabilidades_texto}")
        total_error = len(correos_categoria) - total_ok
        print(f"Total: {total_ok} ok, {total_error} error de {len(correos_categoria)}")
        print()
