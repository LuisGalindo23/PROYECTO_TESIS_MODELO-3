"""
Genera un grafico estatico (PNG) con los indicadores que devuelve
evaluar() (src/dominio/evaluar_modelo.py): matriz de confusion y barras de
metricas. Es una herramienta de uso local -- evaluar_modelo.py no corre en
Azure, asi que no hace falta un dashboard servido; una imagen alcanza.
"""
import matplotlib
matplotlib.use("Agg")  #Backend sin interfaz grafica: necesario para generar el PNG sin una pantalla disponible
import matplotlib.pyplot as plt
import numpy as np


def graficar_resultados(resultado: dict, ruta_salida: str) -> None:
    figura, (eje_matriz, eje_metricas) = plt.subplots(1, 2, figsize=(11, 4.5))

    #Matriz de confusion (2x2): filas=real, columnas=predicho
    matriz = np.array([
        [resultado["vn"], resultado["fp"]],
        [resultado["fn"], resultado["vp"]],
    ])
    eje_matriz.imshow(matriz, cmap="Blues")
    eje_matriz.set_xticks([0, 1])
    eje_matriz.set_yticks([0, 1])
    eje_matriz.set_xticklabels(["Benigno", "Phishing"])
    eje_matriz.set_yticklabels(["Benigno", "Phishing"])
    eje_matriz.set_xlabel("Predicho")
    eje_matriz.set_ylabel("Real")
    eje_matriz.set_title("Matriz de confusion")
    for fila in range(2):
        for columna in range(2):
            eje_matriz.text(
                columna, fila, str(matriz[fila, columna]),
                ha="center", va="center",
                color="white" if matriz[fila, columna] > matriz.max() / 2 else "black",
            )

    #Barras de metricas (todas expresadas como porcentaje 0-100)
    nombres_metricas = ["VP%", "FP%", "Precision", "Exactitud", "F1"]
    valores_metricas = [
        resultado["porcentaje_verdaderos_positivos"],
        resultado["porcentaje_falsos_positivos"],
        resultado["precision"],
        resultado["exactitud"],
        resultado["f1_score"],
    ]
    eje_metricas.bar(nombres_metricas, valores_metricas, color="#4C72B0")
    eje_metricas.set_ylim(0, 112) #112 en vez de 100: deja espacio para la etiqueta de texto sin que se superponga con el titulo cuando una barra llega a 100
    eje_metricas.set_yticks([0, 20, 40, 60, 80, 100]) #Los ticks del eje siguen mostrando solo el rango 0-100 real
    eje_metricas.set_ylabel("%")
    eje_metricas.set_title("Metricas de evaluacion")
    for i, valor in enumerate(valores_metricas):
        eje_metricas.text(i, valor + 2, f"{valor:.1f}", ha="center")

    figura.tight_layout()
    figura.savefig(ruta_salida)
    plt.close(figura)
