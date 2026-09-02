from src.dominio.graficos_evaluacion import graficar_resultados

RESULTADO_EJEMPLO = {
    "porcentaje_verdaderos_positivos": 22.86,
    "porcentaje_falsos_positivos": 0.0,
    "precision": 100.0,
    "exactitud": 22.86,
    "f1_score": 37.21,
    "vp": 8,
    "vn": 0,
    "fp": 0,
    "fn": 27,
}


def test_graficar_resultados_crea_un_archivo_png(tmp_path):
    ruta_salida = tmp_path / "evaluacion.png"

    graficar_resultados(RESULTADO_EJEMPLO, str(ruta_salida))

    assert ruta_salida.exists()
    with open(ruta_salida, "rb") as f:
        firma_png = f.read(8)
    assert firma_png == b"\x89PNG\r\n\x1a\n"


def test_graficar_resultados_no_falla_con_matriz_de_confusion_toda_en_cero(tmp_path):
    resultado_vacio = {
        "porcentaje_verdaderos_positivos": 0.0,
        "porcentaje_falsos_positivos": 0.0,
        "precision": 0.0,
        "exactitud": 0.0,
        "f1_score": 0.0,
        "vp": 0, "vn": 0, "fp": 0, "fn": 0,
    }
    graficar_resultados(resultado_vacio, str(tmp_path / "evaluacion_vacia.png"))
