"""
Tests de caracterizacion para src/dominio/features.py: fijan el
comportamiento actual (ya en produccion, sin cobertura previa) para que un
cambio futuro que lo altere sin querer se detecte aqui.
"""
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from src.dominio.features import (
    contar_urls,
    tiene_url_sospechosa,
    dominio_coincide,
    contar_palabras_urgencia,
    proporcion_mayusculas,
    solicita_datos_sensibles,
    saludo_generico,
    extraer_features_numericas,
    construir_matriz_features,
    crear_vectorizador_tfidf,
    explicar_clasificacion,
    NOMBRES_FEATURES,
)


def test_contar_urls_cuenta_multiples_urls():
    texto = "Visita http://a.com y tambien https://b.com/x para mas info"
    assert contar_urls(texto) == 2


def test_contar_urls_retorna_cero_sin_urls():
    assert contar_urls("Este correo no tiene enlaces") == 0


def test_tiene_url_sospechosa_detecta_ip_literal():
    assert tiene_url_sospechosa("Ingresa en http://192.168.1.1/login") is True


def test_tiene_url_sospechosa_detecta_acortador_conocido():
    assert tiene_url_sospechosa("Click aqui: http://bit.ly/xyz123") is True


def test_tiene_url_sospechosa_false_para_dominio_normal():
    assert tiene_url_sospechosa("Visita http://banco.com/login") is False


def test_dominio_coincide_true_cuando_remitente_y_url_son_del_mismo_dominio():
    assert dominio_coincide("alertas@banco.com", "Verifica en http://banco.com/verificar") is True


def test_dominio_coincide_false_cuando_remitente_y_url_son_de_dominios_distintos():
    assert dominio_coincide("alertas@banco.com", "Verifica en http://banco-fake.com/verificar") is False


def test_dominio_coincide_true_cuando_no_hay_urls_en_el_texto():
    assert dominio_coincide("alertas@banco.com", "Este correo no tiene enlaces") is True


def test_contar_palabras_urgencia_cuenta_las_frases_conocidas():
    texto = "Esto es URGENTE, actue ahora mismo antes de que se bloquee"
    # "urgente" y "ahora mismo" son dos frases distintas de _PALABRAS_URGENCIA
    assert contar_palabras_urgencia(texto) == 2


def test_contar_palabras_urgencia_retorna_cero_sin_coincidencias():
    assert contar_palabras_urgencia("Hola, como estas") == 0


def test_proporcion_mayusculas_calcula_la_proporcion_correcta():
    # "URGENTE" (7 mayusculas) + "aviso" (5 minusculas) = 12 letras, 7 en mayuscula
    assert proporcion_mayusculas("URGENTE aviso") == 7 / 12


def test_proporcion_mayusculas_retorna_cero_si_no_hay_letras():
    assert proporcion_mayusculas("12345 !!!") == 0.0


def test_solicita_datos_sensibles_detecta_patron_conocido():
    assert solicita_datos_sensibles("Por favor confirme su contraseña aqui") is True


def test_solicita_datos_sensibles_false_sin_patron():
    assert solicita_datos_sensibles("Reunion programada para el jueves") is False


def test_saludo_generico_detecta_saludo_conocido():
    assert saludo_generico("Estimado cliente, le informamos...") is True


def test_saludo_generico_false_con_nombre_propio():
    assert saludo_generico("Hola Juan, te escribo para...") is False


def test_extraer_features_numericas_devuelve_el_vector_en_el_orden_documentado():
    vector = extraer_features_numericas(
        asunto="URGENTE: verifique su contraseña",
        cuerpo="Estimado cliente, ingrese en http://192.168.1.1/login antes de 24 horas",
        remitente="alertas@banco.com",
    )
    assert len(vector) == len(NOMBRES_FEATURES) == 7
    urls, url_sospechosa, dominio_coincide_valor, urgencia, mayusculas, datos_sensibles, saludo = vector
    assert urls == 1.0
    assert url_sospechosa == 1.0  # IP literal
    assert dominio_coincide_valor == 0.0  # banco.com (remitente) != 192.168.1.1 (url)
    assert urgencia >= 1.0  # "verifique" esta en _PALABRAS_URGENCIA
    assert datos_sensibles == 1.0  # "contraseña"
    assert saludo == 1.0  # "estimado cliente"


def test_construir_matriz_features_combina_tfidf_e_ingenieradas():
    df = pd.DataFrame({
        "asunto": ["Verifique su cuenta", "Reunion de equipo"],
        "cuerpo": ["Ingrese en http://banco-fake.com ahora", "Nos vemos el jueves a las 10am"],
        "remitente": ["alertas@banco.com", "jefe@empresa.com"],
    })
    vectorizador = crear_vectorizador_tfidf()
    matriz = construir_matriz_features(df, vectorizador, ajustar=True)

    n_palabras_tfidf = len(vectorizador.get_feature_names_out())
    assert matriz.shape == (2, n_palabras_tfidf + len(NOMBRES_FEATURES))


def test_construir_matriz_features_no_reajusta_el_vectorizador_cuando_ajustar_es_false():
    df_entrenamiento = pd.DataFrame({
        "asunto": ["Verifique su cuenta"],
        "cuerpo": ["Ingrese en http://banco-fake.com ahora"],
        "remitente": ["alertas@banco.com"],
    })
    vectorizador = crear_vectorizador_tfidf()
    construir_matriz_features(df_entrenamiento, vectorizador, ajustar=True)
    vocabulario_original = set(vectorizador.get_feature_names_out())

    df_nuevo = pd.DataFrame({
        "asunto": ["Palabras nunca vistas antes zzz"],
        "cuerpo": ["Otro texto completamente distinto qqq"],
        "remitente": ["a@b.com"],
    })
    construir_matriz_features(df_nuevo, vectorizador, ajustar=False)

    assert set(vectorizador.get_feature_names_out()) == vocabulario_original


def test_explicar_clasificacion_funciona_con_modelo_calibrado():
    """
    entrenar_modelo.py envuelve el SVC lineal en CalibratedClassifierCV
    (reemplazo de SVC(probability=True), deprecado) -- ese wrapper no
    expone coef_ en su nivel superior. Este test confirma que
    explicar_clasificacion sigue funcionando con ese wrapper real, no solo
    con un SVC directo (ver _coef_del_modelo en features.py).
    """
    df = pd.DataFrame({
        "asunto": ["Verifique su cuenta", "Confirme sus datos", "Reunion de equipo", "Almuerzo de trabajo"],
        "cuerpo": [
            "Ingrese en http://banco-fake.com ahora",
            "Su contraseña debe confirmarse de inmediato",
            "Nos vemos el jueves a las 10am",
            "Reservé una mesa para el viernes",
        ],
        "remitente": ["alertas@banco-fake.com", "soporte@x.com", "jefe@empresa.com", "compañero@empresa.com"],
    })
    etiquetas = ["phishing", "phishing", "benigno", "benigno"]
    vectorizador = crear_vectorizador_tfidf()
    X = construir_matriz_features(df, vectorizador, ajustar=True)

    modelo = CalibratedClassifierCV(SVC(kernel="linear"), ensemble=False, cv=2)
    modelo.fit(X, etiquetas)

    explicacion = explicar_clasificacion(
        modelo, vectorizador,
        df.loc[0, "asunto"], df.loc[0, "cuerpo"], df.loc[0, "remitente"],
    )

    assert len(explicacion["features_ingenieradas"]) == len(NOMBRES_FEATURES)
    assert "top_palabras" in explicacion
