import pandas as pd
import pytest

from src.dominio.limpieza_dataset import limpiar_dataset


def test_lanza_error_si_falta_una_columna_requerida():
    df = pd.DataFrame({"asunto": ["a"], "cuerpo": ["b"], "remitente": ["c@x.com"]})  # falta 'etiqueta'
    with pytest.raises(ValueError, match="etiqueta"):
        limpiar_dataset(df)


def test_rellena_nulos_de_texto_con_string_vacio():
    df = pd.DataFrame({
        "asunto": [None],
        "cuerpo": [None],
        "remitente": [None],
        "etiqueta": ["benigno"],
    })
    resultado = limpiar_dataset(df)
    assert resultado.loc[0, "asunto"] == ""
    assert resultado.loc[0, "cuerpo"] == ""
    assert resultado.loc[0, "remitente"] == ""


def test_recorta_espacios_en_blanco_al_inicio_y_final():
    df = pd.DataFrame({
        "asunto": ["  Hola  "],
        "cuerpo": ["  Cuerpo  "],
        "remitente": ["  a@b.com  "],
        "etiqueta": ["benigno"],
    })
    resultado = limpiar_dataset(df)
    assert resultado.loc[0, "asunto"] == "Hola"
    assert resultado.loc[0, "cuerpo"] == "Cuerpo"
    assert resultado.loc[0, "remitente"] == "a@b.com"


def test_normaliza_etiqueta_con_espacios_y_mayusculas():
    df = pd.DataFrame({
        "asunto": ["a"], "cuerpo": ["b"], "remitente": ["c@x.com"],
        "etiqueta": [" Phishing "],
    })
    resultado = limpiar_dataset(df)
    assert resultado.loc[0, "etiqueta"] == "phishing"


def test_lanza_error_si_etiqueta_no_es_benigno_ni_phishing():
    df = pd.DataFrame({
        "asunto": ["a"], "cuerpo": ["b"], "remitente": ["c@x.com"],
        "etiqueta": ["spam"],
    })
    with pytest.raises(ValueError, match="etiqueta"):
        limpiar_dataset(df)


def test_elimina_filas_duplicadas_exactas_conservando_la_primera():
    df = pd.DataFrame({
        "asunto": ["Hola", "Hola"],
        "cuerpo": ["Texto", "Texto"],
        "remitente": ["a@b.com", "a@b.com"],
        "etiqueta": ["benigno", "benigno"],
    })
    resultado = limpiar_dataset(df)
    assert len(resultado) == 1


def test_conserva_el_indice_original_de_las_filas_no_duplicadas():
    df = pd.DataFrame({
        "asunto": ["Hola", "Hola", "Chau"],
        "cuerpo": ["Texto", "Texto", "Otro"],
        "remitente": ["a@b.com", "a@b.com", "c@d.com"],
        "etiqueta": ["benigno", "benigno", "phishing"],
    })
    resultado = limpiar_dataset(df)
    assert list(resultado.index) == [0, 2]


def test_lanza_error_si_etiqueta_es_nula():
    df = pd.DataFrame({
        "asunto": ["a"], "cuerpo": ["b"], "remitente": ["c@x.com"],
        "etiqueta": [None],
    })
    with pytest.raises(ValueError, match="etiqueta"):
        limpiar_dataset(df)
