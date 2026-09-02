import pandas as pd

from src.dominio.adaptador_spaphish import adaptar_spaphish


def test_asunto_se_copia_tal_cual():
    df = pd.DataFrame({
        "subject": ["Aviso importante"],
        "body": ["Cuerpo"],
        "Label": [1],
        "urls": [None],
    })
    resultado = adaptar_spaphish(df)
    assert resultado.loc[0, "asunto"] == "Aviso importante"


def test_mapea_label_1_a_phishing_y_0_a_benigno():
    df = pd.DataFrame({
        "subject": ["a", "b"],
        "body": ["x", "y"],
        "Label": [1, 0],
        "urls": [None, None],
    })
    resultado = adaptar_spaphish(df)
    assert list(resultado["etiqueta"]) == ["phishing", "benigno"]


def test_remitente_queda_vacio_por_no_estar_disponible_en_el_origen():
    df = pd.DataFrame({
        "subject": ["a"],
        "body": ["x"],
        "Label": [1],
        "urls": [None],
    })
    resultado = adaptar_spaphish(df)
    assert resultado.loc[0, "remitente"] == ""


def test_cuerpo_queda_igual_si_no_hay_urls():
    df = pd.DataFrame({
        "subject": ["a"],
        "body": ["Sin enlaces aqui"],
        "Label": [0],
        "urls": [None],
    })
    resultado = adaptar_spaphish(df)
    assert resultado.loc[0, "cuerpo"] == "Sin enlaces aqui"


def test_reinserta_una_url_en_el_cuerpo():
    df = pd.DataFrame({
        "subject": ["a"],
        "body": ["Se ha producido un error en la entrega"],
        "Label": [1],
        "urls": ["[http://everensec.com/tnWC.asp?x=1]"],
    })
    resultado = adaptar_spaphish(df)
    assert "http://everensec.com/tnWC.asp?x=1" in resultado.loc[0, "cuerpo"]


def test_reinserta_varias_urls_de_una_lista():
    df = pd.DataFrame({
        "subject": ["a"],
        "body": ["Cuerpo"],
        "Label": [1],
        "urls": ["[http://a.com],[https://b.com]"],
    })
    resultado = adaptar_spaphish(df)
    assert "http://a.com" in resultado.loc[0, "cuerpo"]
    assert "https://b.com" in resultado.loc[0, "cuerpo"]


def test_reemplaza_saltos_de_linea_internos_por_espacio():
    # Los saltos de linea internos (\n) mezclados con el \r\n del propio CSV
    # confunden a Excel al mostrar el archivo (una linea en blanco dentro
    # del cuerpo se ve como una fila vacia). No afecta a TF-IDF ni al
    # regex de URLs (\s ya trata \n igual que un espacio), pero se
    # normaliza para que el CSV se pueda inspeccionar bien en Excel.
    df = pd.DataFrame({
        "subject": ["Asunto\ncon salto"],
        "body": ["Primera linea\n\nSegunda linea"],
        "Label": [1],
        "urls": [None],
    })
    resultado = adaptar_spaphish(df)
    assert "\n" not in resultado.loc[0, "asunto"]
    assert "\n" not in resultado.loc[0, "cuerpo"]
    assert resultado.loc[0, "asunto"] == "Asunto con salto"
    assert "Primera linea" in resultado.loc[0, "cuerpo"]
    assert "Segunda linea" in resultado.loc[0, "cuerpo"]


def test_columnas_de_salida_son_las_del_esquema_del_proyecto():
    df = pd.DataFrame({
        "subject": ["a"],
        "body": ["x"],
        "Label": [1],
        "urls": [None],
    })
    resultado = adaptar_spaphish(df)
    assert list(resultado.columns) == ["asunto", "cuerpo", "remitente", "etiqueta"]
