from src.dominio.lectura_csv import leer_csv_dataset


def test_lee_un_csv_utf8_con_coma(tmp_path):
    ruta = tmp_path / "dataset.csv"
    ruta.write_text("asunto,cuerpo\nHola,Texto con ñ\n", encoding="utf-8")

    df = leer_csv_dataset(str(ruta))

    assert list(df.columns) == ["asunto", "cuerpo"]
    assert df.loc[0, "cuerpo"] == "Texto con ñ"


def test_lee_un_csv_cp1252_con_punto_y_coma_como_el_que_exporta_excel(tmp_path):
    # Mismo formato que Excel guarda por defecto en configuracion regional
    # en espanol: cp1252 (no UTF-8) con ";" como separador.
    ruta = tmp_path / "dataset.csv"
    ruta.write_bytes("asunto;cuerpo\nHola;Texto con ñ\n".encode("cp1252"))

    df = leer_csv_dataset(str(ruta))

    assert list(df.columns) == ["asunto", "cuerpo"]
    assert df.loc[0, "cuerpo"] == "Texto con ñ"
