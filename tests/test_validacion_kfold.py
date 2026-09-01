import pandas as pd

from src.dominio.validacion_kfold import validar_con_kfold


def test_devuelve_f1_promedio_y_desviacion_como_numeros_entre_0_y_1():
    df = pd.DataFrame({
        "asunto": [f"Verifique su cuenta {i}" for i in range(5)] + [f"Reunion de equipo {i}" for i in range(5)],
        "cuerpo": [f"Ingrese en http://banco-fake-{i}.com y confirme su contraseña" for i in range(5)]
                  + [f"Nos vemos el jueves {i} a las 10am en la oficina" for i in range(5)],
        "remitente": [f"alertas{i}@banco-fake.com" for i in range(5)] + [f"jefe{i}@empresa.com" for i in range(5)],
        "etiqueta": ["phishing"] * 5 + ["benigno"] * 5,
    })

    resultado = validar_con_kfold(df, n_splits=2)

    assert 0.0 <= resultado["f1_promedio"] <= 1.0
    assert 0.0 <= resultado["f1_desviacion"] <= 1.0


def test_f1_promedio_es_alto_cuando_las_clases_son_facilmente_separables():
    df = pd.DataFrame({
        "asunto": [f"Verifique su cuenta {i}" for i in range(10)] + [f"Reunion de equipo {i}" for i in range(10)],
        "cuerpo": [f"Ingrese en http://banco-fake-{i}.com y confirme su contraseña ahora" for i in range(10)]
                  + [f"Nos vemos el jueves {i} a las 10am en la oficina del piso 3" for i in range(10)],
        "remitente": [f"alertas{i}@banco-fake.com" for i in range(10)] + [f"jefe{i}@empresa.com" for i in range(10)],
        "etiqueta": ["phishing"] * 10 + ["benigno"] * 10,
    })

    resultado = validar_con_kfold(df, n_splits=5)

    assert resultado["f1_promedio"] > 0.9


def test_respeta_la_cantidad_de_folds_configurada():
    df = pd.DataFrame({
        "asunto": [f"Verifique su cuenta {i}" for i in range(6)] + [f"Reunion de equipo {i}" for i in range(6)],
        "cuerpo": [f"Ingrese en http://banco-fake-{i}.com" for i in range(6)]
                  + [f"Nos vemos el jueves {i}" for i in range(6)],
        "remitente": [f"alertas{i}@banco-fake.com" for i in range(6)] + [f"jefe{i}@empresa.com" for i in range(6)],
        "etiqueta": ["phishing"] * 6 + ["benigno"] * 6,
    })

    resultado = validar_con_kfold(df, n_splits=3)

    assert len(resultado["f1_por_fold"]) == 3
