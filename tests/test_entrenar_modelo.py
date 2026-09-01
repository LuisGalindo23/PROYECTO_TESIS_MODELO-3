"""
Test de humo para src/dominio/entrenar_modelo.py: confirma que el flujo
completo (leer CSV -> limpiar -> vectorizar -> entrenar -> guardar) corre
sin errores y produce artefactos usables, usando rutas temporales para no
tocar modelo/svm_phishing.pkl real.
"""
import joblib
import pandas as pd

from src.dominio.entrenar_modelo import entrenar_y_guardar


def test_entrenar_y_guardar_produce_modelo_y_vectorizador_usables(tmp_path):
    df = pd.DataFrame({
        "asunto": [
            "Verifique su cuenta urgente",
            "Su cuenta sera bloqueada",
            "Confirme sus datos ahora",
            "Alerta de seguridad en su cuenta",
            "Ultima oportunidad para verificar",
            "Reunion de equipo el jueves",
            "Almuerzo con el cliente manana",
            "Agenda de la semana",
            "Recordatorio de capacitacion",
            "Actualizacion del proyecto",
        ],
        "cuerpo": [
            "Estimado cliente, ingrese en http://banco-fake.com para verificar sus datos",
            "Confirme su contraseña ahora mismo o perdera el acceso",
            "Haga clic aqui e ingrese su numero de tarjeta para confirmar",
            "Su cuenta sera cerrada, verifique ahora en http://alerta-banco.com",
            "Estimado cliente, actue ahora o perdera el acceso a su cuenta",
            "Nos vemos a las 10am en la sala de conferencias",
            "Reservé una mesa para las 13:00, confirmame si puedes ir",
            "Adjunto la agenda de reuniones de esta semana",
            "La capacitacion es el viernes a las 9am, no falten",
            "El proyecto avanza segun lo planeado, hablamos manana",
        ],
        "remitente": [
            "alertas@banco-fake.com", "seguridad@bancx.com", "soporte@servicio-x.com",
            "alertas@alerta-banco.com", "cliente@promo-gratis.com",
            "jefe@empresa.com", "compañero@empresa.com", "rrhh@empresa.com",
            "capacitacion@empresa.com", "lider@empresa.com",
        ],
        "etiqueta": [
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "benigno", "benigno", "benigno", "benigno", "benigno",
        ],
    })
    ruta_csv = tmp_path / "correos.csv"
    df.to_csv(ruta_csv, index=False)
    ruta_modelo = tmp_path / "modelo.pkl"
    ruta_vectorizador = tmp_path / "vectorizador.pkl"

    resultado = entrenar_y_guardar(str(ruta_csv), str(ruta_modelo), str(ruta_vectorizador))

    assert 0.0 <= resultado["f1_promedio_kfold"] <= 1.0
    assert 0.0 <= resultado["f1_desviacion_kfold"] <= 1.0
    assert ruta_modelo.exists()
    assert ruta_vectorizador.exists()

    modelo = joblib.load(ruta_modelo)
    vectorizador = joblib.load(ruta_vectorizador)
    assert list(modelo.classes_) == ["benigno", "phishing"]
    assert vectorizador.get_feature_names_out() is not None
