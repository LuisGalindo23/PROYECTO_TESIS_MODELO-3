"""
Tests de caracterizacion para src/dominio/procesar_correo.py, usando dobles
de prueba (FakeModelo, FakeVectorizador, FakeGestor) para probar la logica
de clasificacion/marcado/notificacion sin depender de un SVM real
entrenado. Es el archivo de mas riesgo del dominio: decide que se marca,
mueve y notifica, e incluye el guard anti-bucle de auto-notificacion.
"""
import logging

import numpy as np
from scipy.sparse import csr_matrix

from src.dominio.correo import Correo
from src.dominio.features import NOMBRES_FEATURES
from src.dominio.procesar_correo import (
    procesar_correo,
    ejecutar_ciclo_de_escaneo,
    PREFIJO_ALERTA_NOTIFICACION,
)

LOGGER = logging.getLogger("test_procesar_correo")


class FakeVectorizador:
    """Vectorizador de mentira: mismo contrato (transform/get_feature_names_out) sin TF-IDF real."""

    def __init__(self, vocabulario=("banco", "verifique")):
        self._vocabulario = list(vocabulario)

    def transform(self, textos):
        return csr_matrix((len(textos), len(self._vocabulario)))

    def get_feature_names_out(self):
        return np.array(self._vocabulario)


class FakeModelo:
    """
    Modelo de mentira: las predicciones/probabilidades se consumen en orden
    (una entrada por cada correo procesado), para poder controlar
    exactamente que devuelve el "SVM" en cada test sin entrenar uno real.
    """

    def __init__(self, predicciones, probabilidades_phishing, n_columnas_tfidf=2):
        self.classes_ = ["benigno", "phishing"]
        self._predicciones = list(predicciones)
        self._probabilidades = list(probabilidades_phishing)
        self.coef_ = csr_matrix(np.zeros((1, n_columnas_tfidf + len(NOMBRES_FEATURES))))

    def predict(self, X):
        return [self._predicciones.pop(0)]

    def predict_proba(self, X):
        prob_phishing = self._probabilidades.pop(0)
        return [[1 - prob_phishing, prob_phishing]]


class FakeGestor:
    def __init__(self, items_no_leidos=None):
        self.marcados = []
        self.movidos = []
        self._items_no_leidos = items_no_leidos or []

    def obtener_no_leidos(self):
        return self._items_no_leidos

    def marcar_como_phishing(self, item, probabilidad):
        self.marcados.append((item, probabilidad))

    def mover_a_carpeta(self, item, carpeta_destino):
        self.movidos.append((item, carpeta_destino))


def test_correo_benigno_no_se_marca_ni_se_mueve_ni_se_notifica():
    correo = Correo(item="item-1", asunto="Reunion de equipo", cuerpo="Nos vemos el jueves", remitente="jefe@empresa.com")
    modelo = FakeModelo(predicciones=["benigno"], probabilidades_phishing=[0.05])
    llamadas = []

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), FakeGestor(), "carpeta-destino", LOGGER,
        lambda *a: llamadas.append(a), "destino@empresa.com",
    )

    assert resultado is False
    assert llamadas == []


def test_correo_phishing_se_marca_y_mueve():
    correo = Correo(item="item-1", asunto="Verifique su cuenta", cuerpo="http://banco-fake.com", remitente="alertas@banco.com")
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        None, "destino@empresa.com",
    )

    assert resultado is True
    assert gestor.marcados == [("item-1", 0.9)]
    assert gestor.movidos == [("item-1", "carpeta-destino")]


def test_correo_phishing_dispara_notificacion_con_los_datos_correctos():
    correo = Correo(item="item-1", asunto="Verifique su cuenta", cuerpo="http://banco-fake.com", remitente="alertas@banco.com")
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    llamadas = []

    procesar_correo(
        correo, modelo, FakeVectorizador(), FakeGestor(), "carpeta-destino", LOGGER,
        lambda *a: llamadas.append(a), "destino@empresa.com",
    )

    assert len(llamadas) == 1
    item, asunto, cuerpo_notificacion, destino = llamadas[0]
    assert item == "item-1"
    assert asunto == "Verifique su cuenta"
    assert destino == "destino@empresa.com"
    assert "90.00%" in cuerpo_notificacion


def test_correo_phishing_no_notifica_si_enviar_notificacion_es_none():
    correo = Correo(item="item-1", asunto="Verifique su cuenta", cuerpo="texto", remitente="a@b.com")
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        None, "destino@empresa.com",
    )

    assert resultado is True
    assert gestor.marcados
    assert gestor.movidos


def test_fallo_al_notificar_no_impide_marcar_y_mover():
    correo = Correo(item="item-1", asunto="Verifique su cuenta", cuerpo="texto", remitente="a@b.com")
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()

    def notificador_que_falla(*_args):
        raise RuntimeError("fallo simulado de notificacion")

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        notificador_que_falla, "destino@empresa.com",
    )

    assert resultado is True
    assert gestor.marcados
    assert gestor.movidos


def test_guard_antibucle_bloquea_autonotificacion():
    correo = Correo(
        item="item-1",
        asunto=PREFIJO_ALERTA_NOTIFICACION + "Se detecto phishing",
        cuerpo="texto",
        remitente="destino@empresa.com",
    )
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()
    llamadas = []

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        lambda *a: llamadas.append(a), "destino@empresa.com",
    )

    assert resultado is False
    assert gestor.marcados == []
    assert gestor.movidos == []
    assert llamadas == []


def test_guard_antibucle_no_bloquea_si_el_remitente_es_distinto_al_destino():
    correo = Correo(
        item="item-1",
        asunto=PREFIJO_ALERTA_NOTIFICACION + "Se detecto phishing",
        cuerpo="texto",
        remitente="atacante@otro.com",
    )
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        None, "destino@empresa.com",
    )

    assert resultado is True
    assert gestor.marcados


def test_guard_antibucle_no_bloquea_si_el_asunto_no_tiene_el_prefijo():
    correo = Correo(item="item-1", asunto="Aviso normal", cuerpo="texto", remitente="destino@empresa.com")
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])
    gestor = FakeGestor()

    resultado = procesar_correo(
        correo, modelo, FakeVectorizador(), gestor, "carpeta-destino", LOGGER,
        None, "destino@empresa.com",
    )

    assert resultado is True
    assert gestor.marcados


def test_ejecutar_ciclo_cuenta_los_correos_marcados_como_phishing():
    correo_phishing = Correo(item="p", asunto="Verifique", cuerpo="texto", remitente="a@b.com")
    correo_benigno = Correo(item="b", asunto="Reunion", cuerpo="texto", remitente="a@b.com")
    crudos_a_correo = {"raw-p": correo_phishing, "raw-b": correo_benigno}
    gestor = FakeGestor(items_no_leidos=["raw-p", "raw-b"])
    modelo = FakeModelo(predicciones=["phishing", "benigno"], probabilidades_phishing=[0.9, 0.1])

    cantidad = ejecutar_ciclo_de_escaneo(
        gestor, modelo, FakeVectorizador(), "carpeta-destino", LOGGER,
        crudos_a_correo.get, None, "destino@empresa.com",
    )

    assert cantidad == 1
    assert gestor.marcados == [("p", 0.9)]


def test_ejecutar_ciclo_continua_si_un_correo_falla_al_convertirse():
    correo_ok = Correo(item="ok", asunto="Verifique", cuerpo="texto", remitente="a@b.com")

    def a_correo(item_crudo):
        if item_crudo == "malo":
            raise RuntimeError("item corrupto")
        return correo_ok

    gestor = FakeGestor(items_no_leidos=["malo", "bueno"])
    modelo = FakeModelo(predicciones=["phishing"], probabilidades_phishing=[0.9])

    cantidad = ejecutar_ciclo_de_escaneo(
        gestor, modelo, FakeVectorizador(), "carpeta-destino", LOGGER,
        a_correo, None, "destino@empresa.com",
    )

    assert cantidad == 1
    assert gestor.marcados == [("ok", 0.9)]
