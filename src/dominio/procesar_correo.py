"""
Nucleo de dominio: clasifica un Correo ya normalizado (ver src/dominio/correo.py) con el SVM entrenado y, si es phishing, delega
marcar/mover al puerto GestorDeCorreoPort y dispara la notificacion (o ninguna, si se pasa None).
"""
import logging  #Modulo para operaciones de logs
from datetime import datetime, timezone  #Para medir el tiempo medio de deteccion contra hora de recepcion

import pandas as pd  #Modulo de lectura/manipulacion del CSV como tabla (DataFrame)

from src.dominio.correo import Correo
from src.dominio.features import construir_matriz_features, explicar_clasificacion
from src.puertos.gestor_correo_port import GestorDeCorreoPort

ETIQUETA_PHISHING = "phishing"
NOMBRE_CARPETA_PHISHING = "Phishing Detectado"
CATEGORIA_PHISHING = "Posible Phishing"
PREFIJO_ASUNTO = "[PHISHING] "
PREFIJO_ALERTA_NOTIFICACION = "[ALERTA PHISHING] "

_UMBRAL_MAYUSCULAS_ALTO = 0.3 #Umbral que decide si la proporción de mayúsculas del asunto es suficientemente alta como para 
#considerarse una señal de alarma
_EPSILON_CONTRIBUCION = 0.005  # Contribuciones por debajo de esto se consideran ruido (no contribuye debido a que el formato número 
# redondea a 2 decimales considerando el umbral)

def _formatear_analisis_natural(explicacion: dict) -> str:
    """
    Traduce el resultado de explicar_clasificacion a una lista de senales en lenguaje natural, para el cuerpo del correo de notificacion.
    Solo se incluye una senal si su contribucion fue positiva (empujo hacia "phishing").
    """
    valores = {nombre: valor for nombre, valor, _contribucion in explicacion["features_ingenieradas"]}
    contribuciones = {nombre: contribucion for nombre, _valor, contribucion in explicacion["features_ingenieradas"]}

    senales_con_peso = []
    if valores.get("urls", 0) > 0 and contribuciones.get("urls", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((contribuciones["urls"], "Contiene enlaces (URLs) en el mensaje."))
    if valores.get("url_sospechosa", False) and contribuciones.get("url_sospechosa", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((
            contribuciones["url_sospechosa"],
            "Alguno de los enlaces usa una direccion IP o un acortador conocido "
            "(bit.ly, tinyurl, etc.), una tecnica comun para ocultar el destino real.",
        ))
    if valores.get("urgencia", 0) > 0 and contribuciones.get("urgencia", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((
            contribuciones["urgencia"],
            "Usa lenguaje que transmite urgencia y presiona a responder sin demora.",
        ))
    if valores.get("mayusculas", 0) > _UMBRAL_MAYUSCULAS_ALTO and contribuciones.get("mayusculas", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((contribuciones["mayusculas"], "El asunto usa una proporcion alta de mayusculas."))
    if valores.get("datos_sensibles", False) and contribuciones.get("datos_sensibles", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((
            contribuciones["datos_sensibles"],
            "Solicita informacion privada o financiera de forma directa, o pide actuar "
            "sin darte tiempo para verificar por otro medio.",
        ))
    if valores.get("saludo_generico", False) and contribuciones.get("saludo_generico", 0) > _EPSILON_CONTRIBUCION:
        senales_con_peso.append((
            contribuciones["saludo_generico"],
            "Usa un saludo generico en vez de dirigirse a ti por tu nombre.",
        ))

    senales_con_peso.sort(key=lambda par: par[0], reverse=True)
    senales = [texto for _contribucion, texto in senales_con_peso]

    palabras_relevantes = [
        palabra
        for palabra, _valor_tfidf, contribucion in explicacion["top_palabras"]
        if contribucion > _EPSILON_CONTRIBUCION
    ]

    if senales:
        cuerpo_senales = "Este correo fue clasificado como phishing porque se detectaron las siguientes senales:\n"
        cuerpo_senales += "\n".join(f"- {senal}" for senal in senales)
    else:
        cuerpo_senales = (
            "Este correo fue clasificado como phishing por similitud general de su texto "
            "con correos de phishing conocidos, sin una senal especifica aislada."
        )

    if palabras_relevantes:
        palabras_texto = ", ".join(f'"{palabra}"' for palabra in palabras_relevantes)
        cuerpo_senales += f"\n\nPalabras clave del texto que mas influyeron: {palabras_texto}."

    return cuerpo_senales


def _formatear_notificacion(probabilidad: float, remitente: str, asunto: str, explicacion: dict) -> str:
    """
    Arma el cuerpo del correo de notificacion: un encabezado con remitente,
    asunto y probabilidad, seguido del desglose de _formatear_analisis_natural.
    """
    encabezado = (
        f"Se detecto un posible correo de phishing.\n\n"
        f"Remitente: {remitente}\n"
        f"Asunto: {asunto}\n"
        f"Probabilidad de phishing: {probabilidad:.2%}\n\n"
        f"Analisis del modelo:"
    )
    return f"{encabezado}\n{_formatear_analisis_natural(explicacion)}"


def _formatear_tiempo_medio_deteccion(hora_recepcion) -> str:
    """
    Indicador "tiempo medio de deteccion": lapso entre que el correo llego a la bandeja de entrada (correo.hora_recepcion)
    y que la notificacion de phishing se envió con exito. Retorna "N/A" si no se conoce la hora de recepcion.
    """
    if hora_recepcion is None:
        return "N/A"
    return f"{(datetime.now(timezone.utc) - hora_recepcion).total_seconds():.2f}s"


def procesar_correo(
    correo: Correo,
    modelo,
    vectorizador,
    gestor: GestorDeCorreoPort,
    carpeta_destino,
    logger: logging.Logger,
    enviar_notificacion,
    correo_destino_notificacion: str,
) -> bool:
    """
    Clasifica un unico Correo y, si es phishing, lo marca y mueve a traves de `gestor` (GestorDeCorreoPort), y notifica via `enviar_notificacion`
    """
    
    destino_normalizado = (correo_destino_notificacion or "").strip().lower()
    if (
        destino_normalizado
        and correo.asunto.startswith(PREFIJO_ALERTA_NOTIFICACION)
        and (correo.remitente or "").strip().lower() == destino_normalizado
    ):
        return False
    df_correo = pd.DataFrame([{ #Formato de DataFrame de un unico correo, el que espera construir_matriz_features
        "asunto": correo.asunto,
        "cuerpo": correo.cuerpo,
        "remitente": correo.remitente,
    }])

    # Se reutiliza para transformar el correo nuevo a ese mismo espacio vectorial.
    X = construir_matriz_features(df_correo, vectorizador, ajustar=False)
    prediccion = modelo.predict(X)[0] #Clasifica el correo; [0] extrae el unico resultado (la matriz X tiene una sola fila)
    probabilidad = modelo.predict_proba(X)[0][list(modelo.classes_).index(ETIQUETA_PHISHING)] #Probabilidad especificamente de la clase "phishing"
    if prediccion == ETIQUETA_PHISHING:
        # Si el correo se clasifico como phishing, se calcula la explicacion ANTES de marcar/mover el correo, reusando el mismo texto, ya usado para clasificar.
     
        explicacion = explicar_clasificacion(
            modelo, vectorizador, correo.asunto, correo.cuerpo, correo.remitente, X=X,
        )
        tiempo_medio_deteccion = "N/A" #Se sobreescribe solo si la notificacion se envia con exito
        if enviar_notificacion is not None: #None significa notificaciones desactivadas.
            try:
                cuerpo_notificacion = _formatear_notificacion(
                    probabilidad, correo.remitente, correo.asunto, explicacion,
                )
                enviar_notificacion(correo.item, correo.asunto, cuerpo_notificacion, correo_destino_notificacion)
                tiempo_medio_deteccion = _formatear_tiempo_medio_deteccion(correo.hora_recepcion)
            except Exception:
                # Un fallo al notificar no debe impedir que el correo igual se marque y mueva, lo unico que se pierde es la alerta,
                # no la deteccion en si.
                logger.exception("Error al enviar la notificacion de phishing")
        gestor.marcar_como_phishing(correo.item, probabilidad)
        gestor.mover_a_carpeta(correo.item, carpeta_destino)
        # Se registra en log el asunto original.
        logger.info(
            "PHISHING detectado (prob=%.2f) - remitente=%s - asunto=%s - tiempo_medio_deteccion=%s",
            probabilidad, correo.remitente, correo.asunto, tiempo_medio_deteccion,
        )
        return True

    logger.info( #Se dispara igual si el correo es benigno, para tener un registro completo de todo lo que se escaneo
        "Benigno (prob_phishing=%.2f) - remitente=%s - asunto=%s",
        probabilidad, correo.remitente, correo.asunto,
    )
    return False

def ejecutar_ciclo_de_escaneo(
    gestor: GestorDeCorreoPort,
    modelo,
    vectorizador,
    carpeta_destino,
    logger: logging.Logger,
    a_correo,
    enviar_notificacion,
    correo_destino_notificacion: str,
) -> int:
    """
    Recorre los correos no leidos (via gestor.obtener_no_leidos) y clasifica cada uno con procesar_correo. Retorna la cantidad marcada como phishing.
    `a_correo` es el conversor especifico del adaptador, se inyecta en vez de asumir un formato fijo, porque gestor.obtener_no_leidos() retorna items "crudos" (dict de Graph),
    no Correo directamente. Cada item se procesa de forma aislada.
    """

    items_no_leidos = gestor.obtener_no_leidos() #Lista de items crudos.
    cantidad_marcados = 0 #Contador de cuantos se detectaron como phishing en este ciclo
    for item in items_no_leidos:
        try:
            correo = a_correo(item) #Normaliza el item crudo a la representacion neutral Correo
            if procesar_correo(
                correo, modelo, vectorizador, gestor, carpeta_destino, logger,
                enviar_notificacion, correo_destino_notificacion,
            ):
                cantidad_marcados += 1 #procesar_correo retorna True solo si fue clasificado como phishing
        except Exception:
            logger.exception("Error al procesar un correo del ciclo de escaneo; se continua con los restantes")
    return cantidad_marcados
