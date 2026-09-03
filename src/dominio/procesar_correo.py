"""
Nucleo de dominio: clasifica un Correo ya normalizado (ver
src/dominio/correo.py) con el SVM entrenado y, si es phishing, delega
marcar/mover al puerto GestorDeCorreoPort y dispara la notificacion
inyectada (o ninguna, si se pasa None). No depende de ningun adaptador
concreto -- al contrario, el adaptador (src/adaptadores/graph_api_adapter.py)
importa de aqui las constantes de marcado/notificacion compartidas, en
particular PREFIJO_ALERTA_NOTIFICACION, la unica salvaguarda contra un
bucle infinito de auto-notificacion.
"""
import logging  #Modulo para operaciones de logs
from datetime import datetime, timezone  #Para medir el tiempo medio de deteccion contra correo.hora_recepcion

import pandas as pd  #Modulo para construir el DataFrame de un unico correo, formato que espera construir_matriz_features

from src.dominio.correo import Correo
from src.dominio.features import construir_matriz_features, explicar_clasificacion
from src.puertos.gestor_correo_port import GestorDeCorreoPort

ETIQUETA_PHISHING = "phishing"  #Etiqueta que usa el SVM para la clase positiva (debe coincidir con el dataset de entrenamiento)
#Las siguientes 4 constantes son compartidas por ambos adaptadores (COM y Graph); viven aqui, no duplicadas en cada uno,
#para que el guard anti-bucle de mas abajo (que depende de PREFIJO_ALERTA_NOTIFICACION) no pueda desincronizarse entre ellos
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
    Traduce el resultado de explicar_clasificacion a una lista de senales en
    lenguaje natural, para el cuerpo del correo de notificacion (dirigido a
    alguien no tecnico). Solo se incluye una senal si su contribucion fue
    positiva (empujo hacia "phishing"). Las senales se ordenan de mayor a
    menor contribucion.

    Las frases evitan citar textualmente las palabras que busca el propio
    clasificador (_PALABRAS_URGENCIA, _PATRONES_DATOS_SENSIBLES,
    _SALUDOS_GENERICOS en features.py), para que el correo de notificacion
    no se autoclasifique como phishing.
    """
    valores = {nombre: valor for nombre, valor, _contribucion in explicacion["features_ingenieradas"]}
    contribuciones = {nombre: contribucion for nombre, _valor, contribucion in explicacion["features_ingenieradas"]}

    # Se usa .get() con valores por defecto ("sin senal") en vez de indexado
    # directo: explicacion["features_ingenieradas"] puede venir vacio (p.ej.
    # en tests que mockean explicar_clasificacion), y un indexado directo
    # lanzaria KeyError -- que procesar_correo atraparia en su try/except
    # generico, dejando de enviar la notificacion sin ningun aviso claro.
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
    Indicador "tiempo medio de deteccion": lapso entre que el correo llego
    a la bandeja de entrada (correo.hora_recepcion, provisto por el
    adaptador de origen) y que la notificacion de phishing se envio con
    exito. Retorna "N/A" si no se conoce la hora de recepcion.
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
    Clasifica un unico Correo y, si es phishing, lo marca y mueve a traves
    de `gestor` (GestorDeCorreoPort), y notifica via `enviar_notificacion`
    -- pasar None desactiva el envio de notificaciones. Ambos colaboradores
    son inyectados para que esta funcion no dependa del adaptador de origen
    concreto (hoy, Microsoft Graph). Retorna True si fue clasificado como
    phishing.
    """
    # Guard anti-bucle: si esta funcion detecta como phishing su propia
    # alerta (porque correo_destino_notificacion es el mismo buzon
    # monitoreado, "auto-notificacion"), notificaria de nuevo, generando un
    # bucle infinito. Se exige que coincidan DOS cosas -- el prefijo Y que
    # el remitente sea exactamente el destino de notificacion -- porque
    # confiar solo en el asunto (dato que cualquiera que nos envie un
    # correo controla) permitiria que un atacante evada la clasificacion
    # entera con solo escribir "[ALERTA PHISHING] ..." como asunto.
    #
    # Importante: esto NO es una garantia criptografica. El remitente
    # ("From") de un correo se puede falsificar a nivel de protocolo SMTP;
    # lo que en la practica bloquea a un atacante es la politica anti-spoof
    # del propio servidor de correo (SPF/DKIM/DMARC en Exchange/M365), no
    # este chequeo. Este segundo chequeo sube el costo del bypass (ya no
    # basta con escribir un asunto, hay que lograr que el remitente pase la
    # verificacion del servidor) sin afectar el caso legitimo, pero no lo
    # vuelve imposible.
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
    # ajustar=False: el vectorizador ya fue ajustado (fit) en entrenamiento;
    # aqui solo se reutiliza para transformar el correo nuevo a ese mismo
    # espacio vectorial -- nunca se debe re-ajustar sobre datos en produccion.
    X = construir_matriz_features(df_correo, vectorizador, ajustar=False)
    prediccion = modelo.predict(X)[0] #Clasifica el correo; [0] extrae el unico resultado (la matriz X tiene una sola fila)
    probabilidad = modelo.predict_proba(X)[0][list(modelo.classes_).index(ETIQUETA_PHISHING)] #Probabilidad especificamente de la clase "phishing"
    if prediccion == ETIQUETA_PHISHING: #Si el correo se clasifico como phishing
        # Se calcula la explicacion ANTES de marcar/mover el correo, reusando
        # el mismo texto ya usado para clasificar (mismo motivo que en el
        # comentario de logger.info mas abajo: evitar depender de un
        # "item"/estado que el marcado/movido pueda invalidar despues).
        # Se pasa X=X (ya calculado dos lineas arriba) para que
        # explicar_clasificacion no vuelva a construir la misma matriz de
        # features (TF-IDF + 7 ingenieradas) por segunda vez.
        explicacion = explicar_clasificacion(
            modelo, vectorizador, correo.asunto, correo.cuerpo, correo.remitente, X=X,
        )
        tiempo_medio_deteccion = "N/A" #Se sobreescribe solo si la notificacion se envia con exito
        if enviar_notificacion is not None: #None significa notificaciones desactivadas (ver docstring de esta funcion)
            try:
                cuerpo_notificacion = _formatear_notificacion(
                    probabilidad, correo.remitente, correo.asunto, explicacion,
                )
                enviar_notificacion(correo.item, correo.asunto, cuerpo_notificacion, correo_destino_notificacion)
                tiempo_medio_deteccion = _formatear_tiempo_medio_deteccion(correo.hora_recepcion)
            except Exception:
                # Un fallo al notificar (p.ej. Graph/Outlook caido) no debe
                # impedir que el correo igual se marque y mueva -- lo unico
                # que se pierde es la alerta, no la deteccion en si.
                logger.exception("Error al enviar la notificacion de phishing")
        gestor.marcar_como_phishing(correo.item, probabilidad)
        gestor.mover_a_carpeta(correo.item, carpeta_destino)
        # Se loguea el asunto original (no el mutado con prefijo "[PHISHING] "):
        # Correo es inmutable, a diferencia del item COM que antes se releia
        # despues de marcar/mover.
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
    Recorre los correos no leidos (via gestor.obtener_no_leidos) y clasifica
    cada uno con procesar_correo. Retorna la cantidad marcada como phishing.
    `a_correo` es el conversor especifico del adaptador (_mensaje_a_correo
    para Graph) -- se inyecta en vez de asumir un formato fijo, porque
    gestor.obtener_no_leidos() retorna items "crudos" (dict de Graph), no
    Correo directamente. Cada item se procesa de forma aislada: si uno
    falla (p.ej. un mensaje con un campo inesperado que lanza un error al
    leerlo), se loguea y se continua con los demas -- sin esto, una
    excepcion en un solo correo abortaria el resto del lote, dejando SIN
    clasificar a todos los demas correos no leidos de esa notificacion.
    """
    items_no_leidos = gestor.obtener_no_leidos() #Lista de items crudos (formato depende del adaptador inyectado en `gestor`)
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
