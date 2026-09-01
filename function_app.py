"""
Azure Functions: recibe notificaciones de Microsoft Graph (correo nuevo en
el buzon monitoreado) y renueva la suscripcion de Graph antes de que
expire. Los triggers (@app.route, @app.timer_trigger) son envoltorios
delgados sobre manejar_notificacion/ejecutar_renovacion, para poder
testear la logica sin el runtime de Functions -- mismo patron que main()
en src/monitor_outlook.py.
"""
import logging  #Modulo para operaciones de logs, integrado con Application Insights por el runtime de Functions
import os  #Para leer variables de entorno (App Settings de la Function App) y construir rutas de archivo

import azure.functions as func  #SDK de Azure Functions: define los tipos HttpRequest/HttpResponse/TimerRequest y los decoradores @app.*
import joblib  #Para deserializar el modelo SVM y el vectorizador TF-IDF guardados en disco
import requests  #Se usa aqui solo para distinguir requests.HTTPError (404 real) de otros errores en ejecutar_renovacion

from src.adaptadores.graph_api_adapter import (
    GestorDeCorreoGraph,
    _mensaje_a_correo,
    crear_suscripcion,
    enviar_notificacion_phishing,
    obtener_token,
    renovar_suscripcion,
)
from src.dominio.procesar_correo import procesar_correo  #La misma logica de clasificacion que usa el camino de Outlook COM

app = func.FunctionApp()  #Punto de registro de las Functions de este archivo (recibir_notificacion, renovar_suscripcion_trigger)

_MODELO = None  #Cache a nivel de modulo: el mismo worker de Functions procesa multiples invocaciones,
_VECTORIZADOR = None  #asi que cargar el modelo una sola vez (no en cada request) ahorra tiempo de arranque


def _cargar_modelo():
    """
    Carga (con memoizacion) el modelo SVM y el vectorizador TF-IDF desde
    disco. Las rutas se resuelven relativas al directorio de este mismo
    modulo (no al cwd del worker de Functions, que no esta garantizado que
    sea /home/site/wwwroot -- es la guia oficial de Microsoft para Python
    en Azure Functions).
    """
    global _MODELO, _VECTORIZADOR
    if _MODELO is None:
        directorio_base = os.path.dirname(os.path.abspath(__file__))
        ruta_modelo = os.path.join(directorio_base, os.environ["RUTA_MODELO"])
        ruta_vectorizador = os.path.join(directorio_base, os.environ["RUTA_VECTORIZADOR"])
        # Se cargan ambos en variables locales antes de asignar a los
        # globales: si joblib.load del vectorizador falla despues de que el
        # modelo ya cargo bien, _MODELO debe quedar en None para que el
        # proximo intento reintente la carga completa, en vez de quedar a
        # medio poblar (lo que haria fallar cada invocacion posterior en
        # este worker con _VECTORIZADOR=None).
        modelo = joblib.load(ruta_modelo)
        vectorizador = joblib.load(ruta_vectorizador)
        _MODELO, _VECTORIZADOR = modelo, vectorizador
    return _MODELO, _VECTORIZADOR


def _notificaciones_del_cuerpo(cuerpo) -> list:
    """
    Extrae la lista de notificaciones de un cuerpo de webhook de Graph,
    tolerando una forma inesperada -- la ruta es AuthLevel.ANONYMOUS
    (requisito del handshake de Graph), asi que cualquier llamante no
    autenticado puede mandar un JSON valido pero con forma distinta a la
    esperada (p.ej. "value" no siendo una lista, o el cuerpo entero no
    siendo un dict). En ese caso se trata como si no hubiera notificaciones
    en vez de lanzar una excepcion.
    """
    if not isinstance(cuerpo, dict):
        return []
    valor = cuerpo.get("value", [])
    return valor if isinstance(valor, list) else []


def _hay_notificacion_con_client_state_valido(cuerpo, client_state_esperado: str) -> bool:
    """
    Chequeo barato (sin red ni carga de modelo) para decidir si vale la
    pena hacer el setup costoso (obtener_token, _cargar_modelo). Si
    ninguna entrada del batch pasaria la validacion de clientState, no hay
    nada que procesar y no tiene sentido pagar ese costo -- relevante
    porque la ruta es AuthLevel.ANONYMOUS (requisito del handshake de
    Graph) y cualquier llamante no autenticado podria, si no fuera por
    este chequeo, forzar una adquisicion de token + carga de modelo por
    request.
    """
    return any(
        isinstance(notificacion, dict) and notificacion.get("clientState") == client_state_esperado
        for notificacion in _notificaciones_del_cuerpo(cuerpo)
    )


def manejar_notificacion(
    cuerpo,
    client_state_esperado: str,
    gestor,
    modelo,
    vectorizador,
    logger: logging.Logger,
    enviar_notificacion,
    correo_destino_notificacion: str,
):
    """
    Logica pura para procesar el batch de notificaciones reales de Graph
    (el handshake de validacion ya se resolvio antes, en el wrapper del
    trigger). Cada entrada se procesa de forma aislada: si una falla (p.ej.
    un reintento de Graph sobre una notificacion ya procesada, cuyo
    mensaje_id ya no existe tras el /move), se loguea y se continua con las
    demas -- una entrada fallida no debe abortar el resto del batch ni
    hacer que la Function devuelva 500 (lo que llevaria a Graph a
    reintentar indefinidamente y, tras fallos repetidos, a eliminar la
    suscripcion). Retorna (status_code, body).
    """
    notificaciones = _notificaciones_del_cuerpo(cuerpo)
    notificaciones_validas = []
    for notificacion in notificaciones:
        if isinstance(notificacion, dict) and notificacion.get("clientState") == client_state_esperado:
            notificaciones_validas.append(notificacion)
        else:
            # Se loguea cada entrada invalida individualmente (no solo
            # cuando el lote completo es invalido), para no perder la
            # visibilidad de un lote mixto -- el caso mas relevante a
            # loguear es justamente un llamante no autenticado mezclando
            # entradas falsas con una notificacion real.
            logger.warning("Notificacion con clientState invalido o con formato inesperado, se ignora")

    if not notificaciones_validas:
        return 202, "OK"

    # La carpeta destino se resuelve una sola vez por lote (no por cada
    # notificacion): Graph puede agrupar varias notificaciones en un mismo
    # webhook, y repetir la llamada por cada una desperdicia latencia y
    # expone a throttling durante una rafaga de correo.
    try:
        carpeta_destino = gestor.obtener_carpeta_destino()
    except Exception:
        logger.exception("Error al obtener la carpeta destino; se descarta el lote de notificaciones")
        return 202, "OK"

    for notificacion in notificaciones_validas:
        try:
            mensaje_id = notificacion["resourceData"]["id"]  #Graph solo manda el id del mensaje en la notificacion, no su contenido
            mensaje = gestor.obtener_mensaje(mensaje_id)  #Por eso hay que pedirle el contenido completo a Graph aparte
            correo = _mensaje_a_correo(mensaje)  #Normaliza el JSON de Graph a la representacion neutral Correo
            procesar_correo(
                correo, modelo, vectorizador, gestor, carpeta_destino, logger,
                enviar_notificacion, correo_destino_notificacion,
            )
        except Exception:
            logger.exception("Error al procesar una notificacion de Graph; se continua con las restantes")

    return 202, "OK"


@app.route(route="recibir_notificacion", auth_level=func.AuthLevel.ANONYMOUS)
def recibir_notificacion(req: func.HttpRequest) -> func.HttpResponse:
    logger = logging.getLogger("recibir_notificacion")

    # Handshake de validacion de Graph: debe responderse en <10s, antes de
    # cualquier otro trabajo (Graph lo exige al crear/renovar la
    # suscripcion). Ni siquiera se lee el cuerpo del request.
    validation_token = req.params.get("validationToken")
    if validation_token is not None:
        return func.HttpResponse(validation_token, status_code=200)

    try:
        cuerpo = req.get_json()
    except ValueError:
        cuerpo = {}

    # Chequeo barato antes de pagar el costo de obtener_token/_cargar_modelo:
    # si ninguna entrada del batch tiene el clientState esperado, no hay
    # nada que procesar. La lectura de WEBHOOK_CLIENT_STATE tambien esta
    # protegida: si el App Setting no existe o no se resolvio (p.ej. la
    # Key Vault reference todavia no propago), no debe propagar como 500.
    try:
        client_state_esperado = os.environ["WEBHOOK_CLIENT_STATE"]
    except KeyError:
        logger.exception("Falta el App Setting WEBHOOK_CLIENT_STATE; se descarta esta notificacion en vez de devolver 500")
        return func.HttpResponse("OK", status_code=202)

    if not _hay_notificacion_con_client_state_valido(cuerpo, client_state_esperado):
        for _notificacion in _notificaciones_del_cuerpo(cuerpo):
            logger.warning("Notificacion con clientState invalido, se ignora")
        return func.HttpResponse("OK", status_code=202)

    # Igual que el resto del modulo: un fallo aqui -- App Setting faltante
    # (incluyendo GRAPH_CLIENT_SECRET, una Key Vault reference que puede
    # tardar en propagar), token de AAD, o carga del modelo -- no debe
    # propagar como 500. Antes solo el token/modelo estaban protegidos; las
    # lecturas de os.environ quedaban afuera y un KeyError ahi se colaba
    # sin capturar. Graph reintentaria y, tras fallos repetidos, terminaria
    # eliminando la suscripcion. Se descarta este batch y se confia en que
    # el siguiente webhook (o el usuario, revisando Application Insights)
    # lo detecte si el fallo persiste.
    try:
        #Credenciales del App Registration y configuracion del buzon -- se leen recien aqui (no antes),
        #porque el chequeo barato de arriba puede evitar tener que leerlas en absoluto
        tenant_id = os.environ["GRAPH_TENANT_ID"]
        client_id = os.environ["GRAPH_CLIENT_ID"]
        client_secret = os.environ["GRAPH_CLIENT_SECRET"]  #Viene de una Key Vault reference en la configuracion de la Function App, no en texto plano
        buzon = os.environ["CORREO_MONITOREADO"]
        correo_destino_notificacion = os.environ["CORREO_DESTINO_NOTIFICACION"]
        token = obtener_token(tenant_id, client_id, client_secret)
        gestor = GestorDeCorreoGraph(token, buzon)
        modelo, vectorizador = _cargar_modelo()
    except Exception:
        logger.exception(
            "Error al leer la configuracion, obtener el token de Graph o cargar el modelo; "
            "se descarta esta notificacion en vez de devolver 500"
        )
        return func.HttpResponse("OK", status_code=202)

    #procesar_correo espera un colaborador de 4 argumentos (item, asunto, cuerpo, destino); esta closure
    #"cierra" sobre token/buzon (que procesar_correo no conoce ni deberia conocer) para cumplir esa firma
    def notificar(mensaje_id, asunto, cuerpo_notificacion, destino):
        enviar_notificacion_phishing(token, buzon, mensaje_id, asunto, cuerpo_notificacion, destino)

    status_code, body = manejar_notificacion(
        cuerpo, client_state_esperado, gestor, modelo, vectorizador,
        logger, notificar, correo_destino_notificacion,
    )
    return func.HttpResponse(body, status_code=status_code)


def ejecutar_renovacion(
    token: str,
    suscripcion_id: str,
    renovar,
    crear=None,
    buzon: str = None,
    notification_url: str = None,
    client_state: str = None,
    logger: logging.Logger = None,
) -> None:
    """
    Logica pura del Timer trigger: renueva la suscripcion via `renovar`.

    Si la renovacion falla porque la suscripcion ya expiro o fue eliminada
    (Graph responde 404 al PATCH), una suscripcion lapsada es una
    interrupcion silenciosa y permanente -- nadie vuelve a recibir correo
    y la unica senal es una excepcion sin atender en Application Insights.
    Para tener una via de recuperacion automatica, se intenta recrear la
    suscripcion desde cero via `crear` (si fue provisto), reutilizando la
    misma notification_url/client_state.

    Solo se recrea ante un 404 real (la suscripcion definitivamente ya no
    existe). Cualquier otro fallo -- timeout, error de red, un 429/503
    transitorio de Graph -- NO dispara la recreacion: la suscripcion
    original sigue siendo valida, y recrearla igual dejaria DOS
    suscripciones activas sobre el mismo buzon (correo duplicado, y
    procesar_correo fallando al intentar mover un mensaje que la otra
    suscripcion ya movio). Esos fallos transitorios simplemente se
    reintentan en el proximo ciclo del Timer (24h despues).
    """
    try:
        renovar(token, suscripcion_id)
    except requests.HTTPError as error:
        es_404 = error.response is not None and error.response.status_code == 404
        if not es_404:
            if logger is not None:
                logger.exception(
                    "Fallo transitorio al renovar la suscripcion de Graph (no es 404); "
                    "no se recrea, se reintentara en el proximo ciclo del Timer trigger"
                )
            return
        if logger is not None:
            logger.exception("La suscripcion de Graph ya no existe (404), intentando recrear...")
        _recrear_suscripcion(crear, token, buzon, notification_url, client_state, logger)
    except Exception:
        # Cualquier excepcion que no sea un HTTPError (timeout, error de
        # conexion, etc.) tambien se trata como transitoria por la misma
        # razon: sin poder confirmar que la suscripcion realmente
        # desaparecio, recrearla arriesga duplicarla.
        if logger is not None:
            logger.exception(
                "Fallo transitorio al renovar la suscripcion de Graph; "
                "no se recrea, se reintentara en el proximo ciclo del Timer trigger"
            )


def _recrear_suscripcion(crear, token, buzon, notification_url, client_state, logger) -> None:
    """Intenta recrear la suscripcion tras confirmar que expiro (404). Ver ejecutar_renovacion."""
    if crear is None:
        return
    try:
        nueva_suscripcion = crear(token, buzon, notification_url, client_state)
    except Exception:
        if logger is not None:
            logger.exception(
                "Fallo tambien la recreacion de la suscripcion de Graph tras el fallo de "
                "renovacion. Revisar credenciales/conectividad manualmente."
            )
        return
    if logger is not None:
        logger.error("Suscripcion recreada tras fallo de renovacion. ACCION REQUERIDA: actualizar el app setting GRAPH_SUBSCRIPTION_ID a %s", nueva_suscripcion["id"])


#Expresion cron de 6 campos (segundo minuto hora dia mes dia-semana): "0 0 */24 * * *" = una vez cada 24 horas, en punto.
#Corre bien antes de las ~60h de vigencia de cada suscripcion (ver _HORAS_VIGENCIA_SUSCRIPCION en graph_api_adapter.py).
@app.timer_trigger(schedule="0 0 */24 * * *", arg_name="temporizador", run_on_startup=False)
def renovar_suscripcion_trigger(temporizador: func.TimerRequest) -> None:
    #Envoltorio delgado (igual patron que recibir_notificacion): solo lee configuracion y delega
    #toda la logica real a ejecutar_renovacion, para que esta se pueda testear sin el runtime de Functions
    logger = logging.getLogger("renovar_suscripcion_trigger")
    tenant_id = os.environ["GRAPH_TENANT_ID"]
    client_id = os.environ["GRAPH_CLIENT_ID"]
    client_secret = os.environ["GRAPH_CLIENT_SECRET"]
    suscripcion_id = os.environ["GRAPH_SUBSCRIPTION_ID"]  #Se actualiza manualmente si ejecutar_renovacion tiene que recrear la suscripcion (ver su docstring)
    buzon = os.environ["CORREO_MONITOREADO"]
    notification_url = os.environ["GRAPH_NOTIFICATION_URL"]  #Usada solo si hay que recrear la suscripcion desde cero
    client_state = os.environ["WEBHOOK_CLIENT_STATE"]

    token = obtener_token(tenant_id, client_id, client_secret)
    ejecutar_renovacion(
        token, suscripcion_id, renovar_suscripcion,
        crear=crear_suscripcion, buzon=buzon, notification_url=notification_url,
        client_state=client_state, logger=logger,
    )
