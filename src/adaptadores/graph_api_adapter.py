"""
Adaptador de Microsoft Graph API: implementa GestorDeCorreoPort para un
buzon de Microsoft 365/Exchange Online, autenticandose con client
credentials (app-only, sin usuario interactivo). Usado desde las Azure
Functions en function_app.py.
"""
import base64  #Modulo para codificar el .eml adjunto de la notificacion en base64 (formato que exige el API de Graph)
import threading  #Para proteger el cache de apps MSAL de una condicion de carrera entre invocaciones concurrentes
import msal  #Libreria oficial de Microsoft para el flujo de autenticacion OAuth2 (client credentials)
import requests  #Modulo para hacer las llamadas HTTP REST a Microsoft Graph
from datetime import datetime, timedelta, timezone  #Para calcular la fecha de expiracion de las suscripciones

from src.dominio.correo import Correo
from src.dominio.procesar_correo import (
    NOMBRE_CARPETA_PHISHING,
    CATEGORIA_PHISHING,
    PREFIJO_ASUNTO,
    PREFIJO_ALERTA_NOTIFICACION,
)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"  #URL base de todos los endpoints de Microsoft Graph que usa este adaptador

_TIMEOUT_SEGUNDOS = 30  #Limite de espera por cada llamada HTTP, para no dejar la Function colgada si Graph no responde

# Cachea una instancia de ConfidentialClientApplication por (tenant_id,
# client_id, client_secret): MSAL guarda internamente el ultimo token que
# obtuvo y su expiracion (los tokens de Graph duran ~60-90 min), asi que
# reutilizar la MISMA instancia entre llamadas hace que
# acquire_token_for_client devuelva el token cacheado sin ida y vuelta a
# AAD, en vez de pedir uno nuevo en cada notificacion/ciclo del Timer
# trigger. Si se creara una instancia nueva cada vez (como se hacia
# antes), ese cache interno nunca se reutilizaria y cada llamada pagaria
# el costo completo de un login OAuth.
#
# El client_secret forma parte de la clave (no solo tenant_id/client_id):
# GRAPH_CLIENT_SECRET viene de una Key Vault reference pensada para
# rotarse. Sin el secret en la clave, un worker que ya tuviera una app
# cacheada seguiria usando el secreto viejo -- embebido en esa instancia
# de MSAL -- hasta que su token cacheado expirara, ignorando la rotacion.
#
# El acceso al dict esta protegido por un Lock: Azure Functions puede
# despachar invocaciones concurrentes al mismo worker de Python (varios
# hilos), y sin el Lock, dos invocaciones que lleguen casi al mismo tiempo
# con las mismas credenciales podrian pasar el "if clave not in
# _apps_msal" antes de que ninguna alcance a guardar la suya, creando dos
# instancias y descartando (sin liberar) una de ellas.
_apps_msal: dict = {}
_apps_msal_lock = threading.Lock()


def obtener_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Obtiene un access token app-only (client credentials) para Microsoft Graph, cacheado por MSAL mientras siga vigente."""
    clave = (tenant_id, client_id, client_secret)
    with _apps_msal_lock:
        if clave not in _apps_msal:
            _apps_msal[clave] = msal.ConfidentialClientApplication(
                client_id,
                authority=f"https://login.microsoftonline.com/{tenant_id}",  #URL de login especifica del tenant (no la generica "common")
                client_credential=client_secret,
            )
        app_msal = _apps_msal[clave]
    #Flujo client credentials: la app se autentica a si misma (sin usuario interactivo de por medio)
    #".default" = usar exactamente los permisos ya consentidos en el App Registration, no pedir mas
    #MSAL revisa su cache interno primero; solo hace la llamada de red si no hay token vigente
    #(acquire_token_for_client no necesita el lock: MSAL es seguro para llamadas concurrentes sobre la misma instancia)
    resultado = app_msal.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in resultado:
        #Si MSAL no devuelve un token, el diccionario trae el motivo en "error_description" en vez de "access_token"
        raise RuntimeError(f"No se pudo obtener token de Graph: {resultado.get('error_description')}")
    return resultado["access_token"]


def _parsear_fecha_recepcion(valor):
    """
    Convierte el receivedDateTime de Graph (ISO-8601 UTC, ej.
    "2024-01-15T10:30:00.1234567Z") a un datetime timezone-aware. Graph
    puede devolver hasta 7 digitos de fraccion de segundo; se truncan a 6
    (microsegundos) porque datetime.fromisoformat no admite mas.
    """
    if not valor:
        return None
    valor_sin_z = valor.rstrip("Z")
    if "." in valor_sin_z:
        parte_entera, fraccion = valor_sin_z.split(".")
        valor_sin_z = f"{parte_entera}.{fraccion[:6]}"
    return datetime.fromisoformat(valor_sin_z).replace(tzinfo=timezone.utc)


def _mensaje_a_correo(mensaje: dict) -> Correo:
    """Convierte un mensaje JSON de Microsoft Graph a la representacion neutral Correo."""
    return Correo(
        item=mensaje["id"],  #El id de Graph funciona como el "handle" que despues usan marcar_como_phishing/mover_a_carpeta
        asunto=mensaje.get("subject") or "",
        cuerpo=(mensaje.get("body") or {}).get("content") or "",  #El cuerpo viene anidado dentro de un objeto "body"
        remitente=((mensaje.get("from") or {}).get("emailAddress") or {}).get("address") or "",  #Y el remitente, dentro de "from.emailAddress.address"
        hora_recepcion=_parsear_fecha_recepcion(mensaje.get("receivedDateTime")),
    )


def _listar_todas_las_paginas(url: str, headers: dict, params: dict = None) -> list:
    """
    Hace GET a `url` y sigue "@odata.nextLink" hasta agotar todas las
    paginas, acumulando el contenido de "value" de cada una. Graph pagina
    cualquier listado (mensajes, carpetas, etc.); sin esto, los elementos
    mas alla de la primera pagina se perderian en silencio.
    """
    elementos = []
    while url:  #Se repite mientras Graph siga devolviendo una pagina siguiente
        respuesta = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT_SEGUNDOS)
        respuesta.raise_for_status()  #Lanza una excepcion si Graph responde con un codigo de error (4xx/5xx)
        cuerpo = respuesta.json()
        elementos.extend(cuerpo.get("value", []))  #Acumula los elementos de esta pagina con los de las anteriores
        url = cuerpo.get("@odata.nextLink")  #None si ya no hay mas paginas -> termina el while
        params = None  # el nextLink ya incluye los query params originales de la primera llamada
    return elementos


class GestorDeCorreoGraph:
    """Implementa GestorDeCorreoPort sobre Microsoft Graph API."""

    def __init__(self, token: str, buzon: str):
        self._buzon = buzon  #Direccion del buzon monitoreado (Graph opera "en nombre de" este buzon, no del usuario que se autentico)
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}  #Se reutilizan en cada llamada HTTP de esta instancia

    def obtener_no_leidos(self) -> list:
        """
        Retorna los mensajes JSON crudos de Graph (sin convertir a Correo),
        para tener la misma forma de contrato que GestorDeCorreoOutlookCOM
        (que retorna MailItem crudos). La conversion a Correo la hace un
        `a_correo` inyectado externamente (ver `_mensaje_a_correo` en este
        modulo o `_item_a_correo` en outlook_com_adapter), igual que ya
        ocurre para el adaptador COM en ejecutar_ciclo_de_escaneo.

        Sigue "@odata.nextLink" hasta agotar todas las paginas (via
        _listar_todas_las_paginas): sin esto, los correos no leidos mas
        alla de la primera pagina se perderian en silencio.
        """
        url = f"{GRAPH_BASE}/users/{self._buzon}/mailFolders/Inbox/messages"
        params = {"$filter": "isRead eq false"}  #Filtro OData: solo trae correos con isRead=false, no toda la bandeja
        return _listar_todas_las_paginas(url, self._headers, params)

    def obtener_mensaje(self, mensaje_id: str) -> dict:
        """Trae el contenido completo de un mensaje por su id (las notificaciones de Graph solo traen el id, no el contenido)."""
        url = f"{GRAPH_BASE}/users/{self._buzon}/messages/{mensaje_id}"
        respuesta = requests.get(url, headers=self._headers, timeout=_TIMEOUT_SEGUNDOS)
        respuesta.raise_for_status()
        return respuesta.json()

    def obtener_carpeta_destino(self, nombre: str = NOMBRE_CARPETA_PHISHING) -> str:
        """Busca la subcarpeta `nombre` en el buzon; si no existe, la crea. Retorna su id (usado luego por mover_a_carpeta)."""
        # Se listan TODAS las carpetas (paginando via _listar_todas_las_paginas)
        # antes de decidir si crear una nueva: sin esto, un buzon con mas
        # carpetas que el tamano de pagina por defecto podria no encontrar
        # "Phishing Detectado" si esta en una pagina posterior, y terminaria
        # creando una carpeta duplicada.
        url = f"{GRAPH_BASE}/users/{self._buzon}/mailFolders"
        for carpeta in _listar_todas_las_paginas(url, self._headers):
            if carpeta["displayName"] == nombre:
                return carpeta["id"]  #Ya existe: se reutiliza, no se crea una nueva

        #No se encontro en ninguna pagina: se crea la carpeta desde cero
        respuesta = requests.post(
            url, headers=self._headers, json={"displayName": nombre}, timeout=_TIMEOUT_SEGUNDOS,
        )
        respuesta.raise_for_status()
        return respuesta.json()["id"]

    def marcar_como_phishing(self, item, probabilidad: float) -> None:
        """Antepone PREFIJO_ASUNTO al asunto y agrega CATEGORIA_PHISHING (preservando categorias existentes)."""
        # probabilidad no se usa en el marcado Graph; existe solo por
        # paridad de firma con GestorDeCorreoPort.
        url = f"{GRAPH_BASE}/users/{self._buzon}/messages/{item}"
        actual = requests.get(url, headers=self._headers, timeout=_TIMEOUT_SEGUNDOS)  #Se lee el estado actual del mensaje antes de modificarlo
        actual.raise_for_status()
        cuerpo_actual = actual.json()
        asunto_actual = cuerpo_actual.get("subject") or ""
        nuevo_asunto = (
            #Si ya tiene el prefijo (p.ej. se reprocesa el mismo correo), no se duplica
            asunto_actual if asunto_actual.startswith(PREFIJO_ASUNTO) else f"{PREFIJO_ASUNTO}{asunto_actual}"
        )

        # Preservar categorías existentes (Graph retorna categories como lista de strings).
        # Agregar CATEGORIA_PHISHING solo si no está ya presente (evitar duplicados).
        categorias_existentes = cuerpo_actual.get("categories") or []
        categorias_nuevas = list(categorias_existentes)  # Copia para no mutar el original
        if CATEGORIA_PHISHING not in categorias_nuevas:
            categorias_nuevas.append(CATEGORIA_PHISHING)

        respuesta = requests.patch(  #PATCH actualiza solo los campos enviados, no reemplaza el mensaje completo
            url, headers=self._headers,
            json={"subject": nuevo_asunto, "categories": categorias_nuevas},
            timeout=_TIMEOUT_SEGUNDOS,
        )
        respuesta.raise_for_status()

    def mover_a_carpeta(self, item, carpeta_destino) -> None:
        """Mueve el mensaje `item` a la carpeta `carpeta_destino` (id devuelto por obtener_carpeta_destino)."""
        url = f"{GRAPH_BASE}/users/{self._buzon}/messages/{item}/move"
        respuesta = requests.post(
            url, headers=self._headers, json={"destinationId": carpeta_destino}, timeout=_TIMEOUT_SEGUNDOS,
        )
        respuesta.raise_for_status()  #Nota: Graph asigna un id NUEVO al mensaje movido; el "item" original queda invalido despues de esto


def enviar_notificacion_phishing(
    token: str, buzon: str, mensaje_id: str, asunto: str, cuerpo_notificacion: str, destino: str,
) -> None:
    """
    Envia una alerta a `destino` adjuntando el correo original detectado
    como phishing (`mensaje_id`) en formato MIME (.eml) -- Graph expone el
    contenido crudo de un mensaje via GET .../$value, a diferencia de
    Outlook COM que permite guardar directamente un .msg.
    """
    headers_auth = {"Authorization": f"Bearer {token}"}
    url_mime = f"{GRAPH_BASE}/users/{buzon}/messages/{mensaje_id}/$value"
    respuesta_mime = requests.get(url_mime, headers=headers_auth, timeout=_TIMEOUT_SEGUNDOS)  #Descarga el correo original en formato MIME crudo (bytes)
    respuesta_mime.raise_for_status()
    contenido_base64 = base64.b64encode(respuesta_mime.content).decode("ascii")  #Graph exige los adjuntos codificados en base64

    payload = {
        "message": {
            "subject": f"{PREFIJO_ALERTA_NOTIFICACION}{asunto}",
            "body": {"contentType": "Text", "content": cuerpo_notificacion},
            "toRecipients": [{"emailAddress": {"address": destino}}],
            "attachments": [{
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": "correo_original.eml",
                "contentType": "message/rfc822",  #Tipo MIME estandar para un correo completo adjunto
                "contentBytes": contenido_base64,
            }],
        }
    }
    url_send = f"{GRAPH_BASE}/users/{buzon}/sendMail"  #sendMail construye y envia el correo en una sola llamada (no requiere crear un borrador aparte)
    respuesta = requests.post(
        url_send, headers={**headers_auth, "Content-Type": "application/json"},
        json=payload, timeout=_TIMEOUT_SEGUNDOS,
    )
    respuesta.raise_for_status()


# Graph no permite suscripciones a mensajes con vigencia mayor a ~4230
# minutos (~70h); se usan 60h para dejar margen antes de que expire.
_HORAS_VIGENCIA_SUSCRIPCION = 60


def _expiracion_suscripcion() -> str:
    """Calcula la fecha/hora de expiracion (60h a partir de ahora), en el formato ISO-8601 que exige Graph."""
    expiracion = datetime.now(timezone.utc) + timedelta(hours=_HORAS_VIGENCIA_SUSCRIPCION)
    return expiracion.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def crear_suscripcion(token: str, buzon: str, notification_url: str, client_state: str) -> dict:
    """
    Crea la suscripcion de Graph (webhook) que notifica a `notification_url`
    cada vez que llega un correo nuevo a la Bandeja de entrada de `buzon`.
    `client_state` es un secreto compartido que Graph reenvia en cada
    notificacion, para que el receptor pueda validar que realmente viene
    de Graph (ver _hay_notificacion_con_client_state_valido en function_app.py).
    """
    payload = {
        "changeType": "created",  #Solo notifica cuando se CREA un mensaje nuevo (no ediciones/eliminaciones)
        "notificationUrl": notification_url,
        "resource": f"/users/{buzon}/mailFolders('Inbox')/messages",  #Recurso al que se suscribe: los mensajes de la Bandeja de entrada de ese buzon
        "expirationDateTime": _expiracion_suscripcion(),
        "clientState": client_state,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    respuesta = requests.post(f"{GRAPH_BASE}/subscriptions", headers=headers, json=payload, timeout=_TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()  #Graph tambien valida `notification_url` en este paso (el handshake que responde recibir_notificacion en function_app.py)
    return respuesta.json()  #Incluye el "id" de la suscripcion creada, que hay que guardar como GRAPH_SUBSCRIPTION_ID


def renovar_suscripcion(token: str, suscripcion_id: str) -> dict:
    """Extiende la expiracion de una suscripcion existente por otras 60h. Lanza HTTPError (404) si la suscripcion ya no existe."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    respuesta = requests.patch(
        f"{GRAPH_BASE}/subscriptions/{suscripcion_id}",
        headers=headers, json={"expirationDateTime": _expiracion_suscripcion()}, timeout=_TIMEOUT_SEGUNDOS,
    )
    respuesta.raise_for_status()
    return respuesta.json()
