import os  # Modulo para construir la ruta del archivo temporal .msg
import tempfile  # Modulo para crear un directorio temporal que se autolimpia
from datetime import timezone  # Para normalizar item.ReceivedTime a UTC-aware antes de guardarlo en Correo

from src.dominio.correo import Correo
from src.dominio.procesar_correo import (
    NOMBRE_CARPETA_PHISHING,
    CATEGORIA_PHISHING,
    PREFIJO_ASUNTO,
    PREFIJO_ALERTA_NOTIFICACION,
)

def conectar_outlook():
    """|
    Conecta con la instancia de Outlook de escritorio abierta en la máquina
    local y retorna el namespace MAPI, punto de entrada para acceder a
    carpetas y correos. Requiere que Outlook esté instalado (pywin32).
    """
    import win32com.client  # import local: necesario para conexion con outlook local

    outlook = win32com.client.Dispatch("Outlook.Application") #Crea una instancia de la aplicación Outlook
    namespace = outlook.GetNamespace("MAPI") #Punto de entrada para acceder a correos, carpetas y cuentas.
    namespace.Logon("", "", False, False) #Realiza la conexión con la sesión activa sin mostrar ningun mensaje
    return namespace


def obtener_carpeta_destino(bandeja_entrada, nombre_carpeta: str = NOMBRE_CARPETA_PHISHING):
    """
    Busca la subcarpeta `nombre_carpeta` dentro de la Bandeja de entrada.
    Si no existe, la crea. Retorna la carpeta MAPI resultante.
    """
    for carpeta in bandeja_entrada.Folders:
        if carpeta.Name == nombre_carpeta:
            return carpeta
    return bandeja_entrada.Folders.Add(nombre_carpeta)


def marcar_como_phishing(item):
    """
    Marca un MailItem como phishing: le asigna la categoría de Outlook "Posible Phishing"
    y antepone [PHISHING] al asunto (si no lo tiene ya, para evitar prefijos duplicados
    si el correo se re-procesa). Guarda los cambios con item.Save().
    """
    if not item.Subject.startswith(PREFIJO_ASUNTO): #Evita anteponer el prefijo dos veces si el correo se reprocesa
        item.Subject = f"{PREFIJO_ASUNTO}{item.Subject}"

    #Outlook COM guarda las categorias como un solo string separado por comas (no una lista),
    #asi que hay que parsearlo manualmente para poder revisar/agregar sin perder las existentes
    texto_categorias = item.Categories or ""
    partes = texto_categorias.split(",")
    categorias_actuales = []
    for parte in partes:
        parte_limpia = parte.strip() #Quita espacios sobrantes (Outlook suele guardar "Cat1, Cat2" con espacio despues de la coma)
        if parte_limpia:
            categorias_actuales.append(parte_limpia)
    if CATEGORIA_PHISHING not in categorias_actuales: #No duplicar la categoria si el correo ya la tenia
        categorias_actuales.append(CATEGORIA_PHISHING)
    item.Categories = ", ".join(categorias_actuales) #Se reconstruye el string en el mismo formato que espera Outlook

    item.Save() #Sin este llamado, los cambios a Subject/Categories quedan solo en memoria y no se persisten


def mover_a_carpeta(item, carpeta_destino):
    """Mueve un correo a la carpeta destino -Phishing Detectado-"""
    item.Move(carpeta_destino)


def enviar_notificacion_phishing(item, asunto: str, cuerpo_notificacion: str, destino: str) -> None:
    """
    Envia un correo nuevo a `destino` con `cuerpo_notificacion` como cuerpo,
    adjuntando `item` (el correo original detectado como phishing) como
    archivo .msg. `asunto` es el asunto original ya saneado (no se lee
    item.Subject en vivo, para no depender de si `item` fue mutado). El
    archivo temporal se borra automaticamente al terminar.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directorio_temporal:
        ruta_msg = os.path.join(directorio_temporal, "correo_original.msg")
        item.SaveAs(ruta_msg, 3)  # 3 = olMSG

        nuevo_correo = item.Application.CreateItem(0)  # 0 = olMailItem
        nuevo_correo.To = destino
        nuevo_correo.Subject = f"{PREFIJO_ALERTA_NOTIFICACION}{asunto}"
        nuevo_correo.Body = cuerpo_notificacion
        nuevo_correo.Attachments.Add(ruta_msg)
        nuevo_correo.Send()


def _item_a_correo(item) -> Correo:
    """
    Convierte un MailItem de Outlook COM a la representacion neutral Correo.

    Nota: item.SenderEmailAddress puede devolver un legacyExchangeDN
    ("/O=EXCHANGELABS/...") en vez de una direccion SMTP para remitentes
    internos de Exchange/M365 (cuando item.SenderEmailType != "SMTP"). Esto
    no afecta la clasificacion en si, pero si eso ocurre, el guard
    anti-bucle de procesar_correo (que compara remitente contra
    correo_destino_notificacion por igualdad exacta de string SMTP) dejaria
    de reconocer las propias alertas del sistema como tales.

    item.ReceivedTime es un pywintypes.Time (timezone-aware): se normaliza
    a UTC via .astimezone() para que sea comparable con datetime.now(utc)
    en _formatear_tiempo_medio_deteccion, sin importar la zona horaria
    local configurada en Outlook.
    """
    return Correo(
        item=item,
        asunto=item.Subject or "",
        cuerpo=item.Body or "",
        remitente=item.SenderEmailAddress or "",
        hora_recepcion=item.ReceivedTime.astimezone(timezone.utc) if item.ReceivedTime else None,
    )


class GestorDeCorreoOutlookCOM:
    """
    Implementa GestorDeCorreoPort sobre Outlook COM, delegando a las
    funciones sueltas ya existentes en este modulo (se mantienen sin
    cambios para no romper su uso directo/tests existentes).
    """

    def __init__(self, bandeja_entrada):
        self._bandeja_entrada = bandeja_entrada #La carpeta Bandeja de entrada MAPI, obtenida por main() en monitor_outlook.py

    def obtener_no_leidos(self) -> list:
        #list(...Items) copia la coleccion antes de filtrar: iterar y mover items de la misma
        #coleccion COM en vivo puede saltarse elementos, ya que Outlook la reindexa al mover uno
        return [item for item in list(self._bandeja_entrada.Items) if item.UnRead]

    def obtener_carpeta_destino(self, nombre: str = NOMBRE_CARPETA_PHISHING):
        return obtener_carpeta_destino(self._bandeja_entrada, nombre) #Delega a la funcion suelta de arriba, que ya recibe la bandeja como parametro

    def marcar_como_phishing(self, item, probabilidad: float) -> None:
        # probabilidad no se usa en el marcado COM; existe solo por paridad
        # de firma con GestorDeCorreoPort (que el adaptador Graph si usa
        # de forma equivalente).
        marcar_como_phishing(item)

    def mover_a_carpeta(self, item, carpeta_destino) -> None:
        mover_a_carpeta(item, carpeta_destino)
