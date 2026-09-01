"""
Punto de entrada del sistema en produccion para Outlook de escritorio:
escanea periodicamente (polling) la Bandeja de entrada de Outlook,
clasifica cada correo no leido con el SVM ya entrenado (via
src.dominio.procesar_correo, compartido con el camino de Azure/Graph en
function_app.py) y mueve los que detecta como phishing a la carpeta
"Phishing Detectado".

Solo main() toca la API COM real y el bucle de tiempo, y por eso no tiene
test unitario (es responsabilidad de integracion).
"""
import logging #Módulo para operaciones de logs
import time #Módulo para operaciones relacionadas con tiempo real del sistema.

from src.dominio.procesar_correo import ejecutar_ciclo_de_escaneo
from src.adaptadores.outlook_com_adapter import (
    conectar_outlook,
    enviar_notificacion_phishing,
    GestorDeCorreoOutlookCOM,
    _item_a_correo,
)

RUTA_MODELO = "modelo/svm_phishing.pkl" #Modelo SVM previamente entrenado y serializado. Contiene los parámetros aprendidos durante el entrenamiento.
RUTA_VECTORIZADOR = "modelo/vectorizador_tfidf.pkl" #Modelo del vectorizador TDIF con los pesos de los elementos textuales
RUTA_LOG = "logs/monitor.log" #Archivo donde almacenan los logs
INTERVALO_SEGUNDOS = 60 #El tiempo periódico de escaneo de correos
NOTIFICAR_POR_CORREO = True  # ACTIVO: se estan enviando notificaciones reales al detectar phishing
CORREO_DESTINO_NOTIFICACION = "jvp2351@outlook.com"  # Destino real activo (no es un placeholder)

# Número de índice de la carpeta "Bandeja de entrada" en Outlook
# (constante fija de la API MAPI = 6)
# Soluciona el problema de identificar la bandeja de entrada por texto en diferentes idiomas
CARPETA_BANDEJA_ENTRADA = 6


def configurar_logger() -> logging.Logger:
    logger = logging.getLogger("monitor_phishing") #Define un registrador(logger)
    logger.setLevel(logging.INFO) #Define el umbral de mensajes = "info"
    manejador = logging.FileHandler(RUTA_LOG, encoding="utf-8") #Establece el formato de codificación más usado y la ruta
    manejador.setFormatter(logging.Formatter("%(asctime)s - %(message)s")) #Establece el formato del registro: el tiempo del evento y el contenido
    logger.addHandler(manejador) #Guarda la configuración en el registrador
    return logger #Devuelve el registrador ya configurado


def main():
    import joblib #Módulo para guardar en disco el modelo y vectorizador

    logger = configurar_logger() #Configurar el registrador
    logger.info("Iniciando monitor de phishing. Intervalo: %s segundos.", INTERVALO_SEGUNDOS)

    modelo = joblib.load(RUTA_MODELO) #Carga el modelo
    vectorizador = joblib.load(RUTA_VECTORIZADOR) #Carga el vectorizador

    namespace = conectar_outlook() #Realiza la conexión a outlook
    bandeja_entrada = namespace.GetDefaultFolder(CARPETA_BANDEJA_ENTRADA) #Carga la carpeta de bandeja de entrada
    gestor = GestorDeCorreoOutlookCOM(bandeja_entrada)
    carpeta_destino = gestor.obtener_carpeta_destino() #Carga la subcarpeta "Pishing Detectado"

    notificador = enviar_notificacion_phishing if NOTIFICAR_POR_CORREO else None

    while True:
        try:
            cantidad = ejecutar_ciclo_de_escaneo(
                gestor, modelo, vectorizador, carpeta_destino, logger,
                _item_a_correo, notificador, CORREO_DESTINO_NOTIFICACION,
            )
            if cantidad: #0 es Falso y otro numero es Verdadero
                print(f"[{time.strftime('%H:%M:%S')}] {cantidad} correo(s) marcado(s) como phishing.")
        except Exception:
            logger.exception("Error durante el ciclo de escaneo") #Mensaje de salida por si se genera un error
        time.sleep(INTERVALO_SEGUNDOS) #Define la pausa de 60 segundos entre cada intento


if __name__ == "__main__":
    main()
