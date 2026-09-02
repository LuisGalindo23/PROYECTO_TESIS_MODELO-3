"""
Representacion neutral de un correo electronico, independiente del
adaptador de origen (hoy, Microsoft Graph). `item` es el handle opaco que
el adaptador de origen necesita para despues marcar/mover ese correo (el
id de mensaje de Graph).
"""
from dataclasses import dataclass #Módulo para generar automáticamente los constructores de la clase
from datetime import datetime #Módulo para establecer el momento respecto del concepto de tiempo
from typing import Any, Optional #Módulos para establecer el tipo de variable no definido y nulabilidad explícita, respectivamente


@dataclass(frozen=True)  #Inmutable: una vez creado, un Correo no cambia -- evita releer datos que un adaptador ya invalido (ver comentarios en procesar_correo.py)
class Correo:
    item: Any  #Handle opaco especifico del adaptador (el id de string de Graph); pasa "tal cual" a gestor.marcar_como_phishing/mover_a_carpeta
    asunto: str
    cuerpo: str
    remitente: str
    #Momento (timezone-aware, UTC) en que el correo llego a la bandeja de entrada, segun el origen (Local/Azure).
    #None si el adaptador no lo provee -- procesar_correo lo trata como "sin dato" en vez de fallar.
    hora_recepcion: Optional[datetime] = None
