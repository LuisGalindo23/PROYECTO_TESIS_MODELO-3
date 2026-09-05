"""
Representacion neutral de un correo electronico.
"""
from dataclasses import dataclass #Módulo para generar automáticamente los constructores de la clase
from datetime import datetime #Módulo para establecer el momento respecto del concepto de tiempo
from typing import Any, Optional #Módulos para establecer el tipo de variable no definido y nulabilidad explícita, respectivamente


@dataclass(frozen=True)  #Inmutable: una vez creado, un Correo no cambia.
class Correo:
    item: Any #Tipo de correo variable por el adaptor.
    asunto: str
    cuerpo: str
    remitente: str
    #Timezone en que el correo llego a la bandeja de entrada, segun el origen (Azure).
    #None si el adaptador no lo provee.
    hora_recepcion: Optional[datetime] = None
