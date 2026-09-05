"""
Puerto (contrato) que define lo que el núcleo de dominio necesita de "algo que gestione correo", sin decir cómo lo hace.

Al ser un typing.Protocol, el cumplimiento es estructural.

"""
from typing import Protocol, runtime_checkable


@runtime_checkable  #Permite isinstance(objeto, GestorDeCorreoPort) para verificar que el adaptador cumple el tipo de la clase GestorDeCorreoPort.
class GestorDeCorreoPort(Protocol):
    def obtener_no_leidos(self) -> list: ...  #Retorna items "crudos" del adaptador.
    def obtener_carpeta_destino(self, nombre: str): ...  #Busca (o crea si no existe) la carpeta destino.
    def marcar_como_phishing(self, item, probabilidad: float) -> None: ...  #Antepone PREFIJO_ASUNTO al asunto y agrega CATEGORIA_PHISHING (preservando categorias existentes) 
    def mover_a_carpeta(self, item, carpeta_destino) -> None: ...  #Mueve `item` a la carpeta ya resuelta por obtener_carpeta_destino
