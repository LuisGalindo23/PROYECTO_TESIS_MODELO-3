"""
Puerto (contrato) que define lo que el núcleo de dominio necesita de
"algo que gestione correo", sin decir cómo lo hace.

Al ser un typing.Protocol, el cumplimiento es estructural.

"""
from typing import Protocol, runtime_checkable


@runtime_checkable  #Permite isinstance(objeto, GestorDeCorreoPort) para verificar en tests que un adaptador cumple el contrato (ver tests/puertos/)
class GestorDeCorreoPort(Protocol):
    def obtener_no_leidos(self) -> list: ...  #Retorna items "crudos" del adaptador (MailItem o dict de Graph), no Correo -- ver ejecutar_ciclo_de_escaneo
    def obtener_carpeta_destino(self, nombre: str): ...  #Busca (o crea si no existe) la carpeta destino; retorna su handle/id especifico del adaptador
    def marcar_como_phishing(self, item, probabilidad: float) -> None: ...  #probabilidad existe por si un adaptador futuro quisiera usarla; ninguno la usa hoy
    def mover_a_carpeta(self, item, carpeta_destino) -> None: ...  #Mueve `item` a la carpeta ya resuelta por obtener_carpeta_destino
