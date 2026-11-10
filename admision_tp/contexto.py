"""Medición offline del contexto que un PromptSpec le pasa al modelo, sobre trazas ya jugadas.

Rejuega cada traza con `AgentePrompt`: el "modelo" repite las respuestas grabadas y el "entorno"
devuelve las observaciones del log. La conversación sale del mismo código que en inferencia y se
mide con `caracteres_de_contexto` del kit. Supone que el modelo respondería lo mismo con otro
prompt: mide formato, no cambios de conducta.
"""

from collections import Counter
import re

from admision.correr import caracteres_de_contexto
from admision.ejemplo import extraer_json
from admision.entorno import MAX_LLAMADAS

from .agente_prompt import AgentePrompt
from .prompts import PromptSpec

PARTES = ("sistema", "inicial", "busquedas", "otras_observaciones", "correcciones", "respuestas")


class _EntornoGrabado:
    def __init__(self, paciente_id: str, observaciones: list[str], cerrado: bool):
        self.paciente_id, self.observaciones, self.cerrado = paciente_id, observaciones, cerrado
        self.llamadas = 0

    def paso(self, herramienta, argumentos):
        if self.llamadas >= len(self.observaciones):
            raise ValueError("la traza tiene menos observaciones que llamadas")
        self.llamadas += 1
        return self.observaciones[self.llamadas - 1], self.cerrado and self.llamadas == len(self.observaciones)


def rejugar(prompt: PromptSpec, paciente_id: str, respuestas: list[str], observaciones: list[str],
            cerrado: bool) -> list[dict]:
    """Conversación que arma `AgentePrompt(prompt)` con esas respuestas y observaciones.

    `cerrado` indica si el entorno terminó el caso en la última llamada (ficha o tope).
    """
    if not respuestas:
        raise ValueError("la traza no tiene respuestas del modelo")
    turnos = iter(respuestas)
    entorno = _EntornoGrabado(paciente_id, list(observaciones), cerrado)
    agente = AgentePrompt(lambda conversaciones: [next(turnos) for _ in conversaciones], prompt,
                          max_turnos=len(respuestas))
    mensajes = agente.jugar_lote([entorno])[0]
    if entorno.llamadas != len(observaciones):
        raise ValueError("la traza tiene observaciones que el agente no pidió")
    return mensajes


def paciente_de(mensajes: list[dict], origen: PromptSpec) -> str:
    """Id del paciente según el `mensaje_inicial` del prompt con el que se jugó la traza."""
    patron = re.escape(origen.mensaje_inicial).replace(re.escape("{paciente_id}"), "(.+?)")
    coincidencia = re.fullmatch(patron, mensajes[1]["content"], re.S)
    if coincidencia is None:
        raise ValueError("el mensaje inicial no corresponde al prompt de origen")
    return coincidencia.group(1)


def rejugar_traza(traza: dict, prompt: PromptSpec, origen: PromptSpec) -> list[dict]:
    """Rejuega una traza de `admision.correr` (jugada con `origen`) con `prompt`."""
    mensajes, log = traza["mensajes"], traza["log"]
    cerrado = traza["decision"] is not None or len(log) >= MAX_LLAMADAS
    return rejugar(prompt, paciente_de(mensajes, origen),
                   [m["content"] for m in mensajes if m["role"] == "assistant"],
                   [e["observacion"] for e in log], cerrado)


def desglose(mensajes: list[dict], prompt: PromptSpec) -> dict:
    """`caracteres_de_contexto` repartido por tipo de mensaje; la suma es el total del kit."""
    partes = Counter({parte: 0 for parte in PARTES})
    restantes = sum(m["role"] == "assistant" for m in mensajes)
    herramienta = None
    for i, mensaje in enumerate(mensajes):
        contenido = str(mensaje["content"])
        if mensaje["role"] == "assistant":
            restantes -= 1
            parte, obj = "respuestas", extraer_json(contenido)
            herramienta = str(obj.get("herramienta", "")) if obj else None
        elif i == 0:
            parte = "sistema"
        elif i == 1:
            parte = "inicial"
        elif contenido == prompt.mensaje_json_invalido:
            parte = "correcciones"
        else:
            parte = "busquedas" if herramienta == "buscar_normativa" else "otras_observaciones"
        partes[parte] += len(contenido) * restantes
    if sum(partes.values()) != caracteres_de_contexto(mensajes):
        raise RuntimeError("el desglose no coincide con caracteres_de_contexto")
    return dict(partes)
