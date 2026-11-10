"""Coordinador del agente con todos los textos inyectados mediante ``PromptSpec``.

Termina cada caso cuando el entorno lo termina, cuando llega al tope de llamadas del kit
(`admision.entorno.MAX_LLAMADAS`), tras `max_sin_json` respuestas seguidas sin JSON o, si el
modelo sabe medir su contexto, cuando la conversación ya no entra en él. Nunca recorta mensajes.
"""

from admision.ejemplo import extraer_json
from admision.entorno import MAX_LLAMADAS

from .prompts import PromptSpec

# No es texto para el modelo: queda al final de la conversación devuelta para auditar el corte.
MENSAJE_CONTEXTO_EXCEDIDO = (
    "[agente] Caso terminado sin registrar decisión: la conversación ocupa {tokens} tokens de "
    "entrada y el modelo admite {limite}. No se recorta el contexto.")


def max_turnos_para(max_llamadas: int, max_sin_json: int) -> int:
    """Turnos que alcanzan para usar todas las llamadas con hasta `max_sin_json - 1` reintentos
    sin JSON antes de cada una (los reintentos no consumen llamadas del entorno)."""
    return max_llamadas * max_sin_json


class _PartidaPrompt:
    def __init__(self, entorno, prompt: PromptSpec, max_turnos: int, max_sin_json: int,
                 max_llamadas: int):
        self.entorno = entorno
        self.prompt = prompt
        self.max_turnos = max_turnos
        self.max_sin_json = max_sin_json
        self.max_llamadas = max_llamadas
        self.mensajes = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.mensaje_inicial_para(entorno.paciente_id)},
        ]
        self.turnos = self.sin_json = self.llamadas = 0
        self.activa = True

    def recibir(self, respuesta: str) -> None:
        self.turnos += 1
        self.mensajes.append({"role": "assistant", "content": respuesta})
        obj = extraer_json(respuesta)
        if obj is None:
            self.sin_json += 1
            if self.sin_json >= self.max_sin_json:
                self.activa = False
                return
            self.mensajes.append({"role": "user", "content": self.prompt.mensaje_json_invalido})
        else:
            self.sin_json = 0
            herramienta = str(obj.get("herramienta", ""))
            observacion, fin = self.entorno.paso(herramienta, obj.get("argumentos") or {})
            self.llamadas += 1
            if fin:
                self.activa = False
                return
            self.mensajes.append({
                "role": "user",
                "content": self.prompt.observacion_para(herramienta, observacion),
            })
        if self.turnos >= self.max_turnos or self.llamadas >= self.max_llamadas:
            self.activa = False

    def cortar_por_contexto(self, tokens: int, limite: int) -> None:
        self.mensajes.append({"role": "user", "content": MENSAJE_CONTEXTO_EXCEDIDO.format(
            tokens=tokens, limite=limite)})
        self.activa = False


class AgentePrompt:
    """Mismo protocolo JSON y contrato de lotes que el agente de referencia.

    `max_turnos=None` lo deriva del tope de llamadas del kit. Si `modelo_lote` ofrece
    `medir_contexto(conversacion) -> (tokens, limite)`, antes de cada llamada se termina sin
    decisión cada caso que no entra; los demás casos del lote siguen.
    """

    def __init__(self, modelo_lote, prompt: PromptSpec, max_turnos: int | None = None,
                 max_sin_json: int = 3, max_llamadas: int = MAX_LLAMADAS):
        if not isinstance(prompt, PromptSpec):
            raise TypeError("prompt debe ser PromptSpec")
        if max_turnos is None:
            max_turnos = max_turnos_para(max_llamadas, max_sin_json)
        self.modelo_lote = modelo_lote
        self.prompt = prompt
        self.max_turnos = max_turnos
        self.max_sin_json = max_sin_json
        self.max_llamadas = max_llamadas

    @property
    def identidad_prompt(self) -> dict:
        """Identidad que deben guardar los futuros manifiestos de experimentos."""
        return {"nombre": self.prompt.nombre, "version": self.prompt.version, "sha256": self.prompt.sha256}

    def _activas(self, partidas: list) -> list:
        activas = [partida for partida in partidas if partida.activa]
        medir = getattr(self.modelo_lote, "medir_contexto", None)
        if medir is None:
            return activas
        for partida in activas:
            tokens, limite = medir(partida.mensajes)
            if tokens > limite:
                partida.cortar_por_contexto(tokens, limite)
        return [partida for partida in activas if partida.activa]

    def jugar_lote(self, entornos: list) -> list[list[dict]]:
        partidas = [_PartidaPrompt(e, self.prompt, self.max_turnos, self.max_sin_json, self.max_llamadas)
                    for e in entornos]
        while activas := self._activas(partidas):
            conversaciones = [partida.mensajes for partida in activas]
            for partida, respuesta in zip(activas, self.modelo_lote(conversaciones), strict=True):
                partida.recibir(respuesta)
        return [partida.mensajes for partida in partidas]
