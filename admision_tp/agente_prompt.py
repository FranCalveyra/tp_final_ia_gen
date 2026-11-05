"""Coordinador del agente con todos los textos inyectados mediante ``PromptSpec``."""

from admision.ejemplo import extraer_json

from .prompts import PromptSpec


class _PartidaPrompt:
    def __init__(self, entorno, prompt: PromptSpec, max_turnos: int, max_sin_json: int):
        self.entorno = entorno
        self.prompt = prompt
        self.max_turnos = max_turnos
        self.max_sin_json = max_sin_json
        self.mensajes = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.mensaje_inicial_para(entorno.paciente_id)},
        ]
        self.turnos = self.sin_json = 0
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
            if fin:
                self.activa = False
                return
            self.mensajes.append({
                "role": "user",
                "content": self.prompt.observacion_para(herramienta, observacion),
            })
        if self.turnos >= self.max_turnos:
            self.activa = False


class AgentePrompt:
    """Mismo protocolo JSON y contrato de lotes que el agente de referencia."""

    def __init__(self, modelo_lote, prompt: PromptSpec, max_turnos: int = 16, max_sin_json: int = 3):
        if not isinstance(prompt, PromptSpec):
            raise TypeError("prompt debe ser PromptSpec")
        self.modelo_lote = modelo_lote
        self.prompt = prompt
        self.max_turnos = max_turnos
        self.max_sin_json = max_sin_json

    @property
    def identidad_prompt(self) -> dict:
        """Identidad que deben guardar los futuros manifiestos de experimentos."""
        return {"nombre": self.prompt.nombre, "version": self.prompt.version, "sha256": self.prompt.sha256}

    def jugar_lote(self, entornos: list) -> list[list[dict]]:
        partidas = [_PartidaPrompt(e, self.prompt, self.max_turnos, self.max_sin_json) for e in entornos]
        while activas := [partida for partida in partidas if partida.activa]:
            conversaciones = [partida.mensajes for partida in activas]
            for partida, respuesta in zip(activas, self.modelo_lote(conversaciones), strict=True):
                partida.recibir(respuesta)
        return [partida.mensajes for partida in partidas]
