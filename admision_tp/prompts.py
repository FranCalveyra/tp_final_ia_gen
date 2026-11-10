"""Prompts versionados y reproducibles para el coordinador del agente."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from string import Formatter


RAIZ = Path(__file__).resolve().parents[1]
RUTA_BASELINE = RAIZ / "configuraciones/prompts/baseline.json"
_CAMPOS = {
    "version", "nombre", "system_prompt", "mensaje_inicial",
    "formato_observacion", "mensaje_json_invalido", "sha256",
}


def _placeholders(plantilla: str) -> set[str]:
    try:
        campos = {nombre for _, nombre, _, _ in Formatter().parse(plantilla) if nombre is not None}
    except ValueError:
        raise ValueError("plantilla con llaves inválidas") from None
    if any(not nombre or "." in nombre or "[" in nombre for nombre in campos):
        raise ValueError("plantilla con placeholder inválido")
    return campos


@dataclass(frozen=True)
class PromptSpec:
    """Todo texto configurable que el coordinador agrega a la conversación.

    El hash cubre la versión del formato y los cuatro textos que afectan al modelo.
    `nombre` es una etiqueta humana y no modifica la identidad del contenido.
    """

    version: int
    nombre: str
    system_prompt: str
    mensaje_inicial: str
    formato_observacion: str
    mensaje_json_invalido: str

    def __post_init__(self):
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("version requiere un entero positivo")
        if not isinstance(self.nombre, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.nombre):
            raise ValueError("nombre de prompt inválido")
        textos = (self.system_prompt, self.mensaje_inicial,
                  self.formato_observacion, self.mensaje_json_invalido)
        if any(not isinstance(texto, str) or not texto.strip() for texto in textos):
            raise ValueError("los textos del prompt no pueden estar vacíos")
        if _placeholders(self.mensaje_inicial) != {"paciente_id"}:
            raise ValueError("mensaje_inicial requiere únicamente {paciente_id}")
        if _placeholders(self.formato_observacion) != {"herramienta", "observacion"}:
            raise ValueError("formato_observacion requiere {herramienta} y {observacion}")

    def _contenido(self) -> dict:
        return {
            "version": self.version,
            "system_prompt": self.system_prompt,
            "mensaje_inicial": self.mensaje_inicial,
            "formato_observacion": self.formato_observacion,
            "mensaje_json_invalido": self.mensaje_json_invalido,
        }

    @property
    def sha256(self) -> str:
        contenido = json.dumps(
            self._contenido(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(contenido).hexdigest()

    def mensaje_inicial_para(self, paciente_id: str) -> str:
        return self.mensaje_inicial.format(paciente_id=paciente_id)

    def observacion_para(self, herramienta: str, observacion: str) -> str:
        return self.formato_observacion.format(herramienta=herramienta, observacion=observacion)

    def como_dict(self) -> dict:
        return {"version": self.version, "nombre": self.nombre, **{
            k: v for k, v in self._contenido().items() if k != "version"
        }, "sha256": self.sha256}

    @classmethod
    def desde_dict(cls, dato: dict) -> "PromptSpec":
        if not isinstance(dato, dict) or set(dato) != _CAMPOS:
            raise ValueError("configuración de prompt incompleta o con campos desconocidos")
        esperado = dato["sha256"]
        if not isinstance(esperado, str) or not re.fullmatch(r"[0-9a-f]{64}", esperado):
            raise ValueError("sha256 de prompt inválido")
        prompt = cls(**{k: dato[k] for k in _CAMPOS if k != "sha256"})
        if prompt.sha256 != esperado:
            raise ValueError("el contenido del prompt no coincide con su sha256")
        return prompt

    @classmethod
    def cargar(cls, ruta: Path) -> "PromptSpec":
        try:
            dato = json.loads(Path(ruta).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("no se pudo leer una configuración de prompt válida") from None
        return cls.desde_dict(dato)


def cargar_prompt_baseline() -> PromptSpec:
    return PromptSpec.cargar(RUTA_BASELINE)
