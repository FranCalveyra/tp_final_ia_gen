"""Configuración reproducible del baseline (sin credenciales)."""

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class Configuracion:
    modelo: str = "google/gemma-4-E4B-it"
    revision: str = "ee0ef6023621cff504d758262d4e04895a5af4a2"
    max_tokens_nuevos: int = 512
    max_contexto: int = 8192
    lote: int = 1

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("La revisión debe ser un commit HF de 40 caracteres")
        if min(self.max_tokens_nuevos, self.max_contexto, self.lote) <= 0:
            raise ValueError("Los límites deben ser positivos")

    @classmethod
    def desde_entorno(cls):
        return cls(modelo=os.getenv("MODELO_ID", cls.modelo),
                   revision=os.getenv("MODELO_REVISION", cls.revision),
                   lote=int(os.getenv("MODELO_LOTE", "1")))
