"""Configuración reproducible del baseline (sin credenciales)."""

from dataclasses import dataclass
import os
import re

COMMIT_HF = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Configuracion:
    modelo: str = "google/gemma-4-E4B-it"
    revision: str = "ee0ef6023621cff504d758262d4e04895a5af4a2"
    max_tokens_nuevos: int = 512
    max_contexto: int = 8192
    lote: int = 1
    # Adapter LoRA opcional. Sin repo se usa la base sola (baseline).
    # La cátedra solo pone HF_TOKEN: el adapter final debe quedar fijado acá, no en el entorno.
    adapter_repo: str | None = None
    adapter_revision: str | None = None
    adapter_subcarpeta: str = "final"

    def __post_init__(self):
        if not COMMIT_HF.fullmatch(self.revision):
            raise ValueError("La revisión debe ser un commit HF de 40 caracteres")
        if min(self.max_tokens_nuevos, self.max_contexto, self.lote) <= 0:
            raise ValueError("Los límites deben ser positivos")
        if self.adapter_repo and not COMMIT_HF.fullmatch(self.adapter_revision or ""):
            raise ValueError("El adapter necesita una revisión HF inmutable de 40 caracteres")
        if self.adapter_revision and not self.adapter_repo:
            raise ValueError("Hay revisión de adapter sin repositorio")

    @classmethod
    def desde_entorno(cls):
        return cls(modelo=os.getenv("MODELO_ID", cls.modelo),
                   revision=os.getenv("MODELO_REVISION", cls.revision),
                   lote=int(os.getenv("MODELO_LOTE", "1")),
                   adapter_repo=os.getenv("ADAPTER_REPO", cls.adapter_repo) or None,
                   adapter_revision=os.getenv("ADAPTER_REVISION", cls.adapter_revision) or None,
                   adapter_subcarpeta=os.getenv("ADAPTER_SUBCARPETA", cls.adapter_subcarpeta))
