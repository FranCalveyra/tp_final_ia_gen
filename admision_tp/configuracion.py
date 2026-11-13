"""Configuración reproducible del baseline (sin credenciales)."""

from dataclasses import dataclass
import os
import re

COMMIT_HF = re.compile(r"[0-9a-f]{40}")
REPO_HF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
SUBCARPETA_HF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
DTYPES = ("bfloat16", "float16", "float32")
DEVICE_MAP = re.compile(r"auto|cuda(:\d+)?")
# Adapter de la entrega (SFT sobre v3-procedimiento, Kaggle v32+v33). Sin ADAPTER_REPO en el entorno se usa este;
# ADAPTER_REPO vacío pide la base sola.
ADAPTER_ENTREGA = ("FranCalveyra/tp-admision", "770c9fac84d5718b0b2cf4c40c3451cad322ac9e")
# Lote de la entrega: con el que corrieron todas las evaluaciones reportadas (evaluaciones/). MODELO_LOTE lo cambia.
LOTE_ENTREGA = 4


def validar_adapter(dato) -> dict:
    """Referencia a un adapter publicado, como la escriben los planes y configuraciones JSON.

    `{"repo": "propietario/nombre", "revision": <commit HF de 40 hex>, "subcarpeta": <carpeta>}`; devuelve el
    mismo dict. Única validación de esa forma (preparación del SFT, evaluación de prompts y planes).
    """
    if (not isinstance(dato, dict) or set(dato) != {"repo", "revision", "subcarpeta"}
            or not all(isinstance(v, str) for v in dato.values())
            or not REPO_HF.fullmatch(dato["repo"]) or not COMMIT_HF.fullmatch(dato["revision"])
            or not SUBCARPETA_HF.fullmatch(dato["subcarpeta"])):
        raise ValueError("adapter requiere repo propietario/nombre, revisión HF inmutable (40 hex) y subcarpeta")
    return dato


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
    # JSONL con tokens, segundos y VRAM por lote. Solo para nuestras corridas; vacío = no mide.
    metricas: str | None = None
    # La entrega corre en A100 (bf16, una GPU). Otros valores son solo para desarrollo (p. ej. T4 x2).
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"

    def __post_init__(self):
        if not COMMIT_HF.fullmatch(self.revision):
            raise ValueError("La revisión debe ser un commit HF de 40 caracteres")
        if min(self.max_tokens_nuevos, self.max_contexto, self.lote) <= 0:
            raise ValueError("Los límites deben ser positivos")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype debe ser uno de {', '.join(DTYPES)}")
        if not DEVICE_MAP.fullmatch(self.device_map):
            raise ValueError("device_map debe ser auto, cuda o cuda:N")
        if self.adapter_repo and not COMMIT_HF.fullmatch(self.adapter_revision or ""):
            raise ValueError("El adapter necesita una revisión HF inmutable de 40 caracteres")
        if self.adapter_revision and not self.adapter_repo:
            raise ValueError("Hay revisión de adapter sin repositorio")

    @classmethod
    def desde_entorno(cls):
        if "ADAPTER_REPO" in os.environ:
            repo, revision = os.environ["ADAPTER_REPO"], None  # Vacío pide la base sola.
        elif {"ADAPTER_REVISION", "ADAPTER_SUBCARPETA"} & set(os.environ):
            # Si no, cambiaría en silencio la revisión o la carpeta del adapter de la entrega.
            raise ValueError("ADAPTER_REVISION y ADAPTER_SUBCARPETA requieren ADAPTER_REPO")
        else:
            repo, revision = ADAPTER_ENTREGA
        return cls(modelo=os.getenv("MODELO_ID", cls.modelo),
                   revision=os.getenv("MODELO_REVISION", cls.revision),
                   lote=int(os.getenv("MODELO_LOTE", str(LOTE_ENTREGA))),
                   adapter_repo=repo or None,
                   adapter_revision=os.getenv("ADAPTER_REVISION", revision) or None,
                   adapter_subcarpeta=os.getenv("ADAPTER_SUBCARPETA", cls.adapter_subcarpeta),
                   metricas=os.getenv("MODELO_METRICAS") or None,
                   dtype=os.getenv("MODELO_DTYPE", cls.dtype),
                   device_map=os.getenv("MODELO_DEVICE_MAP", cls.device_map))
