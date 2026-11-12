"""Puntos de entrada de la entrega."""

from pathlib import Path
import os

from admision_tp.buscador import BuscadorBM25
from admision_tp.prompts import RAIZ, RUTA_ENTREGA, PromptSpec


def construir_buscador(ruta_normas: Path) -> BuscadorBM25:
    return BuscadorBM25(ruta_normas, variante=os.getenv("BUSCADOR_VARIANTE", "baseline"))


def cargar_prompt() -> PromptSpec:
    """PromptSpec de `PROMPT_SPEC` (relativa a la raíz del repo) o el de la entrega; valida su sha256."""
    ruta = Path(os.getenv("PROMPT_SPEC") or RUTA_ENTREGA)
    return PromptSpec.cargar(ruta if ruta.is_absolute() else RAIZ / ruta)


def construir_agente():
    from admision_tp.agente_prompt import AgentePrompt
    from admision_tp.configuracion import Configuracion
    from admision_tp.modelo import ModeloLocal

    prompt = cargar_prompt()  # antes de cargar el modelo: un prompt inválido falla sin usar la GPU
    return AgentePrompt(ModeloLocal.cargar(Configuracion.desde_entorno()), prompt)
