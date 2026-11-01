"""Puntos de entrada de la entrega."""

from pathlib import Path
import os

from admision_tp.buscador import BuscadorBM25


def construir_buscador(ruta_normas: Path) -> BuscadorBM25:
    return BuscadorBM25(ruta_normas, variante=os.getenv("BUSCADOR_VARIANTE", "baseline"))


def construir_agente():
    from admision.ejemplo import AgenteJSON
    from admision_tp.configuracion import Configuracion
    from admision_tp.modelo import ModeloLocal

    return AgenteJSON(ModeloLocal.cargar(Configuracion.desde_entorno()))
