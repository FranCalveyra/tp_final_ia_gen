"""Runner auditable para evaluar prompts sobre casos y esperadas externos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import shutil

from admision.casos import cargar_casos, cargar_esperadas
from admision.correr import jugar, puntuar

from .agente_prompt import AgentePrompt
from .buscador import BuscadorBM25
from .configuracion import Configuracion
from .evaluacion_prompts import calcular_metricas_candidato, medir_longitud_prompt
from .preparacion import _canonical, _codigo, _sha
from .prompts import PromptSpec


_CAMPOS_CONFIG = {
    "version", "modelo", "revision", "semilla", "max_tokens_nuevos", "max_contexto",
    "lote", "dtype", "device_map", "buscador_variante", "max_turnos", "max_sin_json",
}


@dataclass(frozen=True)
class ConfiguracionEvaluacionPrompt:
    version: int
    modelo: str
    revision: str
    semilla: int
    max_tokens_nuevos: int
    max_contexto: int
    lote: int
    dtype: str
    device_map: str
    buscador_variante: str
    max_turnos: int
    max_sin_json: int

    def __post_init__(self):
        if self.version != 1 or type(self.version) is not int:
            raise ValueError("configuración de evaluación: versión no soportada")
        if not isinstance(self.modelo, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.modelo):
            raise ValueError("configuración de evaluación: modelo inválido")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("configuración de evaluación: revisión no inmutable")
        if type(self.semilla) is not int or self.semilla < 0:
            raise ValueError("configuración de evaluación: semilla inválida")
        for campo in ("max_tokens_nuevos", "max_contexto", "lote", "max_turnos", "max_sin_json"):
            if type(getattr(self, campo)) is not int or getattr(self, campo) <= 0:
                raise ValueError(f"configuración de evaluación: {campo} requiere entero positivo")
        if self.buscador_variante not in {"baseline", "sin_comunes"}:
            raise ValueError("configuración de evaluación: buscador_variante inválida")
        # Reutiliza la validación de dtype, device_map y límites del modelo operativo.
        self.configuracion_modelo()

    def configuracion_modelo(self) -> Configuracion:
        return Configuracion(
            modelo=self.modelo, revision=self.revision,
            max_tokens_nuevos=self.max_tokens_nuevos, max_contexto=self.max_contexto,
            lote=self.lote, dtype=self.dtype, device_map=self.device_map,
        )

    def como_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, dato: dict) -> "ConfiguracionEvaluacionPrompt":
        if not isinstance(dato, dict) or set(dato) != _CAMPOS_CONFIG:
            raise ValueError("configuración de evaluación incompleta o con campos desconocidos")
        return cls(**dato)

    @classmethod
    def cargar(cls, ruta) -> "ConfiguracionEvaluacionPrompt":
        try:
            dato = json.loads(Path(ruta).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValueError("no se pudo leer una configuración de evaluación válida") from None
        return cls.desde_dict(dato)


class _EntregaCandidata:
    def __init__(self, prompt, config, modelo_lote):
        self.prompt, self.config, self.modelo_lote = prompt, config, modelo_lote

    def construir_buscador(self, ruta_normas):
        return BuscadorBM25(ruta_normas, variante=self.config.buscador_variante)

    def construir_agente(self):
        return AgentePrompt(
            self.modelo_lote, self.prompt,
            max_turnos=self.config.max_turnos, max_sin_json=self.config.max_sin_json)


def _copiar_archivo(origen, destino) -> bytes:
    origen, destino = Path(origen), Path(destino)
    if origen.is_symlink() or not origen.is_file():
        raise ValueError(f"entrada ausente o enlazada: {destino.name}")
    try:
        contenido = origen.read_bytes()
        destino.write_bytes(contenido)
    except OSError:
        raise ValueError(f"no se pudo conservar la entrada: {destino.name}") from None
    return contenido


def _copiar_normas(origen, destino) -> dict:
    origen, destino = Path(origen), Path(destino)
    if origen.is_symlink() or not origen.is_dir() or any(p.is_symlink() for p in origen.rglob("*")):
        raise ValueError("carpeta de normas ausente o enlazada")
    try:
        shutil.copytree(origen, destino)
    except OSError:
        raise ValueError("no se pudieron conservar las normas") from None
    archivos = [p for p in destino.rglob("*") if p.is_file()]
    if not archivos:
        raise ValueError("carpeta de normas vacía")
    return {p.relative_to(destino).as_posix(): _sha(p.read_bytes()) for p in sorted(archivos)}


def _cargar_modelo_real(config):
    """Carga opt-in: el CLI llega acá; los tests inyectan dobles y no importan torch."""
    import torch

    random.seed(config.semilla)
    torch.manual_seed(config.semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.semilla)
    from .modelo import ModeloLocal
    return ModeloLocal.cargar(config.configuracion_modelo())


def _hashes_artefactos(salida: Path) -> dict:
    nombres = ("trazas.jsonl", "tiempos.json", "resultados.jsonl", "resumen.json", "metricas.json")
    return {nombre: _sha((salida / nombre).read_bytes()) for nombre in nombres}


def ejecutar_evaluacion_prompt(*, prompt, configuracion, casos, esperadas, normas, salida,
                               modelo_lote=None, tokenizer=None) -> dict:
    """Ejecuta un candidato sin exponer las esperadas durante ``jugar``.

    La carpeta de salida se reserva antes de cargar el modelo y nunca se reutiliza. Un
    modelo/tokenizer inyectado habilita tests CPU; sin ellos se carga la revisión fijada
    y se requiere el entorno GPU del proyecto.
    """
    salida = Path(salida)
    if salida.exists() or salida.is_symlink():
        raise ValueError("la carpeta de salida ya existe")
    if not salida.parent.is_dir():
        raise ValueError("la carpeta padre de salida debe existir")
    salida.mkdir()
    entradas = salida / "entradas"
    entradas.mkdir()
    codigo = None
    config = spec = None
    hashes_entrada = {}
    try:
        codigo = _codigo()
        prompt_bytes = _copiar_archivo(prompt, entradas / "prompt.json")
        config_bytes = _copiar_archivo(configuracion, entradas / "configuracion.json")
        casos_bytes = _copiar_archivo(casos, entradas / "casos.jsonl")
        hashes_entrada.update({
            "prompt.json": _sha(prompt_bytes),
            "configuracion.json": _sha(config_bytes),
            "casos.jsonl": _sha(casos_bytes),
        })
        hashes_entrada["normas"] = _copiar_normas(normas, entradas / "normas")
        spec = PromptSpec.cargar(entradas / "prompt.json")
        config = ConfiguracionEvaluacionPrompt.cargar(entradas / "configuracion.json")
        casos_cargados = cargar_casos(entradas / "casos.jsonl")
        ids = [caso.id for caso in casos_cargados]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("los casos deben tener ids únicos y no estar vacíos")

        if (modelo_lote is None) != (tokenizer is None):
            raise ValueError("modelo_lote y tokenizer deben inyectarse juntos")
        if modelo_lote is None:
            modelo_lote = _cargar_modelo_real(config)
            tokenizer = modelo_lote.procesador
        longitud = medir_longitud_prompt(spec, tokenizer)
        entrega = _EntregaCandidata(spec, config, modelo_lote)

        # En este punto no existe ningún snapshot de esperadas en la corrida.
        trazas = jugar(entrega, casos_cargados, entradas / "normas", salida,
                       ruta_casos=entradas / "casos.jsonl")
        tiempos = json.loads((salida / "tiempos.json").read_text(encoding="utf-8"))

        # Se lee una sola vez después de jugar; esta copia exacta es la que se puntúa.
        esperadas_bytes = _copiar_archivo(esperadas, entradas / "esperadas.jsonl")
        hashes_entrada["esperadas.jsonl"] = _sha(esperadas_bytes)
        esperadas_cargadas = cargar_esperadas(entradas / "esperadas.jsonl")
        if set(esperadas_cargadas) != set(ids):
            raise ValueError("casos y esperadas deben tener exactamente los mismos ids")
        resultados, resumen = puntuar(trazas, esperadas_cargadas, salida, tiempos)
        metricas = calcular_metricas_candidato(resultados, trazas, longitud)
        (salida / "metricas.json").write_bytes(_canonical(metricas))

        manifiesto = {
            "version": 1,
            "estado": "completada" if tiempos.get("error") is None else "completada_con_error",
            "codigo": codigo,
            "prompt": {"nombre": spec.nombre, "version": spec.version, "sha256": spec.sha256},
            "configuracion": config.como_dict(),
            "entradas": hashes_entrada,
            "artefactos": _hashes_artefactos(salida),
            "metricas": metricas,
        }
        (salida / "manifiesto.json").write_bytes(_canonical(manifiesto))
        return manifiesto
    except Exception as error:
        # Un intento fallido también queda visible y nunca se reutiliza.
        if not (entradas / "esperadas.jsonl").exists():
            try:
                esperadas_bytes = _copiar_archivo(esperadas, entradas / "esperadas.jsonl")
                hashes_entrada["esperadas.jsonl"] = _sha(esperadas_bytes)
            except (OSError, ValueError):
                pass
        fallo = {
            "version": 1, "estado": "fallida", "codigo": codigo,
            "prompt": ({"nombre": spec.nombre, "version": spec.version, "sha256": spec.sha256}
                       if spec is not None else None),
            "configuracion": config.como_dict() if config is not None else None,
            "entradas": hashes_entrada,
            "error": {"tipo": type(error).__name__, "mensaje": str(error)},
        }
        try:
            (salida / "manifiesto.json").write_bytes(_canonical(fallo))
        except (OSError, TypeError, ValueError):
            pass
        raise


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--configuracion", type=Path, required=True)
    parser.add_argument("--casos", type=Path, required=True)
    parser.add_argument("--esperadas", type=Path, required=True)
    parser.add_argument("--normas", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifiesto = ejecutar_evaluacion_prompt(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=__import__("sys").stderr)
        return 1
    print(json.dumps({"estado": manifiesto["estado"], "prompt": manifiesto["prompt"],
                      "metricas": manifiesto["metricas"]}, ensure_ascii=False, indent=2))
    return 0
