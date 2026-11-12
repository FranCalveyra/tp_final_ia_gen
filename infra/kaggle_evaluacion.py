"""Evaluación completa en Kaggle: juega un plan (configuraciones × normas) con `kaggle_baseline.correr`.

Una celda de notebook:

    python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/actual.json \\
        --salida /kaggle/working/corridas/evaluacion-<fecha>

Valida el plan completo (prompts con su sha256, adapters con revisión inmutable, etiquetas únicas)
**antes** de crear la salida o tocar la GPU. Cada (configuración × versión de normas) corre en su
propia carpeta nueva, con su propio entorno; si una falla, se registra y siguen las demás.
Puntuar y armar la tabla es una fase local separada: `python -m scripts.tabla_evaluacion`.
Ver docs/evaluacion.md.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra import kaggle_baseline as kb  # noqa: E402

RAIZ = kb.RAIZ
ETIQUETA = re.compile(r"[a-z][a-z0-9_]{0,39}")
# Variables que define el plan. Se borran del entorno heredado para que ninguna corrida herede las de otra
# (por ejemplo, un ADAPTER_REPO del notebook en la fila sin entrenar) ni cambie el modelo base fijado.
CONTROLADAS = ("PROMPT_SPEC", "MODELO_LOTE", "BUSCADOR_VARIANTE", "ADAPTER_REPO", "ADAPTER_REVISION",
               "ADAPTER_SUBCARPETA", "MODELO_ID", "MODELO_REVISION")
_CAMPOS_PLAN = {"version", "nombre", "descripcion", "plantilla", "cantidad", "versiones", "lote",
                "buscador", "configuraciones"}
_CAMPOS_CONFIG = {"etiqueta", "descripcion", "prompt", "prompt_sha256", "adapter", "versiones", "cantidad"}
_CAMPOS_ADAPTER = {"repo", "revision", "subcarpeta"}


@dataclass(frozen=True)
class Corrida:
    etiqueta: str
    version: str
    cantidad: int
    prompt: dict
    adapter: dict | None
    variables: dict = field(default_factory=dict)

    @property
    def carpeta(self) -> str:
        return f"{self.etiqueta}-{self.version}"

    def entorno(self, base) -> dict:
        return {**{k: v for k, v in base.items() if k not in CONTROLADAS}, **self.variables}


def validar_plan(dato: dict, dtype: str = "float16", device_map: str = "auto", raiz: Path = RAIZ) -> list[Corrida]:
    """Devuelve las corridas del plan o levanta ValueError con todos los problemas juntos."""
    from admision_tp.buscador import VARIANTES as BUSCADORES
    from admision_tp.configuracion import Configuracion
    from admision_tp.prompts import PromptSpec

    if not isinstance(dato, dict):
        raise ValueError("el plan debe ser un objeto JSON")
    problemas = []
    if dato.get("plantilla"):
        problemas.append("es una plantilla: copiala, completá prompts/adapters reales y sacá `plantilla`")
    if dato.get("version") != 1:
        problemas.append("version del plan debe ser 1")
    if extra := set(dato) - _CAMPOS_PLAN:
        problemas.append(f"campos desconocidos en el plan: {sorted(extra)}")
    buscador = dato.get("buscador", "baseline")
    if buscador not in BUSCADORES:
        problemas.append(f"buscador debe ser uno de: {', '.join(BUSCADORES)}")
    lote = dato.get("lote", 1)
    configs = dato.get("configuraciones")
    if not isinstance(configs, list) or not configs:
        raise ValueError("; ".join(problemas + ["configuraciones debe ser una lista no vacía"]))
    corridas, etiquetas = [], set()
    for i, config in enumerate(configs):
        nombre = f"configuraciones[{i}]"
        if not isinstance(config, dict):
            problemas.append(f"{nombre}: debe ser un objeto")
            continue
        if extra := set(config) - _CAMPOS_CONFIG:
            problemas.append(f"{nombre}: campos desconocidos {sorted(extra)}")
        etiqueta = config.get("etiqueta")
        if not isinstance(etiqueta, str) or not ETIQUETA.fullmatch(etiqueta):
            problemas.append(f"{nombre}: etiqueta inválida (minúsculas, dígitos y _)")
            continue
        if etiqueta in etiquetas:
            problemas.append(f"etiqueta repetida: {etiqueta}")
        etiquetas.add(etiqueta)
        nombre = etiqueta
        versiones = config.get("versiones", dato.get("versiones", list(kb.VARIANTES)))
        if (not isinstance(versiones, list) or not versiones or len(set(versiones)) != len(versiones)
                or any(v not in kb.VARIANTES for v in versiones)):
            problemas.append(f"{nombre}: versiones debe ser una lista sin repetir de {kb.VARIANTES}")
            versiones = []
        cantidad = config.get("cantidad", dato.get("cantidad", 50))
        if type(cantidad) is not int or not 1 <= cantidad <= 50:
            problemas.append(f"{nombre}: cantidad entre 1 y 50")
        ruta = config.get("prompt")
        if not isinstance(ruta, str) or not ruta or Path(ruta).is_absolute():
            problemas.append(f"{nombre}: prompt debe ser una ruta relativa a la raíz del repo")
            continue
        try:
            spec = PromptSpec.cargar(raiz / ruta)  # Valida el sha256 interno del archivo.
        except ValueError as e:
            problemas.append(f"{nombre}: prompt {ruta}: {e}")
            continue
        if config.get("prompt_sha256") != spec.sha256:
            problemas.append(f"{nombre}: prompt_sha256 no coincide con {ruta} ({spec.sha256})")
        adapter = config.get("adapter")
        # Sin adapter, ADAPTER_* vacías: ausentes, Configuracion usaría el adapter de la entrega.
        variables = {"PROMPT_SPEC": ruta, "MODELO_LOTE": str(lote), "BUSCADOR_VARIANTE": buscador,
                     "ADAPTER_REPO": "", "ADAPTER_REVISION": "", "ADAPTER_SUBCARPETA": ""}
        if adapter is not None:
            if (not isinstance(adapter, dict) or set(adapter) - _CAMPOS_ADAPTER
                    or not {"repo", "revision"} <= set(adapter)):
                problemas.append(f"{nombre}: adapter requiere repo y revision (y opcional subcarpeta)")
                continue
            adapter = {"subcarpeta": "final", **adapter}
            variables |= {"ADAPTER_REPO": adapter["repo"], "ADAPTER_REVISION": adapter["revision"],
                          "ADAPTER_SUBCARPETA": adapter["subcarpeta"]}
        try:  # Las mismas reglas que aplicará el agente: revisión de 40 hex, lote positivo, dtype.
            Configuracion(lote=lote if type(lote) is int else 0, dtype=dtype, device_map=device_map,
                          adapter_repo=(adapter or {}).get("repo"),
                          adapter_revision=(adapter or {}).get("revision"),
                          adapter_subcarpeta=(adapter or {}).get("subcarpeta", "final"))
        except (TypeError, ValueError) as e:
            problemas.append(f"{nombre}: {e}")
            continue
        descriptor = {"archivo": ruta, "nombre": spec.nombre, "version": spec.version, "sha256": spec.sha256}
        corridas += [Corrida(etiqueta, v, cantidad, descriptor, adapter, variables) for v in versiones]
    if problemas:
        raise ValueError("plan inválido:\n- " + "\n- ".join(problemas))
    return corridas


def ejecutar_plan(corridas: list[Corrida], destino: Path, dtype: str, device_map: str, token: str | None,
                  registro: dict, correr=None) -> int:
    """Corre cada corrida en `destino/<etiqueta>-<version>`; anota todo en `destino/evaluacion.json`."""
    correr = correr or kb.correr
    if not token:
        raise RuntimeError("Falta HF_TOKEN (variable de entorno o Kaggle Secret)")
    kb.crear_salida(destino)
    registro = {**registro, "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dtype": dtype,
                "device_map": device_map, "corridas": [
                    {"etiqueta": c.etiqueta, "version": c.version, "cantidad": c.cantidad, "carpeta": c.carpeta,
                     "prompt": c.prompt, "adapter": c.adapter, "variables": c.variables, "estado": "pendiente"}
                    for c in corridas]}
    escribir = lambda: (destino / "evaluacion.json").write_text(  # noqa: E731
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir()
    fallidas = 0
    for corrida, fila in zip(corridas, registro["corridas"]):
        fila["estado"] = "en_curso"
        escribir()
        try:
            with kb.patch_entorno(corrida.entorno(os.environ)):
                codigo = correr(corrida.version, corrida.cantidad, destino / corrida.carpeta, dtype, device_map,
                                token)
            error = None
        except Exception as e:  # Una corrida rota no frena las demás; KeyboardInterrupt sí corta.
            codigo, error = None, f"{type(e).__name__}: {e}".replace(token, "***")
        fila.update(estado="terminada" if codigo == 0 else "fallida", returncode=codigo, error=error)
        fallidas += codigo != 0
        escribir()
    registro["fin"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    escribir()
    return 1 if fallidas else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True, help=f"Carpeta nueva, p. ej. en {kb.SALIDA_BASE}")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--validar", action="store_true", help="solo valida el plan; no usa GPU ni token")
    args = parser.parse_args(argv)
    contenido = args.plan.read_bytes()
    corridas = validar_plan(json.loads(contenido), args.dtype, args.device_map)
    for c in corridas:
        print(f"{c.carpeta}: {c.cantidad} casos, prompt {c.prompt['nombre']} ({c.prompt['sha256'][:12]}), "
              f"adapter {'sin entrenar' if not c.adapter else c.adapter['repo'] + '@' + c.adapter['revision']}")
    if args.validar:
        return 0
    if args.salida.exists():
        raise FileExistsError("Elegí una salida nueva para no pisar una corrida")
    token = os.environ.get("HF_TOKEN") or kb.obtener_secreto("HF_TOKEN")
    registro = {"plan": json.loads(contenido), "plan_archivo": str(args.plan),
                "plan_sha256": hashlib.sha256(contenido).hexdigest(),
                "comando": ["python", "infra/kaggle_evaluacion.py", *(argv if argv is not None else sys.argv[1:])],
                "maquina": kb.describir_maquina()}
    codigo = ejecutar_plan(corridas, args.salida, args.dtype, args.device_map, token, registro)
    print(f"returncode={codigo}; resultados en {args.salida}. Puntuar y armar la tabla es una fase local.")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
