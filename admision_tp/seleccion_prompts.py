"""Ranking de candidatos de prompt sobre el conjunto de desarrollo (issue #13).

Lee corridas de `scripts.evaluar_prompts` (una por prompt y carpeta de normas), rechaza las no
comparables y ordena los prompts por casos resueltos (suma sobre carpetas) y, a igualdad, por
tokens del mensaje de sistema medidos por el runner con el tokenizer real.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from admision.comparar import CONTROLES

from .evaluacion_prompts import clave_de_manifiesto
from .preparacion import _canonical, _sha


def _leer_json(ruta: Path) -> dict:
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{ruta}: no se pudo leer") from None


def cargar_corrida(ruta: Path) -> dict:
    """Resumen de una corrida; `ValueError` si no es comparable o sus artefactos no coinciden."""
    ruta = Path(ruta)
    manifiesto = _leer_json(ruta / "manifiesto.json")
    try:
        clave_de_manifiesto(manifiesto)
    except ValueError as error:
        raise ValueError(f"{ruta}: {error}") from None
    contenido = (ruta / "resultados.jsonl").read_bytes()
    if _sha(contenido) != manifiesto.get("artefactos", {}).get("resultados.jsonl"):
        raise ValueError(f"{ruta}: resultados.jsonl no coincide con el hash del manifiesto")
    resultados = [json.loads(linea) for linea in contenido.decode("utf-8").splitlines() if linea.strip()]
    metricas, entradas = manifiesto["metricas"], manifiesto["entradas"]
    resueltos = sum(all(r["controles"][c] for c in CONTROLES) for r in resultados)
    if resueltos != metricas["resueltos_9_de_9"] or len(resultados) != metricas["casos"]:
        raise ValueError(f"{ruta}: resultados y métricas no coinciden")
    conjunto = {k: entradas[k] for k in ("casos.jsonl", "esperadas.jsonl", "normas")}
    return {
        "ruta": str(ruta),
        "prompt": manifiesto["prompt"],
        "configuracion": manifiesto["configuracion"],
        "commit": manifiesto.get("codigo", {}).get("commit"),
        "conjunto": _sha(_canonical(conjunto)),
        "casos": len(resultados),
        "resueltos": resueltos,
        "controles": {c: sum(bool(r["controles"][c]) for r in resultados) for c in CONTROLES},
        "tokens_system": metricas["longitud_prompt"]["por_texto"]["system_prompt"],
        "chat_template": metricas["longitud_prompt"]["chat_template_sha256"],
    }


def rankear(corridas: list[dict]) -> list[dict]:
    """Ordena prompts por (más resueltos, menos tokens de sistema, nombre).

    Exige la misma configuración y plantilla de chat en todas las corridas, y que cada prompt
    tenga exactamente una corrida por cada conjunto (carpeta de normas + casos + esperadas).
    """
    if not corridas:
        raise ValueError("no hay corridas")
    for campo in ("configuracion", "chat_template"):
        if len({json.dumps(c[campo], sort_keys=True) for c in corridas}) != 1:
            raise ValueError(f"las corridas difieren en {campo}: no son comparables")
    conjuntos = {c["conjunto"] for c in corridas}
    por_prompt = defaultdict(list)
    for corrida in corridas:
        por_prompt[corrida["prompt"]["sha256"]].append(corrida)
    filas = []
    for sha, grupo in por_prompt.items():
        vistos = [c["conjunto"] for c in grupo]
        if len(vistos) != len(set(vistos)) or set(vistos) != conjuntos:
            raise ValueError(f"el prompt {grupo[0]['prompt']['nombre']} no tiene exactamente una corrida "
                             f"por conjunto ({len(set(vistos))} de {len(conjuntos)}, {len(vistos)} corridas)")
        filas.append({
            "nombre": grupo[0]["prompt"]["nombre"],
            "sha256": sha,
            "resueltos": sum(c["resueltos"] for c in grupo),
            "casos": sum(c["casos"] for c in grupo),
            "tokens_system": grupo[0]["tokens_system"],
            "por_conjunto": {c["conjunto"][:12]: c["resueltos"] for c in sorted(grupo, key=lambda c: c["conjunto"])},
            "controles": {k: sum(c["controles"][k] for c in grupo) for k in CONTROLES},
            "commits": sorted({c["commit"] for c in grupo if c["commit"]}),
            "corridas": sorted(c["ruta"] for c in grupo),
        })
    filas.sort(key=lambda f: (-f["resueltos"], f["tokens_system"], f["nombre"]))
    for posicion, fila in enumerate(filas, 1):
        fila["posicion"] = posicion
    return filas


def corridas_en(rutas: list[Path]) -> list[Path]:
    """Cada ruta es una corrida (tiene manifiesto.json) o una carpeta con corridas como hijas directas."""
    encontradas = []
    for ruta in map(Path, rutas):
        if (ruta / "manifiesto.json").is_file():
            encontradas.append(ruta)
        else:
            hijas = sorted(p for p in ruta.iterdir() if (p / "manifiesto.json").is_file()) if ruta.is_dir() else []
            if not hijas:
                raise ValueError(f"{ruta}: no es una corrida ni contiene corridas")
            encontradas.extend(hijas)
    return encontradas


def tabla(filas: list[dict]) -> str:
    cabecera = ["#", "prompt", "resueltos", "tokens sistema", *CONTROLES]
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    for f in filas:
        valores = [f["posicion"], f["nombre"], f"{f['resueltos']}/{f['casos']}", f["tokens_system"],
                   *(f"{f['controles'][c]}/{f['casos']}" for c in CONTROLES)]
        lineas.append("| " + " | ".join(map(str, valores)) + " |")
    return "\n".join(lineas)
