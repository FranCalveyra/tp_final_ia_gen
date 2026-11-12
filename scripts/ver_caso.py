"""Muestra un caso jugado: la conversación, las búsquedas en las normas, la ficha registrada y los 9 controles.

Lee la carpeta que deja `admision.correr practica` (`trazas.jsonl` y, si está, `resultados.jsonl`):

    python -m scripts.ver_caso corridas/un-caso/b [--caso pev-001] [--sistema-completo]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _leer(ruta: Path) -> list[dict]:
    return [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip()]


def _secciones(observacion: str) -> str:
    titulos = [linea[3:] for linea in observacion.splitlines() if linea.startswith("## ")]
    if not observacion:
        return "(vacío)"
    return "; ".join(titulos) or observacion.splitlines()[0]


def mostrar(traza: dict, resultado: dict | None, sistema_completo: bool) -> str:
    lineas = [f"# Caso {traza['caso']}", "", "## Conversación"]
    for mensaje in traza["mensajes"]:
        texto = mensaje["content"]
        if mensaje["role"] == "system" and not sistema_completo:
            texto = f"{texto.splitlines()[0]} ({len(texto)} caracteres; --sistema-completo lo muestra entero)"
        lineas.append(f"[{mensaje['role']}] {texto}")
    lineas += ["", "## Búsquedas en las normas (consulta → secciones que trajo el buscador)"]
    busquedas = [x for x in traza["log"] if x["herramienta"] == "buscar_normativa"]
    lineas += [f"- «{x['argumentos'].get('consulta')}» → {_secciones(x['observacion'])}" for x in busquedas]
    lineas += [] if busquedas else ["(ninguna)"]
    lineas += ["", "## Ficha registrada", json.dumps(traza["decision"], ensure_ascii=False, indent=2)]
    lineas += ["", "## Controles"]
    if resultado is None:
        lineas.append("(sin resultados.jsonl: falta puntuar)")
    else:
        controles = resultado["controles"]
        lineas += [f"{'✓' if ok else '✗'} {nombre}" for nombre, ok in controles.items()]
        estado = "Resuelto" if resultado["resuelto"] else "No resuelto"
        lineas.append(f"{estado} ({sum(controles.values())}/{len(controles)} controles)")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("carpeta", type=Path, help="salida de admision.correr practica")
    parser.add_argument("--caso", help="id del caso (por defecto, el primero)")
    parser.add_argument("--sistema-completo", action="store_true", help="mostrar el mensaje de sistema entero")
    args = parser.parse_args(argv)
    trazas = _leer(args.carpeta / "trazas.jsonl")
    elegidas = [t for t in trazas if args.caso in (None, t["caso"])]
    if not elegidas:
        parser.error(f"no hay caso {args.caso} en {args.carpeta / 'trazas.jsonl'}")
    traza = elegidas[0]
    ruta = args.carpeta / "resultados.jsonl"
    resultado = next((r for r in _leer(ruta) if r["caso"] == traza["caso"]), None) if ruta.exists() else None
    print(mostrar(traza, resultado, args.sistema_completo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
