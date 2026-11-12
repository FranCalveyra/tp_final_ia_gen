"""Subconjunto balanceado para seguir entrenando el adapter en el aviso al voluntariado (#15).

El adapter entrenado sobre `v3-procedimiento` nunca responde aviso `true` (10 % de positivos): el
conjunto de desarrollo lo confirmó (Kaggle v35). Este subconjunto toma todas las conversaciones con
aviso `true`, todas las negativas del mismo rango de edad (las que enseñan que la condición no es
solo la edad) y una muestra fija del resto, para no olvidar los demás casos.

El umbral no se escribe acá: sale de los datos (la menor edad entre los positivos). Copia las líneas
tal cual; no cambia ninguna conversación.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

from admision_tp.preparacion import _canonical, _codigo, _sha

EDAD = re.compile(r"^Paciente: .*, (\d+) años\.$", re.M)


def _edad(registro: dict) -> int:
    for mensaje in registro["messages"]:
        if mensaje["role"] == "user" and "Nota de triage" in mensaje["content"]:
            encontrada = EDAD.search(mensaje["content"])
            if encontrada:
                return int(encontrada.group(1))
    raise ValueError(f"{registro['id']}: sin nota de triage con edad")


def _aviso(registro: dict) -> bool:
    accion = json.loads(registro["messages"][-1]["content"])
    if accion.get("herramienta") != "registrar_decision":
        raise ValueError(f"{registro['id']}: no termina en registrar_decision")
    return accion["argumentos"]["aviso_al_voluntariado"] is True


def armar_subconjunto(registros: list[dict], otros: int, semilla: int) -> tuple[list[dict], dict]:
    datos = [(r, _edad(r), _aviso(r)) for r in registros]
    positivos = [r for r, _, aviso in datos if aviso]
    if not positivos:
        raise ValueError("no hay positivos de aviso en los datos")
    minima = min(edad for _, edad, aviso in datos if aviso)
    dificiles = [r for r, edad, aviso in datos if not aviso and edad >= minima]
    resto = [r for r, edad, aviso in datos if not aviso and edad < minima]
    muestra = random.Random(semilla).sample(resto, min(otros, len(resto)))
    elegidos = {id(r) for r in positivos + dificiles + muestra}
    resumen = {"edad_minima_positivos": minima, "grupos": {
        "positivos": len(positivos), "negativos_mismo_rango_de_edad": len(dificiles),
        "otros_negativos": len(muestra)}}
    return [r for r in registros if id(r) in elegidos], resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, required=True, help="conversaciones.jsonl de origen")
    parser.add_argument("--salida", type=Path, required=True, help="carpeta nueva")
    parser.add_argument("--otros", type=int, required=True, help="negativos de menor edad a muestrear")
    parser.add_argument("--semilla", type=int, required=True)
    args = parser.parse_args(argv)
    if args.salida.exists():
        raise ValueError("la salida ya existe; no se sobrescribe")
    contenido = args.datos.read_bytes()
    lineas = [x for x in contenido.decode("utf-8").splitlines() if x.strip()]
    registros = [json.loads(x) for x in lineas]
    elegidos, resumen = armar_subconjunto(registros, args.otros, args.semilla)
    ids = {r["id"] for r in elegidos}
    salida = "".join(x + "\n" for x, r in zip(lineas, registros) if r["id"] in ids).encode("utf-8")
    args.salida.mkdir(parents=True)
    (args.salida / "conversaciones.jsonl").write_bytes(salida)
    manifiesto = {"version": 1, "proposito": "seguir entrenando el adapter en aviso_al_voluntariado (#15)",
                  "comando": "python -m scripts.subconjunto_aviso " + " ".join(sys.argv[1:] if argv is None else argv),
                  "git": _codigo(), "origen": {"ruta": str(args.datos), "sha256": _sha(contenido),
                                               "registros": len(registros)},
                  "semilla": args.semilla, "otros": args.otros, "registros": len(elegidos), **resumen,
                  "archivos_sha256": {"conversaciones.jsonl": _sha(salida)}}
    (args.salida / "manifiesto.json").write_bytes(_canonical(manifiesto))
    print(json.dumps({"registros": len(elegidos), **resumen}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
