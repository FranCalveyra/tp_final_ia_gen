"""Antepone a la decisión final los dos hechos de la nota de triage que determinan el aviso (#15).

El adapter no aprende `aviso_al_voluntariado` como un único token true/false. Con este formato el
asistente primero copia la edad y el acompañante de la nota y recién después escribe el JSON:

    Edad: 88 años. Acompañante: ninguno.
    {"herramienta": "registrar_decision", ...}

Los valores se copian textualmente de la observación de `leer_triage`; el umbral no se escribe en
ningún lado. Solo cambia el último mensaje; el resto de cada registro queda idéntico.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

from admision_tp.preparacion import _canonical, _codigo, _sha

EDAD = re.compile(r"^Paciente: .*, (\d+) años\.$", re.M)
ACOMPANANTE = re.compile(r"^Acompañante: (.+)\.$", re.M)  # el valor puede contener puntos y paréntesis


def _nota(registro: dict) -> str:
    for mensaje in registro["messages"]:
        if mensaje["role"] == "user" and "Nota de triage" in mensaje["content"]:
            return mensaje["content"]
    raise ValueError(f"{registro['id']}: sin nota de triage")


def anteponer(registro: dict) -> dict:
    """Copia del registro con la línea de hechos antepuesta a la decisión final."""
    nota = _nota(registro)
    edad, acompanante = EDAD.search(nota), ACOMPANANTE.search(nota)
    if not edad or not acompanante:
        raise ValueError(f"{registro['id']}: la nota no tiene edad o acompañante")
    ultimo = registro["messages"][-1]["content"]
    if not ultimo.startswith("{") or json.loads(ultimo).get("herramienta") != "registrar_decision":
        raise ValueError(f"{registro['id']}: no termina en registrar_decision sin prefijo")
    prefijo = f"Edad: {edad.group(1)} años. Acompañante: {acompanante.group(1)}.\n"
    if "{" in prefijo or "}" in prefijo:
        raise ValueError(f"{registro['id']}: el prefijo contiene llaves")
    nuevo = copy.deepcopy(registro)
    nuevo["messages"][-1]["content"] = prefijo + ultimo
    return nuevo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, required=True, help="conversaciones.jsonl de origen")
    parser.add_argument("--salida", type=Path, required=True, help="carpeta nueva")
    args = parser.parse_args(argv)
    if args.salida.exists():
        raise ValueError("la salida ya existe; no se sobrescribe")
    contenido = args.datos.read_bytes()
    registros = [json.loads(x) for x in contenido.decode("utf-8").splitlines() if x.strip()]
    salida = "".join(json.dumps(anteponer(r), ensure_ascii=False) + "\n" for r in registros).encode("utf-8")
    git = _codigo()  # Antes de crear la salida: si no, la carpeta nueva marca el repo como sucio.
    args.salida.mkdir(parents=True)
    (args.salida / "conversaciones.jsonl").write_bytes(salida)
    manifiesto = {"version": 1, "proposito": "antepone edad y acompañante a la decisión final para el aviso (#15)",
                  "comando": "python -m scripts.anteponer_datos_aviso " + " ".join(sys.argv[1:] if argv is None else argv),
                  "git": git, "origen": {"ruta": str(args.datos), "sha256": _sha(contenido), "registros": len(registros)},
                  "registros": len(registros), "archivos_sha256": {"conversaciones.jsonl": _sha(salida)}}
    (args.salida / "manifiesto.json").write_bytes(_canonical(manifiesto))
    print(json.dumps({"registros": len(registros)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
