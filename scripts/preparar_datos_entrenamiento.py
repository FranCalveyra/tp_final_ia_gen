"""Construye el dataset reproducible de conversaciones reales y sintéticas."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from admision_tp.conversaciones import validar_archivo
from scripts.generar_sinteticos_entrenamiento import generar_sinteticos
from scripts.seleccionar_entrenamiento import convertir_historial


def escribir_jsonl(ruta: Path, registros: list[dict]) -> None:
    ruta.write_text(
        "".join(json.dumps(registro, ensure_ascii=False, separators=(",", ":")) + "\n" for registro in registros),
        encoding="utf-8",
    )


def preparar(
    raiz: Path,
    salida: Path,
    semilla: int = 12012,
    candidatos: int = 5000,
) -> dict:
    if salida.exists():
        raise FileExistsError(f"la salida ya existe: {salida}")
    historial, resumen_historial = convertir_historial(
        raiz / "historial/historial.csv",
        raiz / "historial_clasificado.csv",
        raiz / "historial_costumbres.json",
        raiz / "historial/trazas.jsonl",
    )
    sinteticos, auditoria, resumen_sinteticos = generar_sinteticos(
        raiz / "normas/original",
        raiz / "historial_costumbres.json",
        semilla=semilla,
        candidatos=candidatos,
    )
    registros = historial + sinteticos
    salida.mkdir(parents=True)
    ruta_conversaciones = salida / "conversaciones.jsonl"
    ruta_auditoria = salida / "auditoria_sinteticos.jsonl"
    escribir_jsonl(ruta_conversaciones, registros)
    escribir_jsonl(ruta_auditoria, auditoria)
    validacion = validar_archivo(ruta_conversaciones)
    if validacion["errores"]:
        raise ValueError(f"dataset combinado inválido: {len(validacion['errores'])} errores")
    manifiesto = {
        "version": 1,
        "procedencia": {
            "practica_usada": False,
            "historial": [
                "historial/historial.csv",
                "historial_clasificado.csv",
                "historial/trazas.jsonl",
                "historial_costumbres.json",
            ],
            "sinteticos": ["admision.generador", "normas/original", "scripts/generar_sinteticos_entrenamiento.py"],
        },
        "reales": resumen_historial,
        "sinteticos": resumen_sinteticos,
        "total_conversaciones": len(registros),
        "por_origen": dict(Counter(registro["origen"] for registro in registros)),
        "validacion": {
            "registros": validacion["registros"],
            "validos": validacion["validos"],
            "sha256": validacion["sha256"],
        },
    }
    (salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifiesto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path("."))
    parser.add_argument("--salida", type=Path, default=Path("datos/entrenamiento"))
    parser.add_argument("--semilla", type=int, default=12012)
    parser.add_argument("--candidatos", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(preparar(args.raiz, args.salida, args.semilla, args.candidatos), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
