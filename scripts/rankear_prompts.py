"""Ordena candidatos de prompt a partir de corridas descargadas de `scripts.evaluar_prompts`.

    python -m scripts.rankear_prompts corridas/prompt-dev/<sesion> [otra corrida o carpeta ...] [--json]

Criterio: más casos resueltos (9/9) sumando todas las carpetas de normas del conjunto de desarrollo,
y a igualdad menos tokens del mensaje de sistema (tokenizer real, medido por el runner). Rechaza
corridas no comparables (fallidas, con error o de plomería) y conjuntos incompletos: no las omite.
"""

import argparse
import json
from pathlib import Path
import sys

from admision_tp.seleccion_prompts import cargar_corrida, corridas_en, rankear, tabla


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rutas", type=Path, nargs="+", help="corridas o carpetas que las contienen")
    parser.add_argument("--json", action="store_true", help="salida JSON en lugar de tabla")
    args = parser.parse_args(argv)
    try:
        filas = rankear([cargar_corrida(ruta) for ruta in corridas_en(args.rutas)])
    except (OSError, ValueError, KeyError) as error:
        print(f"rechazado: {error}", file=sys.stderr)
        return 1
    print(json.dumps(filas, ensure_ascii=False, indent=2) if args.json else tabla(filas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
