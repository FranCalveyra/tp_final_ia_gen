"""Convierte trazas a conversaciones v1 sin curar datos ni seleccionar prompt."""

import argparse
import json
from pathlib import Path

from admision_tp.conversion_trazas import convertir_archivo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--origen", required=True, choices=("historial", "sintetico"))
    parser.add_argument("--sistema", required=True, help="archivo UTF-8 con el prompt elegido explícitamente")
    args = parser.parse_args()
    try:
        sistema = Path(args.sistema).read_text(encoding="utf-8")
        informe = convertir_archivo(args.entrada, args.salida, origen=args.origen, sistema=sistema)
    except (ValueError, OSError) as error:
        mensaje = str(error) if type(error) is ValueError else "no se pudo leer la entrada o crear una salida nueva"
        parser.exit(1, mensaje + "\n")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
