"""Validador CPU del contrato de conversaciones curadas v1."""

import argparse
import json

from admision_tp.conversaciones import validar_archivo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo")
    args = parser.parse_args()
    try:
        informe = validar_archivo(args.archivo)
    except OSError:
        parser.exit(2, "No se pudo leer el archivo.\n")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return int(bool(informe["errores"]))


if __name__ == "__main__":
    raise SystemExit(main())
