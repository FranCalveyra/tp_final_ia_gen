"""Tokeniza una preparación SFT en CPU; no carga pesos ni entrena."""

import argparse
import json
from admision_tp.tokenizacion import tokenizar_preparacion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparacion", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()
    try:
        resultado = tokenizar_preparacion(args.preparacion, args.salida)
    except (ValueError, OSError, ImportError):
        parser.exit(1, "No se pudo tokenizar: verificá snapshots, límite, plantilla y dependencias Transformers.\n")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
