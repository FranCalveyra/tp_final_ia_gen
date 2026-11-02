"""Congela datos y configuración de una futura corrida SFT, sin entrenar."""

import argparse
import json

from admision_tp.preparacion import preparar_corrida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", required=True)
    parser.add_argument("--configuracion", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()
    try:
        manifiesto = preparar_corrida(args.datos, args.configuracion, args.salida)
    except (ValueError, OSError) as error:
        mensaje = str(error) if type(error) is ValueError else "no se pudo escribir la preparación"
        parser.exit(1, mensaje + "\n")
    print(json.dumps({"estado": manifiesto["estado"], "dataset": manifiesto["dataset"],
                      "archivos": manifiesto["archivos"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
