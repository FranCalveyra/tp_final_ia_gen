"""Convierte trazas a conversaciones v1 sin curar datos: conserva todas las observaciones.

Para el dataset de entrenamiento (búsquedas re-ejecutadas y fichas revalidadas) usar
`scripts.generar_conversaciones`.
"""

import argparse
import json
from pathlib import Path

from admision_tp.conversaciones import ORIGENES
from admision_tp.conversion_trazas import convertir_archivo
from admision_tp.prompts import PromptSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--origen", required=True, choices=ORIGENES)
    contexto = parser.add_mutually_exclusive_group(required=True)
    contexto.add_argument("--prompt", help="PromptSpec JSON (todos sus textos)")
    contexto.add_argument("--sistema", help="obsoleto: archivo UTF-8 con solo el mensaje de sistema; "
                                            "el resto de los textos sale del prompt baseline")
    args = parser.parse_args()
    try:
        if args.prompt:
            contexto = {"prompt": PromptSpec.cargar(args.prompt)}
        else:
            contexto = {"sistema": Path(args.sistema).read_text(encoding="utf-8")}
        informe = convertir_archivo(args.entrada, args.salida, origen=args.origen, **contexto)
    except (ValueError, OSError) as error:
        mensaje = str(error) if type(error) is ValueError else "no se pudo leer la entrada o crear una salida nueva"
        parser.exit(1, mensaje + "\n")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
