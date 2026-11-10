"""Validador CPU del contrato de conversaciones curadas v1."""

import argparse
import json

from admision_tp.conversaciones import FORMATO_OBSERVACION, validar_archivo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo")
    parser.add_argument("--prompt", help="PromptSpec JSON cuyo formato de observación se exige "
                                         "(por defecto, el de AgenteJSON)")
    args = parser.parse_args()
    formato = FORMATO_OBSERVACION
    if args.prompt:
        from admision_tp.prompts import PromptSpec
        try:
            formato = PromptSpec.cargar(args.prompt).formato_observacion
        except ValueError as error:
            parser.exit(2, f"{error}\n")
    try:
        informe = validar_archivo(args.archivo, formato)
    except OSError:
        parser.exit(2, "No se pudo leer el archivo.\n")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return int(bool(informe["errores"]))


if __name__ == "__main__":
    raise SystemExit(main())
