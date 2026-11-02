"""Entrenamiento CUDA explícito. Nunca se invoca desde el agente operativo."""

import argparse
import json
from admision_tp.entrenamiento import entrenar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparacion", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--reanudar", help="checkpoint-N local completo dentro de salida")
    p.add_argument("--online", action="store_true", help="autoriza escribir métricas en W&B")
    p.add_argument("--publicar", action="store_true", help="autoriza subir checkpoints a HF privado")
    a = p.parse_args()
    try:
        resultado = entrenar(a.preparacion, a.tokens, a.salida, checkpoint=a.reanudar,
                             online=a.online, publicar=a.publicar)
    except Exception as error:
        p.exit(1, f"Entrenamiento no completado ({type(error).__name__}); revisar manifiestos y estado local.\n")
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
