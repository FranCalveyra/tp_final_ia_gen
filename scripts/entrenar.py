"""Entrenamiento CUDA explícito. Nunca se invoca desde el agente operativo."""

import argparse
import json
import traceback
from admision_tp.entrenamiento import DISPOSITIVOS, PRECISIONES, entrenar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparacion", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--reanudar", help="checkpoint-N local completo dentro de salida")
    p.add_argument("--online", action="store_true", help="autoriza escribir métricas en W&B")
    p.add_argument("--publicar", action="store_true", help="autoriza subir checkpoints a HF privado")
    p.add_argument("--precision", choices=sorted(PRECISIONES), default="bf16",
                   help="bf16 (A100, default) o fp16 (T4, sin bf16 nativo)")
    p.add_argument("--dispositivos", choices=DISPOSITIVOS, default="unico",
                   help="unico (una GPU, default) o auto (reparte el modelo entre GPU)")
    p.add_argument("--guardar-cada", type=int, default=50, help="pasos entre checkpoints (default 50)")
    a = p.parse_args()
    try:
        resultado = entrenar(a.preparacion, a.tokens, a.salida, checkpoint=a.reanudar,
                             online=a.online, publicar=a.publicar, precision=a.precision,
                             dispositivos=a.dispositivos, guardar_cada=a.guardar_cada)
    except Exception as error:
        # estado.json (persistido y publicable) guarda solo el tipo; el traceback va a stderr para poder
        # diagnosticar. En Kaggle pasa por el runner, que tapa los secretos antes de escribir el log.
        traceback.print_exc()
        p.exit(1, f"Entrenamiento no completado ({type(error).__name__}); revisar manifiestos y estado local.\n")
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
