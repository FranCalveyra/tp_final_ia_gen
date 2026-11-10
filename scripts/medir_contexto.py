"""Mide offline cuánto contexto (`caracteres_de_contexto` del kit) cuesta cada PromptSpec.

Rejuega trazas de `admision.correr` con `AgentePrompt` (ver `admision_tp.contexto`): mismas
respuestas y observaciones, distinto prompt. Antes de comparar, verifica que el prompt de origen
reproduzca exactamente cada conversación grabada. No carga modelo ni usa GPU.

    python -m scripts.medir_contexto --trazas corridas/.../trazas.jsonl \\
        --prompt configuraciones/prompts/baseline.json --prompt otro.json
"""

import argparse
import json
from pathlib import Path
import statistics

from admision.correr import caracteres_de_contexto
from admision_tp.contexto import PARTES, desglose, rejugar_traza
from admision_tp.prompts import RUTA_BASELINE, PromptSpec


def medir(trazas: list[dict], prompts: list[PromptSpec], origen: PromptSpec) -> dict:
    base = []
    for numero, traza in enumerate(trazas, 1):
        if rejugar_traza(traza, origen, origen) != traza["mensajes"]:
            raise ValueError(f"traza {numero}: el prompt de origen no reproduce la conversación grabada")
        base.append(caracteres_de_contexto(traza["mensajes"]))
    resultado = {"casos": len(trazas), "origen": origen.nombre, "prompts": {}}
    for prompt in prompts:
        conversaciones = [rejugar_traza(traza, prompt, origen) for traza in trazas]
        valores = [caracteres_de_contexto(c) for c in conversaciones]
        partes = {parte: sum(desglose(c, prompt)[parte] for c in conversaciones) for parte in PARTES}
        ahorros = [1 - v / b for v, b in zip(valores, base) if b]
        resultado["prompts"][prompt.nombre] = {
            "sha256": prompt.sha256,
            "caracteres_por_caso": round(statistics.mean(valores)),
            "ahorro_total": round(1 - sum(valores) / sum(base), 4),
            "ahorro_por_caso_min": round(min(ahorros), 4),
            "ahorro_por_caso_max": round(max(ahorros), 4),
            "participacion": {k: round(v / sum(valores), 4) for k, v in partes.items()},
        }
    return resultado


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trazas", type=Path, required=True, nargs="+", help="trazas.jsonl de admision.correr")
    parser.add_argument("--prompt", type=Path, action="append", help="PromptSpec a medir (repetible)")
    parser.add_argument("--origen", type=Path, default=RUTA_BASELINE, help="PromptSpec con el que se jugaron")
    args = parser.parse_args(argv)
    origen = PromptSpec.cargar(args.origen)
    prompts = [PromptSpec.cargar(ruta) for ruta in (args.prompt or [args.origen])]
    informe = {}
    for ruta in args.trazas:
        trazas = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
        informe[str(ruta)] = medir(trazas, prompts, origen)
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
