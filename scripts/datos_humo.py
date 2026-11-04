"""Datos SINTÉTICOS para probar el pipeline de entrenamiento (humo). No son un dataset curado.

Genera una carpeta nueva con `conversaciones.jsonl` y `configuracion.json` para los CLI de
preparación y tokenización. Solo prueba estructura y ejecución: no representa calidad del modelo.
Nunca usa casos de práctica/test ni fichas esperadas.
"""

import argparse
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def registro(n, semilla):
    ficha = dict(registro="provisorio", numero_documento=None, box_antes_del_registro=False,
                 circuito="ninguno", plazo_circuito_min=None, cobertura="sin_cobertura",
                 entidad_a_notificar=None, aviso_al_voluntariado=False)
    accion = lambda herramienta, argumentos: json.dumps(  # noqa: E731
        {"herramienta": herramienta, "argumentos": argumentos})
    return {"version": 1, "id": f"humo-{n:03d}", "origen": "sintetico",
            "referencias": [f"generador:humo-semilla-{semilla}"],
            "messages": [{"role": "system", "content": "Instrucción sintética de humo"},
                         {"role": "user", "content": f"Paciente sintético {n}"},
                         {"role": "assistant", "content": accion("leer_triage", {"paciente_id": f"P-{n}"})},
                         {"role": "user", "content": "Resultado de leer_triage:\nNota sintética"},
                         {"role": "assistant", "content": accion("registrar_decision", ficha)}]}


def generar(salida, registros=4, semilla=42, epochs=2):
    """Con batch 1 y acumulación 1 hay registros*epochs pasos: pocos y con checkpoints frecuentes."""
    salida = Path(salida)
    if not 1 <= registros <= 50:
        raise ValueError("registros entre 1 y 50")
    salida.mkdir(parents=True)  # Falla si existe: nunca se pisa nada.
    config = json.loads((RAIZ / "configuraciones/sft-ejemplo.json").read_text())
    config.update(semilla=semilla, epochs=epochs, batch_size=1, gradient_accumulation_steps=1)
    (salida / "conversaciones.jsonl").write_text(
        "".join(json.dumps(registro(n, semilla), ensure_ascii=False) + "\n" for n in range(1, registros + 1)))
    (salida / "configuracion.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salida", required=True, help="carpeta nueva")
    p.add_argument("--registros", type=int, default=4)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    a = p.parse_args()
    generar(a.salida, a.registros, a.semilla, a.epochs)
    print(f"datos de humo sintéticos en {a.salida}: {a.registros * a.epochs} pasos de entrenamiento")


if __name__ == "__main__":
    main()
