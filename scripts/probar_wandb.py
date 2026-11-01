"""Prueba de instrumentación; offline por defecto, sin modelo ni datos de pacientes."""

import argparse
import os
from pathlib import Path

ENTIDAD = "francalveyra-ia-gen"
PROYECTO = "tp-admision"


def probar(salida: Path, *, online: bool = False) -> dict:
    """Registra una señal técnica, no una métrica de calidad del agente."""
    if online and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Para publicar se requiere WANDB_API_KEY; no la pases como argumento")
    import wandb

    salida.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY", ENTIDAD),
        project=os.getenv("WANDB_PROJECT", PROYECTO),
        mode="online" if online else "offline", dir=str(salida),
        job_type="setup-check", name="verificacion-instrumentacion",
        config={"proposito": "setup-check", "entrenamiento": False},
        settings=wandb.Settings(disable_git=True, save_code=False,
                                x_disable_stats=True, console="off"),
    ) as corrida:
        corrida.log({"setup_check": 1})
        return {"id": corrida.id, "modo": "online" if online else "offline"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=Path("corridas/wandb-offline"))
    parser.add_argument("--online", action="store_true", help="Publica una corrida real; requiere autorización")
    args = parser.parse_args()
    print(probar(args.salida, online=args.online))


if __name__ == "__main__":
    main()
