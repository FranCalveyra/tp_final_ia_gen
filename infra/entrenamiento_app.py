"""Definición de entrenamiento durable; importar no crea recursos remotos."""

import json
from pathlib import Path
import modal

from admision_tp.persistencia import crear_paquete, ejecutar_persistente

RAIZ = Path(__file__).resolve().parents[1]
app = modal.App("tp-admision-entrenamiento")
volume = modal.Volume.from_name("tp-admision-entrenamiento", create_if_missing=True, version=2)
locks = modal.Dict.from_name("tp-admision-entrenamiento-locks", create_if_missing=True)
imagen = (modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/app"})
    .pip_install_from_requirements(str(RAIZ / "requirements-entrenamiento.txt"))
    .add_local_dir(str(RAIZ / "admision"), "/app/admision")
    .add_local_dir(str(RAIZ / "admision_tp"), "/app/admision_tp"))
for nombre in ("pyproject.toml", "uv.lock", "requirements-entrenamiento.txt", "infra/entrenamiento_app.py"):
    imagen = imagen.add_local_file(str(RAIZ / nombre), f"/app/{nombre}")


@app.function(image=imagen, gpu="A100-40GB", timeout=3600, retries=0, max_containers=1,
              volumes={"/corridas": volume}, secrets=[modal.Secret.from_name("huggingface"),
                                                      modal.Secret.from_name("wandb-secret")])
def ejecutar(paquete, nombre: str, checkpoint: str | None = None, online: bool = False):
    return ejecutar_persistente(paquete, nombre, "/corridas", volume, locks,
                               checkpoint=checkpoint, online=online)


@app.local_entrypoint()
def main(preparacion: str, tokens: str, nombre: str, checkpoint: str = "", online: bool = False,
         limite_segundos: int = 3600):
    if not 60 <= limite_segundos <= 14400:
        raise ValueError("límite entre 60 y 14400 segundos")
    paquete = crear_paquete(preparacion, tokens)  # Validación local antes de pedir GPU.
    resultado = ejecutar.with_options(timeout=limite_segundos).remote(paquete, nombre, checkpoint or None, online)
    print(json.dumps(resultado, indent=2))
