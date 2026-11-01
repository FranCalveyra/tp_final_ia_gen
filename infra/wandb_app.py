"""Smoke online futuro, solo CPU. Definir no construye ni ejecuta la imagen."""

from pathlib import Path
import modal

RAIZ = Path(__file__).resolve().parents[1]
app = modal.App("tp-admision-tracking")
imagen = (modal.Image.debian_slim(python_version="3.12")
          .env({"PYTHONPATH": "/app"})
          .pip_install_from_requirements(str(RAIZ / "requirements-tracking.txt"))
          .add_local_file(str(RAIZ / "scripts/probar_wandb.py"), "/app/probar_wandb.py"))


@app.function(image=imagen, timeout=120,
              secrets=[modal.Secret.from_name("wandb-secret")])
def verificar():
    import tempfile
    from probar_wandb import probar
    with tempfile.TemporaryDirectory() as temporal:
        return probar(Path(temporal), online=True)


@app.local_entrypoint()
def main():
    print(verificar.remote())
