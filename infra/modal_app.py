"""Definición solamente: importar este archivo no lanza una GPU."""

import json
from pathlib import Path
import tempfile

import modal

RAIZ = Path(__file__).resolve().parents[1]
app = modal.App("tp-admision-baseline")
imagen = (modal.Image.debian_slim(python_version="3.12")
          .env({"PYTHONPATH": "/app"})
          .pip_install_from_requirements(str(RAIZ / "requirements-inferencia.txt"))
          .add_local_dir(str(RAIZ / "admision"), "/app/admision")
          .add_local_dir(str(RAIZ / "admision_tp"), "/app/admision_tp")
          .add_local_file(str(RAIZ / "entrega.py"), "/app/entrega.py"))


@app.function(image=imagen, gpu="A100-40GB", timeout=900,
              secrets=[modal.Secret.from_name("huggingface")])
def jugar(casos: str, normas: dict[str, str]) -> dict[str, str]:
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)
        carpeta = raiz / "normas"
        carpeta.mkdir()
        for nombre, texto in normas.items():
            if Path(nombre).name != nombre:
                raise ValueError("Nombre de norma inválido")
            (carpeta / nombre).write_text(texto, encoding="utf-8")
        (raiz / "casos.jsonl").write_text(casos, encoding="utf-8")
        salida = raiz / "salida"
        metricas = raiz / "metricas.jsonl"  # Fuera de `salida`: el kit es dueño de esa carpeta.
        # Se conserva la salida del runner incluso cuando jugar_lote falla.
        resultado = subprocess.run([sys.executable, "-m", "admision.correr", "jugar",
            "--entrega", "/app/entrega.py", "--casos", str(raiz / "casos.jsonl"),
            "--normas", str(carpeta), "--salida", str(salida)],
            cwd="/app", env={**__import__("os").environ, "PYTHONPATH": "/app",
                             "MODELO_METRICAS": str(metricas)},
            capture_output=True, text=True, check=False)
        archivos = {p.name: p.read_text(encoding="utf-8") for p in salida.iterdir()} if salida.exists() else {}
        if metricas.exists():
            archivos["metricas.jsonl"] = metricas.read_text(encoding="utf-8")
        archivos["runner.json"] = json.dumps({"returncode": resultado.returncode,
            "stdout": resultado.stdout, "stderr": resultado.stderr}, ensure_ascii=False, indent=2)
        from admision_tp.configuracion import Configuracion
        from dataclasses import asdict
        archivos["configuracion.json"] = json.dumps(asdict(Configuracion.desde_entorno()), indent=2)
        return archivos


@app.local_entrypoint()
def main(variante: str = "original", cantidad: int = 1, salida: str = "corridas/smoke-original"):
    if variante not in {"original", "b", "c"} or not 1 <= cantidad <= 50:
        raise ValueError("Variante original/b/c y cantidad entre 1 y 50")
    destino = Path(salida)
    if destino.exists():
        raise FileExistsError("Elegí una salida nueva para no pisar una corrida")
    casos = (RAIZ / "practica/casos.jsonl").read_text(encoding="utf-8").splitlines()[:cantidad]
    carpeta = RAIZ / "normas" / variante
    normas = {p.name: p.read_text(encoding="utf-8") for p in sorted(carpeta.iterdir())
              if p.name == "parametros.yaml" or p.match("NOR-*.md")}
    archivos = jugar.remote("\n".join(casos) + "\n", normas)
    destino.mkdir(parents=True)
    for nombre, texto in archivos.items():
        (destino / nombre).write_text(texto, encoding="utf-8")
    print(f"Trazas guardadas en {destino}. Puntuar es una fase local separada.")
