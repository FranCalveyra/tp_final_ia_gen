"""Entrenamiento SFT LoRA en Kaggle (T4 x2), respaldo de desarrollo. Ver docs/kaggle.md.

Script común: una celda de notebook lo ejecuta con `python infra/kaggle_entrenamiento.py ...`.
No duplica lógica: llama a `scripts.entrenar` (que usa `admision_tp.entrenamiento.entrenar`) en un
subproceso para registrar la salida línea a línea. La lógica del agente no conoce Kaggle;
`kaggle_secrets` solo se importa acá y solo si hace falta.

Los checkpoints se publican SIEMPRE en el repo HF privado de la configuración: en Kaggle los archivos
de una versión cortada o vencida se pueden perder, así que HF es el único almacenamiento durable.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from admision_tp.entrenamiento import (DISPOSITIVOS, PRECISIONES, _ejecucion,  # noqa: E402
                                       cargar_entradas, validar_reanudacion)
from infra import kaggle_baseline as kb  # noqa: E402

SALIDA_BASE = Path("/kaggle/working/corridas")


def secreto(nombre: str) -> str | None:
    """Variable de entorno o Kaggle Secret; None si no existe (el llamador decide si es fatal)."""
    if os.environ.get(nombre):
        return os.environ[nombre]
    try:
        return kb.obtener_secreto(nombre) or None
    except Exception:  # Secreto ausente o fuera de Kaggle: no se expone el mensaje.
        return None


def validar_inicio(preparacion, tokens, ejecucion, token_hf, wandb_key, publicar, reanudar=None):
    """Falla antes de gastar cuota. Devuelve la configuración validada."""
    config, _, _, _, identidad, _ = cargar_entradas(preparacion, tokens, ejecucion)
    if publicar:
        if not token_hf:
            raise RuntimeError("Falta HF_WRITE_TOKEN (variable de entorno o Kaggle Secret)")
        if not config.get("hf_repo"):
            raise RuntimeError("La configuración no define hf_repo")
        if not wandb_key:
            raise RuntimeError("Publicar exige W&B online: falta WANDB_API_KEY "
                               "(--sin-publicar solo sirve para ensayos sin durabilidad)")
    if reanudar:
        nombre, revision = reanudar
        if not re.fullmatch(r"checkpoint-\d+", nombre) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Reanudar requiere checkpoint-N y una revisión HF inmutable de 40 hex")
        if not token_hf:
            raise RuntimeError("Falta HF_WRITE_TOKEN para descargar el checkpoint")
    return config, identidad


def descargar_checkpoint(repo, revision, nombre, salida, token, identidad, descargar=None):
    """Baja SOLO la carpeta del checkpoint en la revisión fijada y valida su integridad e identidad."""
    if descargar is None:
        from huggingface_hub import snapshot_download as descargar  # Lazy: pesa y solo se usa acá.
    salida.mkdir(parents=True, exist_ok=True)
    descargar(repo_id=repo, repo_type="model", revision=revision, allow_patterns=[f"{nombre}/*"],
              local_dir=str(salida), token=token)
    checkpoint = salida / nombre
    if not (checkpoint / "corrida.json").is_file():
        raise ValueError("el checkpoint descargado no trae corrida.json")
    # La auditoría de la corrida vive en la raíz de la salida; el checkpoint trae su copia.
    shutil.copy(checkpoint / "corrida.json", salida / "corrida.json")
    validar_reanudacion(salida, checkpoint, identidad)  # Hashes de integridad y mismos datos/config.
    return checkpoint


def armar_comando(preparacion, tokens, salida, ejecucion, online, publicar, checkpoint=None):
    comando = [sys.executable, "-m", "scripts.entrenar", "--preparacion", str(preparacion),
               "--tokens", str(tokens), "--salida", str(salida), "--precision", ejecucion["precision"],
               "--dispositivos", ejecucion["dispositivos"], "--guardar-cada", str(ejecucion["guardar_cada"])]
    if online:
        comando.append("--online")
    if publicar:
        comando.append("--publicar")
    if checkpoint:
        comando += ["--reanudar", str(checkpoint)]
    return comando


def correr(preparacion, tokens, destino: Path, ejecucion: dict, token_hf, wandb_key, publicar=True,
           reanudar=None) -> int:
    ejecucion = _ejecucion(**ejecucion)
    config, identidad = validar_inicio(preparacion, tokens, ejecucion, token_hf, wandb_key, publicar, reanudar)
    online = bool(wandb_key)
    kb.crear_salida(destino)  # Falla si existe: nunca se pisa una corrida.
    salida = destino / "entrenamiento"
    # Nunca se guardan los secretos: solo si existen y qué modo resulta.
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ)}
    entorno.pop("HF_TOKEN", None)
    entorno.pop("WANDB_API_KEY", None)
    if token_hf:
        entorno["HF_TOKEN"] = token_hf
    if wandb_key:
        entorno["WANDB_API_KEY"] = wandb_key
    maquina = kb.describir_maquina()
    runner = {"modo": "reanudacion" if reanudar else "nuevo", "ejecucion": ejecucion, "publicar": publicar,
              "wandb": "online" if online else "offline (sin WANDB_API_KEY)",
              "hf_repo": config["hf_repo"], "modelo": config["modelo"], "revision": config["revision"],
              "semilla": config["semilla"], "configuracion": config,
              "reanudacion": ({"repo": config["hf_repo"], "checkpoint": reanudar[0], "revision": reanudar[1]}
                              if reanudar else None),
              "commit": maquina.get("commit"), "sucio": maquina.get("sucio"), "maquina": maquina,
              "estado": "en_curso", "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    escribir = lambda: (destino / "runner.json").write_text(  # noqa: E731
        json.dumps(runner, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir()  # Si el proceso muere, queda `en_curso`: la corrida no terminó.
    try:
        checkpoint = (descargar_checkpoint(config["hf_repo"], reanudar[1], reanudar[0], salida, token_hf,
                                           identidad) if reanudar else None)
        codigo, cola = kb.ejecutar_con_log(
            armar_comando(preparacion, tokens, salida, ejecucion, online, publicar, checkpoint),
            entorno, destino / "runner.log", [token_hf, wandb_key])
    except Exception as error:
        # Solo el tipo: el mensaje de un tercero podría contener credenciales.
        runner.update(estado="fallida", fin=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error_tipo=type(error).__name__)
        escribir()
        raise
    publicaciones = salida / "publicaciones.json"
    estado = json.loads((salida / "estado.json").read_text()) if (salida / "estado.json").exists() else {}
    runner.update(estado="fallida" if codigo else "terminada", fin=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  returncode=codigo, log_final=cola,
                  publicaciones=(json.loads(publicaciones.read_text()) if publicaciones.exists() else []),
                  vram={k: estado[k] for k in ("vram_pico_gb", "vram_reservada_gb", "vram_pico_por_gpu_gb")
                        if k in estado})
    escribir()
    return codigo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparacion", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True,
                        help=f"Carpeta nueva; se recomienda colgarla de {SALIDA_BASE}")
    parser.add_argument("--precision", default="fp16", choices=sorted(PRECISIONES))
    parser.add_argument("--dispositivos", default="auto", choices=DISPOSITIVOS)
    parser.add_argument("--guardar-cada", type=int, default=50)
    parser.add_argument("--sin-publicar", action="store_true",
                        help="Solo ensayos: sin HF no hay durabilidad (y W&B queda opcional)")
    parser.add_argument("--reanudar-checkpoint", help="checkpoint-N publicado en HF")
    parser.add_argument("--reanudar-revision", help="commit HF inmutable (40 hex) que publicó ese checkpoint")
    args = parser.parse_args(argv)
    if bool(args.reanudar_checkpoint) != bool(args.reanudar_revision):
        parser.error("--reanudar-checkpoint y --reanudar-revision van juntos")
    reanudar = (args.reanudar_checkpoint, args.reanudar_revision) if args.reanudar_checkpoint else None
    ejecucion = {"precision": args.precision, "dispositivos": args.dispositivos,
                 "guardar_cada": args.guardar_cada}
    codigo = correr(args.preparacion, args.tokens, args.salida, ejecucion, secreto("HF_WRITE_TOKEN"),
                    secreto("WANDB_API_KEY"), not args.sin_publicar, reanudar)
    print(f"returncode={codigo}; evidencia en {args.salida}/runner.json y runner.log.")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
