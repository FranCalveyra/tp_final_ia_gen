"""Preparación CPU de una corrida; no entrena, tokeniza ni publica datos."""

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from .configuracion import validar_adapter
from .conversaciones import _json, validar_archivo

RAIZ = Path(__file__).resolve().parents[1]
CAMPOS = {"version", "modelo", "revision", "semilla", "learning_rate", "epochs",
          "batch_size", "gradient_accumulation_steps", "max_seq_length",
          "hf_repo", "wandb_entity", "wandb_project"}


def validar_configuracion(config):
    """Contrato cerrado, sin credenciales; conserva los valores recibidos."""
    if not isinstance(config, dict) or set(config) - {"adapter_inicial", "peso_aviso"} != CAMPOS:
        raise ValueError("configuración: campos faltantes o desconocidos")
    if "peso_aviso" in config:
        # Peso del token del valor de aviso_al_voluntariado en la pérdida (#15, pasada 8).
        peso = config["peso_aviso"]
        if type(peso) not in (int, float) or not math.isfinite(peso) or peso <= 1:
            raise ValueError("configuración: peso_aviso requiere número finito mayor que 1")
    if "adapter_inicial" in config:
        # Seguir entrenando un adapter publicado (#15, aviso): repo privado y revisión inmutable.
        try:
            validar_adapter(config["adapter_inicial"])
        except ValueError as error:
            raise ValueError(f"configuración: adapter_inicial: {error}") from None
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("configuración: versión no soportada")
    for campo in ("modelo", "hf_repo"):
        if not isinstance(config[campo], str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", config[campo]):
            raise ValueError(f"configuración: {campo} requiere propietario/nombre")
    if not isinstance(config["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("configuración: revisión requiere commit HF inmutable")
    for campo in ("wandb_entity", "wandb_project"):
        if not isinstance(config[campo], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config[campo]):
            raise ValueError(f"configuración: {campo} inválido")
    if type(config["semilla"]) is not int or not 0 <= config["semilla"] < 2**32:
        raise ValueError("configuración: semilla requiere entero de 32 bits sin signo")
    for campo in ("batch_size", "gradient_accumulation_steps", "max_seq_length"):
        if type(config[campo]) is not int or config[campo] <= 0:
            raise ValueError(f"configuración: {campo} requiere entero positivo")
    for campo in ("learning_rate", "epochs"):
        valor = config[campo]
        if type(valor) not in (int, float) or not math.isfinite(valor) or valor <= 0:
            raise ValueError(f"configuración: {campo} requiere número finito positivo")


def _canonical(dato):
    return (json.dumps(dato, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(contenido):
    return hashlib.sha256(contenido).hexdigest()


def _codigo():
    def git(*args):
        return subprocess.run(["git", "-C", str(RAIZ), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("no se pudo registrar el estado Git del código") from None


def preparar_corrida(datos, configuracion, salida):
    """Crea una carpeta nueva con snapshots exactos y manifiesto verificable.

    Valida la copia que se conservará, no vuelve a leer el dataset original.
    Una salida existente nunca se reemplaza, tampoco si es un enlace simbólico.
    """
    datos, configuracion, salida = Path(datos), Path(configuracion), Path(salida)
    if salida.exists() or salida.is_symlink():
        raise ValueError("la carpeta de salida ya existe")
    if not salida.parent.is_dir():
        raise ValueError("la carpeta padre de salida debe existir")
    try:
        contenido = datos.read_bytes()
        config = _json(configuracion.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ValueError("no se pudieron leer datos o configuración JSON válidos") from None
    validar_configuracion(config)
    config_bytes = _canonical(config)
    codigo = _codigo()
    # Área temporal hermana: el rename final se hace en el mismo filesystem.
    with tempfile.TemporaryDirectory(prefix=".preparacion-", dir=salida.parent) as temporal:
        carpeta = Path(temporal) / "corrida"
        carpeta.mkdir()
        snapshot = carpeta / "conversaciones.jsonl"
        snapshot.write_bytes(contenido)
        informe = validar_archivo(snapshot)
        if informe["errores"]:
            lineas = ", ".join(str(e["linea"]) for e in informe["errores"][:10])
            raise ValueError(f"dataset inválido; revisar líneas {lineas} con validar_conversaciones")
        (carpeta / "configuracion.json").write_bytes(config_bytes)
        manifiesto = {"version": 1, "estado": "preparada_no_entrenada", "codigo": codigo,
                      "archivos": {"conversaciones.jsonl": _sha(contenido),
                                   "configuracion.json": _sha(config_bytes)},
                      "dataset": {k: informe[k] for k in ("registros", "validos", "por_origen")},
                      "configuracion": config}
        (carpeta / "manifiesto.json").write_bytes(_canonical(manifiesto))
        # Reserva exclusiva: evita reemplazar incluso una salida vacía concurrente.
        try:
            salida.mkdir()
        except FileExistsError:
            raise ValueError("la carpeta de salida ya existe") from None
        try:
            for archivo in carpeta.iterdir():
                archivo.rename(salida / archivo.name)
        except OSError:
            for archivo in salida.iterdir():
                archivo.unlink()
            salida.rmdir()
            raise
    return manifiesto
