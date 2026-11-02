"""Transferencia acotada y ejecución durable; sin dependencia de Modal al importar."""

from pathlib import Path
import re
import sys
import tempfile
import uuid

from .entrenamiento import cargar_entradas, entrenar
from .preparacion import _canonical, _codigo, _json, _sha

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = {"preparacion/conversaciones.jsonl", "preparacion/configuracion.json",
            "preparacion/manifiesto.json", "tokens/tokens.jsonl", "tokens/manifiesto-tokens.json"}
MAX_ARCHIVO = 64 * 1024 * 1024
MAX_PAQUETE = 128 * 1024 * 1024


def hashes_codigo(raiz=RAIZ):
    """Solo fuentes/config necesarias, nunca .git, .env, datos o credenciales."""
    raiz = Path(raiz)
    rutas = [p for carpeta in ("admision", "admision_tp") for p in (raiz / carpeta).rglob("*.py")]
    rutas += [raiz / n for n in ("pyproject.toml", "uv.lock", "requirements-entrenamiento.txt",
                                 "infra/entrenamiento_app.py")]
    if any(p.is_symlink() for p in rutas):
        raise ValueError("fuentes enlazadas no permitidas")
    return {str(p.relative_to(raiz)): _sha(p.read_bytes()) for p in sorted(rutas)}


def crear_paquete(preparacion, tokens):
    datos = {}
    for prefijo, carpeta in (("preparacion", Path(preparacion)), ("tokens", Path(tokens))):
        if carpeta.is_symlink():
            raise ValueError("carpetas enlazadas no permitidas")
        for nombre in sorted(n for n in ARCHIVOS if n.startswith(prefijo + "/")):
            archivo = carpeta / nombre.split("/")[1]
            if archivo.is_symlink() or archivo.stat().st_size > MAX_ARCHIVO:
                raise ValueError("archivo enlazado o demasiado grande")
            datos[nombre] = archivo.read_bytes()
    paquete = {"archivos": datos, "codigo": {**_codigo(), "archivos": hashes_codigo()}}
    validar_paquete(paquete)
    return paquete


def _materializar(paquete, destino):
    for nombre, contenido in paquete["archivos"].items():
        ruta = destino / nombre
        ruta.parent.mkdir(exist_ok=True)
        ruta.write_bytes(contenido)


def validar_paquete(paquete):
    if (not isinstance(paquete, dict) or set(paquete) != {"archivos", "codigo"}
            or not isinstance(paquete["archivos"], dict) or set(paquete["archivos"]) != ARCHIVOS
            or any(type(b) is not bytes or len(b) > MAX_ARCHIVO for b in paquete["archivos"].values())
            or sum(len(b) for b in paquete["archivos"].values()) > MAX_PAQUETE):
        raise ValueError("paquete inválido o demasiado grande")
    codigo = paquete["codigo"]
    if (not isinstance(codigo, dict) or set(codigo) != {"commit", "dirty", "archivos"}
            or not re.fullmatch(r"[0-9a-f]{40}", str(codigo["commit"]))
            or type(codigo["dirty"]) is not bool or codigo["archivos"] != hashes_codigo()):
        raise ValueError("código enviado distinto del código ejecutado")
    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)
        _materializar(paquete, raiz)
        cargar_entradas(raiz / "preparacion", raiz / "tokens")


def ejecutar_persistente(paquete, nombre, montaje, volumen, locks, *, checkpoint=None,
                        online=False, backend=None):
    """No publica HF. Dict excluye ejecuciones simultáneas incluso entre apps."""
    if not isinstance(nombre, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", nombre):
        raise ValueError("nombre de corrida inválido")
    if checkpoint is not None and (not isinstance(checkpoint, str)
                                   or not re.fullmatch(r"checkpoint-\d+", checkpoint)):
        raise ValueError("checkpoint requiere nombre checkpoint-N, no una ruta")
    validar_paquete(paquete)  # Antes de tocar Volume o tomar el lock remoto.
    propietario = uuid.uuid4().hex
    if not locks.put(nombre, propietario, skip_if_exists=True):
        raise ValueError("corrida bloqueada; no se elimina un lock existente automáticamente")
    try:
        volumen.reload()
        carpeta = Path(montaje) / nombre
        if carpeta.is_symlink():
            raise ValueError("corrida enlazada no permitida")
        entradas = carpeta / "transferencia"
        if checkpoint:
            if not entradas.is_dir() or entradas.is_symlink():
                raise ValueError("corrida durable inexistente")
            if any(p.is_symlink() for p in carpeta.rglob("*")):
                raise ValueError("enlaces no permitidos en corrida durable")
            for n, b in paquete["archivos"].items():
                ruta = entradas / n
                if ruta.is_symlink() or ruta.read_bytes() != b:
                    raise ValueError("entradas distintas de la corrida durable")
            if _json((carpeta / "codigo.json").read_text()) != paquete["codigo"]:
                raise ValueError("código distinto de la corrida durable")
        else:
            carpeta.mkdir()  # Nunca sobrescribir una corrida, incluso incompleta.
            entradas.mkdir()
            _materializar(paquete, entradas)
            (carpeta / "codigo.json").write_bytes(_canonical(paquete["codigo"]))
            volumen.commit()
        salida = carpeta / "entrenamiento"
        resultado = entrenar(entradas / "preparacion", entradas / "tokens", salida,
            checkpoint=salida / checkpoint if checkpoint else None, online=online,
            publicar=False, backend=backend, codigo=paquete["codigo"], persistir=volumen.commit)
        return {"corrida": nombre, "ruta_volume": str(salida), **resultado}
    finally:
        # Un cierre abrupto no ejecuta finally: requiere inspección manual del lock.
        error_previo = sys.exc_info()[0] is not None
        try:
            if locks.get(nombre) == propietario:
                locks.pop(nombre)
        except Exception:
            if not error_previo:
                raise
