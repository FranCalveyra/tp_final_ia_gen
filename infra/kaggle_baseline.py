"""Baseline sin entrenar en Kaggle (T4 x2) como respaldo de desarrollo. Ver docs/kaggle.md.

Script común: una celda de notebook lo ejecuta con `python infra/kaggle_baseline.py ...`.
Es independiente de `modal_app.py` a propósito: ese archivo importa `modal` al cargarse.
La lógica del agente no conoce Kaggle; `kaggle_secrets` se importa solo acá y solo si hace falta.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
VARIANTES = ("original", "b", "c")
SALIDA_BASE = Path("/kaggle/working/corridas")


def obtener_secreto(nombre: str) -> str:
    from kaggle_secrets import UserSecretsClient  # Solo existe en Kaggle.

    return UserSecretsClient().get_secret(nombre)


def preparar_entorno(base: dict, dtype: str, device_map: str, metricas: Path, token: str | None) -> dict:
    """Entorno del subproceso. T4 no tiene bf16 nativo: por defecto fp16 y reparto automático."""
    if not token:
        raise RuntimeError("Falta HF_TOKEN (variable de entorno o Kaggle Secret)")
    return {**base, "PYTHONPATH": str(RAIZ), "HF_TOKEN": token, "MODELO_DTYPE": dtype,
            "MODELO_DEVICE_MAP": device_map, "MODELO_METRICAS": str(metricas)}


@contextmanager
def patch_entorno(entorno: dict):
    anterior = dict(os.environ)
    os.environ.clear()
    os.environ.update(entorno)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(anterior)


def crear_salida(destino: Path) -> Path:
    if destino.exists():
        raise FileExistsError("Elegí una salida nueva para no pisar una corrida")
    destino.mkdir(parents=True)
    return destino


def preparar_insumos(variante: str, cantidad: int, destino: Path, raiz: Path = RAIZ) -> tuple[Path, Path]:
    """Copia los primeros N casos y solo las normas (nunca las fichas esperadas)."""
    if variante not in VARIANTES or not 1 <= cantidad <= 50:
        raise ValueError("Variante original/b/c y cantidad entre 1 y 50")
    casos = (raiz / "practica/casos.jsonl").read_text(encoding="utf-8").splitlines()[:cantidad]
    archivo = destino / "casos.jsonl"
    archivo.write_text("\n".join(casos) + "\n", encoding="utf-8")
    normas = destino / "normas"
    normas.mkdir()
    for p in sorted((raiz / "normas" / variante).iterdir()):
        if p.name == "parametros.yaml" or p.match("NOR-*.md"):
            shutil.copy(p, normas / p.name)
    return archivo, normas


def _salida(comando: list[str], **kwargs) -> str:
    try:
        return subprocess.run(comando, capture_output=True, text=True, check=False, **kwargs).stdout.strip()
    except OSError:
        return ""


def describir_maquina() -> dict:
    """Commit, GPU y versiones. Se pregunta a subprocesos para no cargar torch acá."""
    versiones = _salida([sys.executable, "-c",
        "import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'gpus':torch.cuda.device_count(),'bf16_nativo':torch.cuda.is_bf16_supported(including_emulation=False)}))"])
    return {"commit": _salida(["git", "rev-parse", "HEAD"], cwd=RAIZ),
            "sucio": bool(_salida(["git", "status", "--porcelain"], cwd=RAIZ)),
            "gpus": _salida(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                             "--format=csv,noheader"]).splitlines(),
            "torch": json.loads(versiones) if versiones.startswith("{") else versiones or None,
            "python": sys.version.split()[0]}


def ejecutar_con_log(comando: list[str], entorno: dict, log: Path,
                     token: str | list[str] | None) -> tuple[int, str]:
    """Vuelca la salida línea a línea (secretos tapados): si el proceso muere, el log queda hasta ahí.

    `token` es un secreto o una lista de secretos (por ejemplo HF y W&B)."""
    secretos = [token] if isinstance(token, str) else list(token or [])

    def ocultar(texto: str) -> str:
        for secreto in filter(None, secretos):
            texto = texto.replace(secreto, "***")
        return texto
    cola: list[str] = []
    with subprocess.Popen(comando, cwd=RAIZ, env=entorno, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as proceso, \
            open(log, "w", encoding="utf-8") as archivo:
        for linea in proceso.stdout:
            linea = ocultar(linea)
            archivo.write(linea)
            archivo.flush()
            cola = (cola + [linea])[-200:]
        codigo = proceso.wait()
    return codigo, "".join(cola)


def describir_buscador(entorno) -> dict:
    """Variante de búsqueda que usará `entrega.py` (mismo default) y su configuración."""
    from admision_tp.buscador import VARIANTES as BUSCADORES
    nombre = entorno.get("BUSCADOR_VARIANTE", "baseline")
    if nombre not in BUSCADORES:
        raise ValueError(f"BUSCADOR_VARIANTE debe ser una de: {', '.join(BUSCADORES)}")
    return {"variante": nombre, **asdict(BUSCADORES[nombre])}


def correr(variante: str, cantidad: int, destino: Path, dtype: str, device_map: str,
           token: str | None) -> int:
    sys.path.insert(0, str(RAIZ))
    from admision_tp.configuracion import Configuracion
    Configuracion(dtype=dtype, device_map=device_map)  # Valida antes de gastar cuota.
    buscador = describir_buscador(os.environ)  # Ídem: una variante desconocida falla acá.
    crear_salida(destino)
    metricas = destino / "metricas.jsonl"  # Fuera de `salida`: el kit es dueño de esa carpeta.
    entorno = preparar_entorno(os.environ, dtype, device_map, metricas, token)
    # Se registra la configuración que leerá el agente (incluye MODELO_LOTE), no los defaults.
    with patch_entorno(entorno):
        configuracion = Configuracion.desde_entorno()
    casos, normas = preparar_insumos(variante, cantidad, destino)
    runner = {"variante": variante, "cantidad": cantidad, "configuracion": asdict(configuracion),
              "buscador": buscador,
              "maquina": describir_maquina(), "estado": "en_curso",
              "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    escribir = lambda: (destino / "runner.json").write_text(  # noqa: E731
        json.dumps(runner, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir()  # Si el proceso muere, queda `en_curso`: la corrida no terminó.
    codigo, cola = ejecutar_con_log([sys.executable, "-m", "admision.correr", "jugar",
        "--entrega", "entrega.py", "--casos", str(casos), "--normas", str(normas),
        "--salida", str(destino / "salida")], entorno, destino / "runner.log", token)
    # El kit atrapa los errores de jugar_lote, los anota en tiempos.json y sale con 0:
    # sin esto, una corrida que se cayó a mitad de camino parece exitosa.
    error = error_del_kit(destino / "salida" / "tiempos.json")
    if error and codigo == 0:
        codigo = 1
    runner.update(estado="fallida" if codigo else "terminada", fin=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  returncode=codigo, error_agente=error, log_final=cola)
    escribir()
    return codigo


def error_del_kit(tiempos: Path) -> str | None:
    try:
        return json.loads(tiempos.read_text(encoding="utf-8")).get("error") or None
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variante", default="original", choices=VARIANTES)
    parser.add_argument("--cantidad", type=int, default=3)
    parser.add_argument("--salida", type=Path, required=True,
                        help=f"Carpeta nueva; se recomienda colgarla de {SALIDA_BASE}")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or obtener_secreto("HF_TOKEN")
    codigo = correr(args.variante, args.cantidad, args.salida, args.dtype, args.device_map, token)
    print(f"returncode={codigo}; resultados en {args.salida}. Puntuar es una fase local separada.")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
