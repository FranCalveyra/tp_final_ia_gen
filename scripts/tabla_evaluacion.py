"""Puntúa corridas descargadas de Kaggle y arma la tabla de la Parte 4 del informe (fase local, sin GPU).

    python -m scripts.tabla_evaluacion --evaluacion corridas/kaggle/<evaluacion> --salida corridas/tablas/<nueva>
    python -m scripts.tabla_evaluacion --corrida sin_entrenar=corridas/kaggle/v22/baseline-original-n50-l4 \\
        --salida corridas/tablas/<nueva>

`--evaluacion` es una carpeta de `infra/kaggle_evaluacion.py` (con `evaluacion.json`); `--corrida ETIQUETA=CARPETA`
es una corrida suelta de `infra/kaggle_baseline.py` (con `runner.json`). Puntúa con `admision.correr.puntuar` contra
`practica/esperadas-<version>.jsonl` y escribe en una carpeta nueva `tabla.json` (cada número con su procedencia),
`tabla.md` y, por corrida, `resultados.jsonl` y `resumen.json` del kit. Puntuar la práctica es solo para reportar:
nunca para elegir prompt, checkpoint ni hiperparámetros. Ver docs/evaluacion.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from admision.casos import cargar_esperadas, leer_jsonl
from admision.comparar import CONTROLES
from admision.correr import puntuar

RAIZ = Path(__file__).resolve().parents[1]
VERSIONES = ("original", "b", "c")
FILAS = ("sin_entrenar", "solo_prompt", "solo_entrenamiento", "ambos")
REFERENCIA = {"sin entrenar": {"original": 12, "b": 15, "c": 12},
              "fine-tuning supervisado simple (datos perfectos de la cátedra)": {"original": 25, "b": 24, "c": 25}}
NOTA_PRACTICA = ("Puntuado sobre los 50 casos de práctica solo para reportar: la práctica no se usó para "
                 "seleccionar prompt, checkpoint ni hiperparámetros.")


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _mostrar(ruta: Path) -> str:
    """Ruta relativa al directorio actual si se puede (más legible y portable en el informe)."""
    try:
        return str(Path(ruta).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(ruta)


def _num(valor, decimales: int = 0) -> str:
    """Formato del informe: 0,831 y 19.954."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def descubrir(evaluaciones: list[Path], sueltas: list[str]) -> list[dict]:
    """Corridas a puntuar: {etiqueta, carpeta, registro} (registro = fila de evaluacion.json o None)."""
    corridas = []
    for carpeta in evaluaciones:
        manifiesto = json.loads((Path(carpeta) / "evaluacion.json").read_text(encoding="utf-8"))
        for fila in manifiesto["corridas"]:
            corridas.append({"etiqueta": fila["etiqueta"], "carpeta": Path(carpeta) / fila["carpeta"],
                             "registro": fila, "comando_plan": manifiesto.get("comando"),
                             "plan_sha256": manifiesto.get("plan_sha256")})
    for texto in sueltas:
        etiqueta, separador, ruta = texto.partition("=")
        if not separador or not etiqueta or not ruta:
            raise ValueError(f"--corrida espera ETIQUETA=CARPETA: {texto}")
        carpeta = Path(ruta)
        if not (carpeta / "runner.json").is_file() or not (carpeta / "salida/trazas.jsonl").is_file():
            raise ValueError(f"{carpeta}: falta runner.json o salida/trazas.jsonl; sin procedencia no hay número")
        corridas.append({"etiqueta": etiqueta, "carpeta": carpeta, "registro": None})
    return corridas


def _checkpoint(configuracion: dict) -> str:
    if configuracion.get("adapter_repo"):
        return (f"{configuracion['adapter_repo']}@{configuracion['adapter_revision']}"
                f"/{configuracion.get('adapter_subcarpeta') or ''}".rstrip("/"))
    return "sin entrenar"


def _comando_jugar(runner: dict) -> str:
    """Comando equivalente, reconstruido de runner.json (el runner no guarda su línea de comando)."""
    conf, prompt = runner["configuracion"], runner.get("prompt") or {}
    variables = [f"MODELO_LOTE={conf['lote']}",
                 f"BUSCADOR_VARIANTE={(runner.get('buscador') or {}).get('variante', 'baseline')}"]
    if prompt.get("archivo"):
        variables.append(f"PROMPT_SPEC={prompt['archivo']}")
    if conf.get("adapter_repo"):
        variables += [f"ADAPTER_REPO={conf['adapter_repo']}", f"ADAPTER_REVISION={conf['adapter_revision']}",
                      f"ADAPTER_SUBCARPETA={conf['adapter_subcarpeta']}"]
    salida = str(Path(conf["metricas"]).parent) if conf.get("metricas") else "<salida>"
    return " ".join([*variables, "python infra/kaggle_baseline.py", "--variante", runner["variante"],
                     "--cantidad", str(runner["cantidad"]), "--salida", salida, "--dtype", conf["dtype"],
                     "--device-map", conf["device_map"]])


def puntuar_corrida(corrida: dict, practica: Path, salida: Path) -> dict:
    """Celda con números y procedencia, o {"faltante": motivo} si la corrida no sirve para la tabla."""
    carpeta, registro = Path(corrida["carpeta"]), corrida["registro"]
    base = {"etiqueta": corrida["etiqueta"], "carpeta": _mostrar(carpeta)}
    if registro is not None:
        base["version"] = registro["version"]
    runner_json = carpeta / "runner.json"
    if not runner_json.is_file():
        motivo = (registro or {}).get("error") or "no hay runner.json"
        return {**base, "faltante": f"corrida {(registro or {}).get('estado', 'sin datos')}: {motivo}"}
    runner = json.loads(runner_json.read_text(encoding="utf-8"))
    version = runner["variante"]
    if registro is not None and registro["version"] != version:
        raise ValueError(f"{carpeta}: evaluacion.json dice {registro['version']} y runner.json {version}")
    base["version"] = version
    if runner.get("estado") != "terminada":
        motivo = runner.get("error_agente") or (registro or {}).get("error") or "sin error registrado"
        return {**base, "faltante": f"corrida {runner.get('estado')}: {motivo}"}
    ruta_trazas = carpeta / "salida/trazas.jsonl"
    trazas = leer_jsonl(ruta_trazas)
    if len(trazas) != runner["cantidad"]:
        return {**base, "faltante": f"trazas incompletas: {len(trazas)} de {runner['cantidad']}"}
    ruta_esperadas = Path(practica) / f"esperadas-{version}.jsonl"
    tiempos_json = carpeta / "salida/tiempos.json"
    tiempos = json.loads(tiempos_json.read_text(encoding="utf-8")) if tiempos_json.is_file() else None
    destino = salida / "corridas" / f"{corrida['etiqueta']}-{version}"
    destino.mkdir(parents=True)
    _, resumen = puntuar(trazas, cargar_esperadas(ruta_esperadas), destino, tiempos)  # La función del kit.
    conf, maquina = runner["configuracion"], runner.get("maquina") or {}
    avisos = []
    if not runner.get("prompt"):
        avisos.append("runner.json no registra el prompt (corrida anterior al registro de prompt en el runner)")
    if maquina.get("sucio") is not False:
        avisos.append("árbol Git sucio o sin dato: el commit no reconstruye solo el código")
    if any("A100" not in gpu for gpu in maquina.get("gpus") or [""]):
        avisos.append("no corrió en A100: tiempos y resultados no comparables con la entrega")
    segundos = (tiempos or {}).get("jugar_lote_por_caso")
    return {
        **base,
        "resueltos": resumen["resueltos"], "casos": resumen["casos"], "parcial_medio": resumen["parcial_medio"],
        "por_control": resumen["por_control"], "sin_decision": resumen["sin_decision"],
        "llamadas_medio": resumen["llamadas_medio"], "busquedas_por_caso": resumen["busquedas_por_caso"],
        "caracteres_de_contexto_por_caso": resumen["caracteres_de_contexto_por_caso"],
        "segundos_por_caso": segundos, "segundos": resumen.get("segundos"), "error": resumen.get("error"),
        "procedencia": {
            "commit": maquina.get("commit"), "sucio": maquina.get("sucio"),
            "modelo": f"{conf['modelo']}@{conf['revision']}", "checkpoint": _checkpoint(conf),
            "prompt": runner.get("prompt"), "buscador": runner.get("buscador"), "configuracion": conf,
            "hardware": {k: maquina.get(k) for k in ("gpus", "torch", "python")},
            "inicio": runner.get("inicio"), "fin": runner.get("fin"),
            "trazas": _mostrar(ruta_trazas), "trazas_sha256": _sha256(ruta_trazas),
            "esperadas": _mostrar(ruta_esperadas), "esperadas_sha256": _sha256(ruta_esperadas),
            "comando_jugar": _comando_jugar(runner),
            "comando_jugar_origen": "reconstruido de runner.json",
            "comando_plan": corrida.get("comando_plan"), "plan_sha256": corrida.get("plan_sha256"),
            "comando_puntuar": (f"python -m admision.correr puntuar --trazas {_mostrar(ruta_trazas)} "
                                f"--esperadas {_mostrar(ruta_esperadas)} --salida {_mostrar(destino)}"),
        },
        "avisos": avisos,
    }


def _clave(corrida: dict) -> tuple[str, str]:
    runner = Path(corrida["carpeta"]) / "runner.json"
    if runner.is_file():
        return corrida["etiqueta"], json.loads(runner.read_text(encoding="utf-8"))["variante"]
    return corrida["etiqueta"], corrida["registro"]["version"]


def armar(corridas: list[dict], practica: Path, salida: Path, comando: list[str]) -> dict:
    if salida.exists():
        raise FileExistsError(f"{salida} ya existe: elegí una salida nueva para no pisar una tabla")
    claves = [_clave(c) for c in corridas]
    if repetidas := sorted({k for k in claves if claves.count(k) > 1}):
        raise ValueError(f"hay más de una corrida para la misma configuración y versión {repetidas}: elegí una")
    salida.mkdir(parents=True)
    celdas, faltantes = [], []
    for corrida in corridas:
        celda = puntuar_corrida(corrida, practica, salida)
        (faltantes if "faltante" in celda else celdas).append(celda)
    claves = [(c["etiqueta"], c["version"]) for c in celdas]
    filas = list(FILAS) + [e for e in dict.fromkeys(c["etiqueta"] for c in corridas) if e not in FILAS]
    for fila in filas:
        for version in VERSIONES:
            if (fila, version) not in claves and not any(
                    (f["etiqueta"], f.get("version")) == (fila, version) for f in faltantes):
                faltantes.append({"etiqueta": fila, "version": version, "faltante": "sin corrida"})
    return {"generado": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "comando": comando,
            "git": {"commit": _git("rev-parse", "HEAD"), "sucio": bool(_git("status", "--porcelain"))},
            "nota": NOTA_PRACTICA, "filas": filas, "versiones": list(VERSIONES),
            "referencia_catedra": REFERENCIA, "celdas": celdas, "faltantes": faltantes}


def markdown(tabla: dict) -> str:
    celdas = {(c["etiqueta"], c["version"]): c for c in tabla["celdas"]}
    versiones = tabla["versiones"]
    lineas = ["# Evaluación completa: casos resueltos (Parte 4)", "", f"> {tabla['nota']}", "",
              "| Configuración | " + " | ".join(versiones) + " | Checkpoint | Prompt |",
              "|---|" + "---|" * (len(versiones) + 2)]
    for fila in tabla["filas"]:
        propias = [celdas[(fila, v)] for v in versiones if (fila, v) in celdas]
        valores = [f"{celdas[(fila, v)]['resueltos']}/{celdas[(fila, v)]['casos']}" if (fila, v) in celdas
                   else "—" for v in versiones]
        checkpoints = sorted({c["procedencia"]["checkpoint"] for c in propias}) or ["—"]
        prompts = sorted({f"{p['nombre']} v{p['version']} ({p['sha256'][:12]})" if (p := c["procedencia"]["prompt"])
                          else "no registrado" for c in propias}) or ["—"]
        lineas.append(f"| {fila} | " + " | ".join(valores) + f" | {', '.join(checkpoints)} | {', '.join(prompts)} |")
    for nombre, valores in tabla["referencia_catedra"].items():
        lineas.append(f"| *Referencia de la cátedra: {nombre}* | "
                      + " | ".join(f"{valores[v]}/50" for v in versiones) + " | (cátedra) | (cátedra) |")
    lineas += ["", "Las filas de referencia son de la cátedra (Gemma 4 E4B; mission.md, "
               "«Referencia de la cátedra»), "
               "no corridas nuestras.",
               "«—» = no hay número: ver «Faltantes». Ningún valor se completa a mano.", "",
               "## Costo y tiempo por caso", "",
               "| Configuración | Normas | Parcial medio | Caracteres de contexto/caso | Segundos/caso "
               "| Llamadas/caso | Búsquedas/caso | Sin decisión | Hardware |",
               "|---|---|---|---|---|---|---|---|---|"]
    for c in tabla["celdas"]:
        gpus = c["procedencia"]["hardware"].get("gpus") or ["sin dato"]
        hardware = f"{len(gpus)}× {gpus[0].split(',')[0]}" if gpus[0] != "sin dato" else "sin dato"
        segundos = _num(c["segundos_por_caso"], 2) if c["segundos_por_caso"] is not None else "—"
        lineas.append(f"| {c['etiqueta']} | {c['version']} | {_num(c['parcial_medio'], 3)} | "
                      f"{_num(c['caracteres_de_contexto_por_caso'])} | {segundos} | {_num(c['llamadas_medio'], 1)} | "
                      f"{_num(c['busquedas_por_caso'], 2)} | {c['sin_decision']} | {hardware} |")
    lineas += ["", "Segundos/caso = `jugar_lote / casos` del kit (sin cargar el modelo); con lotes los casos corren "
               "a la vez. Resultados en T4 no son comparables con A100.", "", "## Por control", "",
               "| Configuración | Normas | " + " | ".join(CONTROLES) + " |", "|---|---|" + "---|" * len(CONTROLES)]
    for c in tabla["celdas"]:
        lineas.append(f"| {c['etiqueta']} | {c['version']} | "
                      + " | ".join(_num(c["por_control"][k], 2) for k in CONTROLES) + " |")
    lineas += ["", "## Procedencia", ""]
    for c in tabla["celdas"]:
        p = c["procedencia"]
        prompt = (f"{p['prompt']['nombre']} v{p['prompt']['version']}, `{p['prompt']['archivo']}`, "
                  f"sha256 `{p['prompt']['sha256']}`" if p["prompt"] else "no registrado en runner.json")
        lineas += [f"- **{c['etiqueta']} × {c['version']}**: commit `{p['commit']}` (sucio={p['sucio']}); "
                   f"modelo `{p['modelo']}`; checkpoint {p['checkpoint']}; prompt {prompt}; "
                   f"buscador `{(p['buscador'] or {}).get('variante', 'no registrado')}`; "
                   f"lote {p['configuracion']['lote']}, {p['configuracion']['dtype']}; {p['inicio']} → {p['fin']}.",
                   f"  - Trazas `{p['trazas']}` (sha256 `{p['trazas_sha256']}`).",
                   f"  - Jugar ({p['comando_jugar_origen']}): `{p['comando_jugar']}`"]
        if p.get("comando_plan"):
            lineas.append(f"  - Plan: `{' '.join(p['comando_plan'])}` (plan sha256 `{p['plan_sha256']}`)")
        lineas.append(f"  - Puntuar (equivalente): `{p['comando_puntuar']}`")
        lineas += [f"  - Aviso: {aviso}" for aviso in c["avisos"]]
    lineas += ["", "## Faltantes", ""]
    lineas += ([f"- {f['etiqueta']} × {f.get('version', '?')}: {f['faltante']}" for f in tabla["faltantes"]]
               or ["- Ninguno."])
    lineas += ["", f"Tabla generada {tabla['generado']} con `{' '.join(tabla['comando'])}` "
               f"(commit `{tabla['git']['commit']}`, sucio={tabla['git']['sucio']})."]
    return "\n".join(lineas) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluacion", type=Path, action="append", default=[],
                        help="carpeta de infra/kaggle_evaluacion.py (repetible)")
    parser.add_argument("--corrida", action="append", default=[], help="ETIQUETA=CARPETA de una corrida suelta")
    parser.add_argument("--practica", type=Path, default=RAIZ / "practica", help="carpeta con esperadas-*.jsonl")
    parser.add_argument("--salida", type=Path, required=True, help="carpeta nueva")
    args = parser.parse_args(argv)
    if not args.evaluacion and not args.corrida:
        parser.error("pasá al menos una --evaluacion o --corrida")
    comando = ["python", "-m", "scripts.tabla_evaluacion", *(argv if argv is not None else sys.argv[1:])]
    tabla = armar(descubrir(args.evaluacion, args.corrida), args.practica, args.salida, comando)
    (args.salida / "tabla.json").write_text(json.dumps(tabla, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.salida / "tabla.md").write_text(markdown(tabla), encoding="utf-8")
    print(markdown(tabla))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
