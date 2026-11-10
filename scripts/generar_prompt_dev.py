"""Genera el conjunto de desarrollo para seleccionar prompts (issue #13) en una carpeta nueva.

K carpetas de normas (los textos de `normas/original` con valores de parámetros sorteados entre los
que muestran las 800 trazas de referencia, como en el issue #8) × N casos de `admision.generador`
cada una. La ficha esperada de cada caso sale del deductor fijo de Yas
(`scripts.generar_sinteticos_entrenamiento.deducir_ficha`: reglas + las dos costumbres de
`historial_costumbres.json`) con los valores de su carpeta; acá solo se agrega la parte
`verificacion` del formato del kit. Nunca se leen esperadas de práctica ni `normas/b`/`normas/c`.

Disposición (las esperadas quedan fuera del alcance de `revisar_que_no_haya_esperadas` del kit):

    <salida>/entradas/n<k>/casos.jsonl       casos que juega el agente
    <salida>/entradas/n<k>/normas/            carpeta de normas con sus valores
    <salida>/esperadas/esperadas-n<k>.jsonl   fichas esperadas, se usan solo al puntuar
    <salida>/auditoria.jsonl                  estratos de cada caso
    <salida>/manifiesto.json                  semilla, valores, balance, exclusiones y hashes
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from admision.casos import Caso
from admision.comparar import CONTROLES
from admision.entorno import _norm, nota_de_triage
from admision.generador import generar
from admision_tp.normas_variables import CarpetasNormas, parametros_base
from scripts.clasificar_historial import cargar_parametros
from scripts.generar_conversaciones import _git, _leer_jsonl, _relativa, _sha256, valores_observados
from scripts.generar_sinteticos_entrenamiento import (
    circuito_caso,
    costumbres_por_tipo,
    deducir_ficha,
    dificultad_caso,
)

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 20261110          # distinta de 12012 (datos de entrenamiento del #8 y de Yas)
SEMILLA_ENTRENAMIENTO = 12012
CANDIDATOS_ENTRENAMIENTO = 5000
CANDIDATOS = 4000
CARPETAS = 2
CASOS_POR_CARPETA = 20
CIRCUITOS = ("ecg", "codigo_acv", "sepsis_probable", "sepsis_posible", "ninguno")
# Orden de llenado: primero los buckets más restringidos (ART solo trae ecg o ninguno).
BUCKETS = ("art", "dos_credenciales", "acompanante", "sin_nadie", "general")
# Fuentes cuyos pacientes no pueden repetirse en el conjunto de desarrollo.
FUENTES_EXCLUIDAS = ("practica/casos.jsonl", "entrenamiento/trazas.jsonl", "historial/trazas.jsonl",
                     "datos/entrenamiento/conversaciones.jsonl",
                     "datos/conversaciones/v2-baseline/conversaciones.jsonl")
NUMERO = re.compile(r"(?<![\w.\-/])(?:[A-Z]{3}\d{6}|\d{2}-\d{6}-\d|\d{2}-\d{8}-\d|\d{6}/\d{2}"
                    r"|\d{2}\.\d{3}\.\d{3})(?![\w\-/]|\.\d)")
ENCABEZADO = "Nota de triage del paciente "


# --- identidad de pacientes ---------------------------------------------------------------------

def _identidad_de_nota(nota: str) -> tuple[str, str]:
    """(línea de id + línea de nombre y edad, cuerpo sin la línea de id) de una nota de triage."""
    lineas = nota.splitlines()
    return "\n".join(lineas[:2]), "\n".join(lineas[1:])


def _textos(valor):
    if isinstance(valor, str):
        yield valor
    elif isinstance(valor, dict):
        for v in valor.values():
            yield from _textos(v)
    elif isinstance(valor, list):
        for v in valor:
            yield from _textos(v)


def _agregar_texto(huellas: dict, texto: str) -> None:
    huellas["numeros"].update(NUMERO.findall(texto))
    inicio = texto.find(ENCABEZADO)
    if inicio >= 0:
        paciente, cuerpo = _identidad_de_nota(texto[inicio:])
        huellas["pacientes"].add(paciente)
        huellas["triages"].add(cuerpo)


def huellas_excluidas(raiz: Path = RAIZ) -> dict:
    """Pacientes de práctica (solo casos, nunca esperadas) y de todos los datos de entrenamiento.

    Identidad completa: nota de triage sin el id, id + nombre y edad, y cada número de documento,
    afiliado o CUIL que aparezca en cualquier texto. Incluye los 5.000 candidatos de la semilla del
    #8 aunque no se hayan seleccionado.
    """
    huellas = {"triages": set(), "pacientes": set(), "numeros": set()}
    for relativa in FUENTES_EXCLUIDAS:
        for registro in _leer_jsonl(Path(raiz) / relativa):
            if relativa == "practica/casos.jsonl":
                registro = {**registro, "nota_completa": nota_de_triage(Caso.desde_dict(registro))}
            for texto in _textos(registro):
                _agregar_texto(huellas, texto)
    for caso in generar(semilla=SEMILLA_ENTRENAMIENTO, n=CANDIDATOS_ENTRENAMIENTO,
                        prefijo=f"cand-{SEMILLA_ENTRENAMIENTO}"):
        _agregar_texto(huellas, nota_de_triage(caso))
        huellas["numeros"].update(numeros_caso(caso))
    return huellas


def numeros_caso(caso) -> set[str]:
    numeros = {caso.documento_numero} if caso.documento_numero else set()
    numeros |= {c[2] for c in caso.credenciales}
    numeros |= {caso.declarada[2]} if caso.declarada else set()
    numeros |= {caso.art[1]} if caso.art else set()
    return numeros


def solapamiento(caso, huellas: dict) -> str | None:
    """Motivo estable si el paciente ya aparece en práctica o en datos de entrenamiento."""
    paciente, cuerpo = _identidad_de_nota(nota_de_triage(caso))
    if cuerpo in huellas["triages"]:
        return "triage_identico"
    if paciente in huellas["pacientes"]:
        return "paciente_identico"
    if numeros_caso(caso) & huellas["numeros"]:
        return "numero_compartido"
    return None


# --- esperadas ----------------------------------------------------------------------------------

def verificacion_esperada(caso, fila: dict) -> dict:
    """Formato `verificacion` del kit: la afiliación que hay que verificar y las entidades prohibidas.

    `requerida` es la fuente que eligió el deductor (ART, prepaga preferida, credencial o declarada);
    `prohibidas`, cualquier otra entidad que el paciente tenga (credencial o declarada).
    """
    if not fila["entidad_verificada"]:
        return {"requerida": None, "prohibidas": []}
    requerida = [fila["entidad_verificada"], fila["numero_verificado"]]
    otras = [c[1] for c in caso.credenciales] + ([caso.declarada[1]] if caso.declarada else [])
    prohibidas = [e for e in dict.fromkeys(otras) if _norm(e) != _norm(requerida[0])]
    return {"requerida": requerida, "prohibidas": prohibidas}


def esperada_de(caso, parametros: dict, costumbres: dict) -> dict:
    fila, ficha = deducir_ficha(caso, parametros, costumbres)
    ficha = {campo: ficha[campo] for campo in CONTROLES if campo != "verificacion"}
    return {"caso": caso.id, "ficha": ficha, "verificacion": verificacion_esperada(caso, fila)}


# --- selección balanceada -----------------------------------------------------------------------

def grupo_aviso(caso, umbral: int) -> str:
    """Estrato de la costumbre del voluntariado (el control más débil del baseline)."""
    if caso.edad >= umbral and not caso.acompanante:
        return "mayor_sin_acompanante"
    if caso.edad >= umbral:
        return "mayor_con_acompanante"
    return "menor"


def cuotas(n: int) -> dict:
    if n < 10 or n % 5:
        raise ValueError("casos por carpeta: múltiplo de 5, al menos 10")
    especial = max(1, n * 3 // 20)
    return {
        "circuito": {c: n // 5 for c in CIRCUITOS},
        "bucket": {**{b: especial for b in BUCKETS[:-1]}, "general": n - 4 * especial},
        "aviso": {"mayor_sin_acompanante": n // 5, "mayor_con_acompanante": n // 10},
    }


def seleccionar(candidatos, n: int, umbral: int, esperada) -> list[tuple]:
    """Elige n casos con cuotas exactas por circuito, bucket y estrato de aviso.

    `esperada(caso)` devuelve la ficha o levanta ValueError (caso que el deductor no resuelve,
    se saltea). El orden es determinista: bucket más restringido, estrato de aviso y orden del generador.
    """
    q = cuotas(n)
    rango_aviso = {"mayor_sin_acompanante": 0, "mayor_con_acompanante": 1, "menor": 2}
    orden = sorted(candidatos, key=lambda c: (
        BUCKETS.index(dificultad_caso(c)), rango_aviso[grupo_aviso(c, umbral)]))
    elegidos, conteo = [], Counter()
    for caso in orden:
        claves = (("circuito", circuito_caso(caso)), ("bucket", dificultad_caso(caso)),
                  ("aviso", grupo_aviso(caso, umbral)))
        # "menor" no tiene cuota propia: completa lo que dejan los otros estratos.
        if any(conteo[(tipo, valor)] >= q[tipo].get(valor, n) for tipo, valor in claves):
            continue
        try:
            ficha = esperada(caso)
        except ValueError:
            continue
        elegidos.append((caso, ficha))
        conteo.update(claves)
        if len(elegidos) == n:
            break
    faltan = [f"{t}={v}" for t, valores in q.items() for v, meta in valores.items() if conteo[(t, v)] != meta]
    if len(elegidos) != n or faltan:
        raise ValueError(f"no alcanzan los candidatos para las cuotas: {', '.join(faltan) or len(elegidos)}")
    return elegidos


# --- valores de parámetros ----------------------------------------------------------------------

def sortear_valores(observados: dict, originales: dict, semilla: int, k: int) -> list[dict]:
    """k conjuntos distintos entre sí y del original, cada valor entre los observados en la referencia."""
    sorteo = random.Random(f"{semilla}:parametros")
    conjuntos = []
    while len(conjuntos) < k:
        valores = {codigo: sorteo.choice(observados[codigo]) for codigo in sorted(observados)}
        completos = {**originales, **valores}
        if completos != originales and completos not in conjuntos:
            conjuntos.append(completos)
    return conjuntos


# --- informe ------------------------------------------------------------------------------------

def _estratos(caso, esperada: dict, carpeta: str, parametros: dict, umbral: int, generado: str) -> dict:
    ficha = esperada["ficha"]
    return {
        "id": caso.id, "carpeta": carpeta, "caso_generado": generado,
        "circuito_triage": circuito_caso(caso), "dificultad": dificultad_caso(caso),
        "edad": caso.edad, "nivel": caso.nivel, "acompanante": bool(caso.acompanante),
        "grupo_aviso": grupo_aviso(caso, umbral),
        "edad_box": caso.edad >= parametros["P-ADM-2"],
        "documento": caso.documento_tipo or "ninguno", "llegada": caso.llegada,
        "registro": ficha["registro"], "box_antes_del_registro": ficha["box_antes_del_registro"],
        "circuito": ficha["circuito"], "cobertura": ficha["cobertura"],
        "aviso_al_voluntariado": ficha["aviso_al_voluntariado"],
        "verificacion_requerida": esperada["verificacion"]["requerida"] is not None,
        "prohibidas": len(esperada["verificacion"]["prohibidas"]),
    }


def balance(auditorias: list[dict]) -> dict:
    campos = ("circuito_triage", "dificultad", "grupo_aviso", "acompanante", "edad_box", "documento",
              "llegada", "registro", "box_antes_del_registro", "circuito", "cobertura",
              "aviso_al_voluntariado", "verificacion_requerida", "prohibidas")
    edades = Counter("18-59" if a["edad"] < 60 else "60-74" if a["edad"] < 75 else "75+" for a in auditorias)
    return {"casos": len(auditorias), "edad": dict(sorted(edades.items())),
            **{c: dict(sorted(Counter(str(a[c]).lower() for a in auditorias).items())) for c in campos}}


def _jsonl(filas) -> bytes:
    return "".join(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n" for f in filas).encode("utf-8")


# --- punto de entrada ---------------------------------------------------------------------------

def generar_conjunto(salida: Path, *, raiz: Path = RAIZ, semilla: int = SEMILLA, k: int = CARPETAS,
                     n: int = CASOS_POR_CARPETA, candidatos: int = CANDIDATOS) -> dict:
    salida, raiz = Path(salida), Path(raiz)
    if salida.exists() or salida.is_symlink():
        raise FileExistsError(f"la salida ya existe: {salida}")
    if semilla == SEMILLA_ENTRENAMIENTO:
        raise ValueError("la semilla del conjunto de desarrollo debe ser distinta de la de entrenamiento")
    practica, destino = (raiz / "practica").resolve(), salida.resolve()
    if destino == practica or practica in destino.parents:
        raise ValueError("la salida no puede estar dentro de practica/")
    normas = raiz / "normas/original"
    costumbres = costumbres_por_tipo(json.loads((raiz / "historial_costumbres.json").read_text("utf-8")))
    umbral = int(costumbres["aviso_al_voluntariado"]["umbral_edad"])
    unidades = {p["codigo"]: p["unidad"] for p in parametros_base(normas)}
    observados = valores_observados(_leer_jsonl(raiz / "entrenamiento/trazas.jsonl"), unidades)
    huellas = huellas_excluidas(raiz)

    exclusiones, pool, pacientes = Counter(), [], set()
    for caso in generar(semilla=semilla, n=candidatos, prefijo=f"cand-{semilla}"):
        motivo = solapamiento(caso, huellas)
        if motivo is None and caso.paciente_id in pacientes:
            motivo = "paciente_id_repetido_en_candidatos"
        if motivo:
            exclusiones[motivo] += 1
            continue
        pacientes.add(caso.paciente_id)
        pool.append(caso)

    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(tempfile.mkdtemp(dir=salida.parent, prefix=f".{salida.name}-"))
    try:
        with tempfile.TemporaryDirectory(prefix="normas-dev-") as auxiliar:
            carpetas = CarpetasNormas(normas, Path(auxiliar), lambda ruta: None)
            conjuntos = sortear_valores(observados, carpetas.originales, semilla, k)
            usados, auditorias, por_carpeta = set(), [], {}
            for indice, valores in enumerate(conjuntos, 1):
                nombre = f"n{indice}"
                ruta_normas, _ = carpetas.obtener(valores)
                parametros = cargar_parametros(ruta_normas)
                disponibles = [c for c in pool if c.id not in usados]
                elegidos = seleccionar(disponibles, n, umbral,
                                       lambda caso: esperada_de(caso, parametros, costumbres))
                casos, esperadas = [], []
                for numero, (caso, esperada) in enumerate(elegidos, 1):
                    usados.add(caso.id)
                    final = replace(caso, id=f"pd-{nombre}-{numero:03d}")
                    casos.append(final.como_dict())
                    esperadas.append({**esperada, "caso": final.id})
                    auditorias.append(_estratos(final, esperada, nombre, parametros, umbral, caso.id))
                entradas = temporal / "entradas" / nombre
                entradas.mkdir(parents=True)
                shutil.copytree(ruta_normas, entradas / "normas")
                (entradas / "casos.jsonl").write_bytes(_jsonl(casos))
                (temporal / "esperadas").mkdir(exist_ok=True)
                (temporal / "esperadas" / f"esperadas-{nombre}.jsonl").write_bytes(_jsonl(esperadas))
                por_carpeta[nombre] = {
                    "parametros": dict(sorted(valores.items())),
                    "distintos_del_original": {c: v for c, v in sorted(valores.items())
                                               if v != carpetas.originales[c]},
                    "balance": balance([a for a in auditorias if a["carpeta"] == nombre]),
                }
        (temporal / "auditoria.jsonl").write_bytes(_jsonl(auditorias))
        archivos = {p.relative_to(temporal).as_posix(): _sha256(p)
                    for p in sorted(temporal.rglob("*")) if p.is_file()}
        manifiesto = {
            "version": 1,
            "comando": (f"python -m scripts.generar_prompt_dev --semilla {semilla} --carpetas {k}"
                        f" --casos-por-carpeta {n} --candidatos {candidatos}"),
            "git": _git(raiz, salida),
            "proposito": "seleccion de prompt (issue #13); nunca entrenamiento",
            "generador": "admision.generador.generar",
            "semilla": semilla,
            "candidatos": candidatos,
            "carpetas": k,
            "casos_por_carpeta": n,
            "deductor": "scripts.generar_sinteticos_entrenamiento.deducir_ficha (reglas + costumbres)",
            "costumbres": {"aviso_umbral_edad": umbral,
                           "aviso_condicion": costumbres["aviso_al_voluntariado"]["condicion"],
                           "credencial_preferida": costumbres["seleccion_credencial"]["preferido"]},
            "parametros": {"originales": dict(sorted(carpetas.originales.items())),
                           "valores_de_la_referencia": observados,
                           "regla": "un valor por parámetro sorteado entre los observados en "
                                    "entrenamiento/trazas.jsonl; conjuntos distintos entre sí y del original"},
            "cuotas_por_carpeta": cuotas(n),
            "exclusion": {"fuentes": list(FUENTES_EXCLUIDAS) + [
                              f"admision.generador semilla {SEMILLA_ENTRENAMIENTO} n={CANDIDATOS_ENTRENAMIENTO}"],
                          "identidad": "cuerpo de la nota de triage, id + nombre y edad, y cada número "
                                       "de documento/afiliado/CUIL",
                          "candidatos_excluidos": dict(sorted(exclusiones.items())),
                          "uso_de_practica": "solo practica/casos.jsonl para excluir pacientes; nunca esperadas",
                          "normas_b_c": "no se leen"},
            "por_carpeta": por_carpeta,
            "balance_total": balance(auditorias),
            "entradas_sha256": {**{r: _sha256(raiz / r) for r in (*FUENTES_EXCLUIDAS, "historial_costumbres.json")},
                                **{f"normas/original/{p.name}": _sha256(p)
                                   for p in sorted(normas.iterdir()) if p.is_file()}},
            "archivos_sha256": archivos,
            "uso": ("python -m scripts.evaluar_prompts --prompt <spec> --configuracion <config> "
                    "--casos <salida>/entradas/n<k>/casos.jsonl --esperadas <salida>/esperadas/esperadas-n<k>.jsonl "
                    "--normas <salida>/entradas/n<k>/normas --salida <nueva>"),
        }
        (temporal / "manifiesto.json").write_text(
            json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if salida.exists():
            raise FileExistsError(f"la salida ya existe: {salida}")
        os.rename(temporal, salida)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise
    return manifiesto


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", type=Path, default=RAIZ / "datos/prompt-dev/v1")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--carpetas", type=int, default=CARPETAS)
    parser.add_argument("--casos-por-carpeta", type=int, default=CASOS_POR_CARPETA)
    parser.add_argument("--candidatos", type=int, default=CANDIDATOS)
    args = parser.parse_args(argv)
    try:
        manifiesto = generar_conjunto(args.salida, semilla=args.semilla, k=args.carpetas,
                                      n=args.casos_por_carpeta, candidatos=args.candidatos)
    except (FileExistsError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"por_carpeta": {k: v["distintos_del_original"] for k, v in manifiesto["por_carpeta"].items()},
                      "exclusion": manifiesto["exclusion"]["candidatos_excluidos"],
                      "balance_total": manifiesto["balance_total"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
