"""Genera el dataset de conversaciones de entrenamiento v2 (issue #8) en una carpeta nueva.

Tres fuentes, todas en el formato que ve el agente con el prompt y el buscador elegidos:

- `referencia`: las 800 trazas de `entrenamiento/trazas.jsonl` (la cátedra entrenó con ellas);
- `historial`: las filas que la selección de Yas conserva (`correcta`);
- `sintetico`: casos de `admision.generador` con valores de parámetros variados.

Cada búsqueda se vuelve a ejecutar con el buscador real (armado como en `entrega.py`) sobre una
copia de `normas/original` con los valores de parámetros de esa conversación, y cada ficha se
revalida con las reglas y costumbres del clasificador. Lo que no se puede reconstruir o no pasa
se excluye y se cuenta por motivo; la cobertura de evidencia se mide y se informa, no se filtra.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile

from admision.casos import cargar_casos
from admision.entorno import Entorno, nota_de_triage
from admision.generador import generar
from admision_tp.buscador import VARIANTES, BuscadorBM25
from admision_tp.conversaciones import validar_archivo, validar_registro
from admision_tp.conversion_trazas import convertir_traza
from admision_tp.normas_variables import CarpetasNormas, parametros_base, valor_desde_texto
from admision_tp.prompts import RUTA_BASELINE, PromptSpec
from admision_tp.trazas_normativas import fila_desde_traza, parametros_observados, secciones_en
from scripts.clasificar_historial import (
    aviso_segun_costumbre,
    cargar_parametros,
    circuito_esperado,
    entradas_cobertura,
    fuente_cobertura,
    razones_normativas,
    sigue_costumbre_credencial,
)
from scripts.generar_sinteticos_entrenamiento import (
    circuito_caso,
    costumbres_por_tipo,
    credencial_preferida,
    deducir_ficha,
    dificultad_caso,
    seleccionar_casos,
)
from scripts.seleccionar_entrenamiento import (
    cargar_clasificacion,
    cargar_filas,
    costumbre_aviso,
    seleccionar_ids,
)

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 12012
CANDIDATOS = 5000
SECCION_CIRCUITO = {"ecg": "NOR-TDE-01", "codigo_acv": "NOR-TDE-02", "codigo_sepsis": "NOR-TDE-03"}
# Consultas del guion sintético: el mismo vocabulario que las 800 trazas de referencia.
CONSULTAS = {
    "box": "registro del paciente y ubicación en el box",
    "documento": "registro con pasaporte o cédula",
    "ecg": "dolor torácico plazo",
    "codigo_acv": "signos neurológicos focales de inicio súbito plazo",
    "codigo_sepsis": "código sepsis plazo",
    "ambulancia": "ingreso en ambulancia plazo del circuito",
    "art": "accidente de trabajo ART",
    "consulta": "consulta de la cobertura sin credencial",
    "verificacion": "verificación de afiliación",
    "terminacion": "número de afiliado con terminación reservada",
    "sin_cobertura": "afiliación inactiva paciente sin cobertura",
}
ENTRADAS = ("entrenamiento/trazas.jsonl", "historial/trazas.jsonl", "historial/historial.csv",
            "historial_clasificado.csv", "historial_costumbres.json", "practica/casos.jsonl")


class Excluida(ValueError):
    """Conversación descartada; `motivo` es una clave estable para el manifiesto."""

    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo


# --- buscador y búsquedas ---------------------------------------------------------------------

def fabrica_buscador(variante: str | None):
    """Arma el buscador como la entrega: sin variante, `entrega.construir_buscador` tal cual."""
    if variante is None:
        from entrega import construir_buscador
        return construir_buscador
    if variante not in VARIANTES:
        raise ValueError(f"variante desconocida: {variante}; opciones: {', '.join(VARIANTES)}")
    return lambda ruta: BuscadorBM25(ruta, variante=variante)


def observar_busqueda(buscador, consulta: str) -> str:
    """Observación exacta de `buscar_normativa` del entorno del kit para ese buscador."""
    observacion, _ = Entorno(None, buscador).paso("buscar_normativa", {"consulta": consulta})
    return observacion


# --- parámetros ---------------------------------------------------------------------------------

def valores_de_traza(traza, unidades: dict[str, str]) -> dict[str, int]:
    try:
        textos = parametros_observados(traza)
    except ValueError:
        raise Excluida("parametros_contradictorios") from None
    try:
        return {codigo: valor_desde_texto(codigo, texto, unidades) for codigo, texto in textos.items()}
    except ValueError:
        raise Excluida("valor_de_parametro_no_interpretable") from None


def valores_observados(trazas, unidades) -> dict[str, list[int]]:
    """Valores que toma cada parámetro en un conjunto de trazas (los de la referencia varían)."""
    vistos = defaultdict(set)
    for traza in trazas:
        try:
            valores = valores_de_traza(traza, unidades)
        except Excluida:
            continue  # se cuenta al convertirla
        for codigo, valor in valores.items():
            vistos[codigo].add(valor)
    return {codigo: sorted(valores) for codigo, valores in sorted(vistos.items())}


def completar_parametros(fila, observados, originales, candidatos) -> tuple[dict[str, int], list[str]]:
    """Valores observados + original en los no observados, si la ficha no depende de ellos.

    "No depende" se prueba: la ficha registrada sigue sin violar normas con cualquier
    combinación de valores conocidos (los de la referencia y el original) de los no observados.
    """
    faltantes = sorted(c for c in candidatos if c not in observados)
    completos = {**originales, **observados}
    opciones = {c: sorted(set(candidatos[c]) | {originales[c]}) for c in faltantes}
    for combinacion in itertools.product(*opciones.values()):
        if razones_normativas(fila, {**completos, **dict(zip(faltantes, combinacion))}):
            # Para el informe: qué parámetros cambian la ficha por sí solos.
            solos = [c for c in faltantes
                     if any(razones_normativas(fila, {**completos, c: v}) for v in opciones[c])]
            raise Excluida("parametro_relevante_no_observado:" + ",".join(solos or faltantes))
    return completos, faltantes


# --- revalidación y evidencia -------------------------------------------------------------------

def revalidar(fila, parametros, costumbres) -> None:
    razones = razones_normativas(fila, parametros)
    if razones:
        raise Excluida("ficha_viola_normas:" + ",".join(razones))
    if aviso_segun_costumbre(fila, costumbres["aviso_al_voluntariado"]) != (fila["aviso_al_voluntariado"] == "sí"):
        raise Excluida("desvio_costumbre_aviso")
    if not sigue_costumbre_credencial(fila, costumbres["seleccion_credencial"]):
        raise Excluida("desvio_costumbre_credencial")


def secciones_rectoras(fila, parametros) -> list[str]:
    """Secciones cuyo contenido decide la ficha de este caso (ver docs/conversaciones-entrenamiento.md)."""
    secciones = ["NOR-ADM-01"]
    if int(fila["edad"]) >= parametros["P-ADM-2"]:
        secciones.append("NOR-ADM-02")
    if fila["documento_tipo"] and fila["documento_tipo"] != "DNI":
        secciones.append("NOR-ADM-03")
    fuente = fuente_cobertura(fila)
    if fuente and fuente[0] == "art":
        secciones.append("NOR-COB-02")
    elif fuente and fuente[2].endswith(str(parametros["P-COB-2"])):
        secciones.append("NOR-COB-04")
    circuito = circuito_esperado(fila)
    if circuito in SECCION_CIRCUITO:
        secciones.append(SECCION_CIRCUITO[circuito])
        if fila["llegada"] == "en ambulancia":
            secciones.append("NOR-TDE-04")
    return secciones


def evidencia(fila, parametros, observaciones, fundamentos) -> dict:
    recuperadas = sorted({s for obs in observaciones for s in secciones_en(obs)})
    rectoras = secciones_rectoras(fila, parametros)
    return {
        "rectoras": rectoras,
        "recuperadas": recuperadas,
        "rectoras_faltantes": [s for s in rectoras if s not in recuperadas],
        "fundamentos_faltantes": [s for s in dict.fromkeys(fundamentos) if s not in recuperadas],
    }


# --- contaminación con práctica -----------------------------------------------------------------

def _cuerpo_triage(nota: str) -> str:
    return "\n".join(nota.splitlines()[1:])  # sin la línea con el id del paciente


def _numeros(fila) -> set[str]:
    numeros = {fila["documento_numero"]} if fila["documento_numero"] else set()
    for texto in (fila["credenciales"], fila["cobertura_declarada"]):
        numeros |= {numero for _, _, numero in entradas_cobertura(texto)}
    if fila["art_declarada"]:
        numeros.add(fila["art_declarada"].rpartition(" CUIL ")[2])
    if fila["numero_verificado"]:
        numeros.add(fila["numero_verificado"])
    return numeros


def huellas_practica(ruta: Path) -> dict:
    """Triages y números de documento/afiliado de los casos de práctica (nunca sus fichas)."""
    triages, numeros = set(), set()
    for caso in cargar_casos(ruta):
        triages.add(_cuerpo_triage(nota_de_triage(caso)))
        numeros |= {caso.documento_numero} if caso.documento_numero else set()
        numeros |= {c[2] for c in caso.credenciales}
        numeros |= {caso.declarada[2]} if caso.declarada else set()
        numeros |= {caso.art[1]} if caso.art else set()
    return {"triages": triages, "numeros": numeros}


def verificar_no_practica(nota: str, fila, huellas) -> None:
    if _cuerpo_triage(nota) in huellas["triages"]:
        raise Excluida("practica_triage_identico")
    if _numeros(fila) & huellas["numeros"]:
        raise Excluida("practica_documento_o_afiliado_compartido")


# --- fuentes ------------------------------------------------------------------------------------

def _leer_jsonl(ruta: Path) -> list[dict]:
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def convertir_trazas(trazas, *, origen, referencia, contexto, valores_fijos=None):
    """Re-ejecuta búsquedas y revalida cada traza; devuelve (registros, auditorías, exclusiones)."""
    registros, auditorias, excluidas = [], [], Counter()
    for traza in trazas:
        try:
            registro, auditoria = _convertir_una(traza, origen, referencia, contexto, valores_fijos)
        except Excluida as error:
            excluidas[error.motivo] += 1
            continue
        registros.append(registro)
        auditorias.append(auditoria)
    return registros, auditorias, excluidas


def _convertir_una(traza, origen, referencia, contexto, valores_fijos):
    try:
        fila = fila_desde_traza(traza)
    except (ValueError, KeyError):
        raise Excluida("traza_no_reconstruible") from None
    if valores_fijos is not None and {k: v for k, v in fila.items() if k != "nombre"} != valores_fijos[traza["id"]]:
        raise Excluida("traza_distinta_del_csv")
    verificar_no_practica(traza["pasos"][0]["resultado"], fila, contexto["practica"])
    observados = valores_de_traza(traza, contexto["unidades"])
    carpetas = contexto["carpetas"]
    if origen == "historial":
        distintos = [c for c, v in observados.items() if v != carpetas.originales[c]]
        if distintos:
            raise Excluida("historial_con_valores_no_originales")
    parametros, supuestos = completar_parametros(
        fila, observados, carpetas.originales, contexto["candidatos"])
    revalidar(fila, parametros, contexto["costumbres"])
    _, buscador = carpetas.obtener(parametros)
    traza, curado = curar_traza(traza, buscador, secciones_rectoras(fila, parametros))
    observaciones = []

    def buscar(consulta):
        observaciones.append(observar_busqueda(buscador, consulta))
        return observaciones[-1]

    registro = convertir_traza(traza, origen=origen, referencia=referencia,
                               prompt=contexto["prompt"], buscar=buscar)
    fundamentos = traza["pasos"][-1]["argumentos"].get("fundamentos", [])
    auditoria = _auditoria(registro, contexto, parametros, supuestos,
                           evidencia(fila, parametros, observaciones, fundamentos))
    auditoria.update(curado)
    return registro, auditoria


def curar_traza(traza, buscador, rectoras) -> tuple[dict, dict]:
    """Transformaciones de curado sobre una copia de la traza (las búsquedas, del buscador real).

    1. Si la ficha depende de la ART (NOR-COB-02 rectora) y ninguna búsqueda la recupera, se agrega
       una búsqueda «accidente de trabajo ART» justo después de la consulta combinada de ART; la
       consulta original no se modifica. Si no hay dónde insertarla o no recupera NOR-COB-02, se excluye.
    2. Se quitan de `fundamentos` las secciones que ninguna búsqueda recuperó (si queda vacío,
       se quita la clave, que es optativa).
    """
    pasos = [dict(paso, argumentos=dict(paso["argumentos"])) for paso in traza["pasos"]]
    recuperadas = set()
    for paso in pasos:
        if paso["herramienta"] == "buscar_normativa":
            recuperadas |= set(secciones_en(observar_busqueda(buscador, paso["argumentos"]["consulta"])))
    insertadas = []
    if "NOR-COB-02" in rectoras and "NOR-COB-02" not in recuperadas:
        combinada = next((i for i, paso in enumerate(pasos) if paso["herramienta"] == "buscar_normativa"
                          and paso["argumentos"]["consulta"].startswith(CONSULTAS["art"])), None)
        if combinada is None:
            raise Excluida("nor_cob_02_sin_consulta_de_art")
        observacion = observar_busqueda(buscador, CONSULTAS["art"])
        if "NOR-COB-02" not in secciones_en(observacion):
            raise Excluida("busqueda_insertada_no_recupera_nor_cob_02")
        pasos.insert(combinada + 1, {"orden": 0, "herramienta": "buscar_normativa",
                                     "argumentos": {"consulta": CONSULTAS["art"]}, "resultado": observacion})
        recuperadas |= set(secciones_en(observacion))
        insertadas.append({"despues_de_paso": combinada + 1, "consulta": CONSULTAS["art"],
                           "motivo": "NOR-COB-02 rectora no recuperada"})
    for orden, paso in enumerate(pasos, 1):
        paso["orden"] = orden
    final = pasos[-1]["argumentos"]
    quitados = []
    if "fundamentos" in final:
        quitados = [s for s in final["fundamentos"] if s not in recuperadas]
        conservados = [s for s in final["fundamentos"] if s in recuperadas]
        if conservados:
            final["fundamentos"] = conservados
        else:
            del final["fundamentos"]
    return {**traza, "pasos": pasos}, {"busquedas_insertadas": insertadas, "fundamentos_quitados": quitados}


def _auditoria(registro, contexto, parametros, supuestos, cobertura) -> dict:
    variables = contexto["candidatos"]
    return {"id": registro["id"], "origen": registro["origen"],
            "parametros": {c: parametros[c] for c in variables},
            "parametros_supuestos_originales": supuestos, **cobertura}


def _registrar(mensajes, entorno, prompt, herramienta, argumentos, observaciones):
    mensajes.append({"role": "assistant", "content": json.dumps(
        {"herramienta": herramienta, "argumentos": argumentos}, ensure_ascii=False, allow_nan=False)})
    observacion, terminado = entorno.paso(herramienta, argumentos)
    if herramienta == "registrar_decision":
        return
    if terminado or observacion.startswith("Error:"):
        raise ValueError(f"el guion sintético falló en {herramienta}")
    if herramienta == "buscar_normativa":
        observaciones.append(observacion)
    mensajes.append({"role": "user", "content": prompt.observacion_para(herramienta, observacion)})


def guion_sintetico(caso, ficha, buscador, costumbres, prompt) -> tuple[list[dict], list[str]]:
    """Recorre el caso en el entorno del kit; busca cada sección que decide o fundamenta la ficha."""
    entorno, observaciones = Entorno(caso, buscador), []
    mensajes = [{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.mensaje_inicial_para(caso.paciente_id)}]

    def paso(herramienta, **argumentos):
        _registrar(mensajes, entorno, prompt, herramienta, argumentos, observaciones)

    def buscar(clave):
        paso("buscar_normativa", consulta=CONSULTAS[clave])

    paso("leer_triage", paciente_id=caso.paciente_id)
    buscar("box")
    if caso.documento_tipo and caso.documento_tipo != "DNI":
        buscar("documento")
    circuito = ficha["circuito"]
    if circuito != "ninguno":
        buscar(circuito)
        if caso.llegada == "en ambulancia":
            buscar("ambulancia")
    quien = "paciente" if caso.puede_responder else ("acompanante" if caso.acompanante else None)
    if caso.trabajo:
        buscar("art")
        if quien:
            paso("preguntar", paciente_id=caso.paciente_id, a_quien=quien, tema="art")
    if not caso.credenciales:
        buscar("consulta")
        if quien:
            paso("preguntar", paciente_id=caso.paciente_id, a_quien=quien, tema="cobertura")
    if caso.trabajo:
        verificar = caso.art
    elif len(caso.credenciales) > 1:
        verificar = credencial_preferida(caso, costumbres)[1:]
    elif caso.credenciales:
        verificar = caso.credenciales[0][1:]
    elif caso.declarada and quien:
        verificar = caso.declarada[1:]
    else:
        verificar = None
    if verificar:
        buscar("verificacion")
        if not caso.trabajo:
            buscar("terminacion")
        paso("verificar_afiliacion", entidad=verificar[0], numero_afiliado=verificar[1])
    if ficha["cobertura"] == "sin_cobertura":
        buscar("sin_cobertura")
    paso("registrar_decision", **ficha)
    if not entorno.terminado or entorno.decision != ficha:
        raise ValueError(f"el entorno rechazó la ficha sintética {caso.id}")
    return mensajes, observaciones


def generar_sinteticos(contexto, semilla, candidatos):
    """Casos del generador con valores sorteados entre los observados en la referencia."""
    sorteo = random.Random(f"{semilla}:parametros")
    variables = contexto["candidatos"]
    casos = seleccionar_casos(generar(semilla=semilla, n=candidatos, prefijo=f"cand-{semilla}"))
    registros, auditorias, excluidas = [], [], Counter()
    for indice, caso in enumerate(casos, 1):
        valores = {codigo: sorteo.choice(variables[codigo]) for codigo in sorted(variables)}
        ruta, buscador = contexto["carpetas"].obtener(valores)
        parametros = cargar_parametros(ruta)
        caso_final = replace(caso, id=f"S-{semilla}-{indice:04d}")
        try:
            fila, ficha = deducir_ficha(caso, parametros, contexto["costumbres"])
            verificar_no_practica(nota_de_triage(caso_final), fila, contexto["practica"])
        except Excluida as error:
            excluidas[error.motivo] += 1
            continue
        except ValueError:
            excluidas["deduccion_sintetica_invalida"] += 1
            continue
        mensajes, observaciones = guion_sintetico(
            caso_final, ficha, buscador, contexto["costumbres"], contexto["prompt"])
        registro = {"version": 1, "id": caso_final.id, "origen": "sintetico",
                    "referencias": [f"generador:admision.generador:semilla={semilla}:caso={caso.id}",
                                    "normas:normas/original+parametros-variados",
                                    "costumbres:historial_costumbres.json"],
                    "messages": mensajes}
        validar_registro(registro, contexto["prompt"].formato_observacion)
        registros.append(registro)
        auditoria = _auditoria(registro, contexto, contexto["carpetas"].completar(valores), [],
                               evidencia(fila, parametros, observaciones, ficha["fundamentos"]))
        auditoria.update(circuito=circuito_caso(caso), dificultad=dificultad_caso(caso),
                         busquedas_insertadas=[], fundamentos_quitados=[])
        auditorias.append(auditoria)
    return registros, auditorias, excluidas


# --- informe ------------------------------------------------------------------------------------

def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _git(raiz: Path, salida: Path | None = None) -> dict:
    """Commit y estado del árbol. La propia salida (y su carpeta temporal) no cuenta como cambio:
    se escribe dentro del repo antes de registrar el manifiesto y, si contara, siempre daría dirty."""
    def correr(*args):
        return subprocess.run(["git", *args], cwd=raiz, capture_output=True, text=True, check=True).stdout
    excluir = []
    if salida is not None:
        try:
            relativa = Path(salida).resolve().relative_to(Path(raiz).resolve())
        except ValueError:
            relativa = None
        if relativa is not None:
            padre = relativa.parent.as_posix()
            excluir = [f":(exclude,glob){relativa.as_posix()}/**",
                       f":(exclude,glob){padre}/.{relativa.name}-*/**"]
    try:
        estado = correr("status", "--porcelain", "--untracked-files=all", "--", ".", *excluir)
        return {"commit": correr("rev-parse", "HEAD").strip(), "dirty": bool(estado.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _resumen(numeros) -> dict:
    numeros = list(numeros)
    if not numeros:
        return {"n": 0}
    return {"n": len(numeros), "min": min(numeros), "mediana": statistics.median(numeros),
            "media": round(statistics.fmean(numeros), 2), "max": max(numeros)}


def estadisticas(registros, auditorias, originales) -> dict:
    por_origen = defaultdict(list)
    for registro, auditoria in zip(registros, auditorias, strict=True):
        por_origen[registro["origen"]].append((registro, auditoria))
    salida = {}
    for origen, pares in sorted(por_origen.items()):
        audits = [a for _, a in pares]
        faltan_rectoras = Counter(s for a in audits for s in a["rectoras_faltantes"])
        faltan_fundamentos = Counter(s for a in audits for s in a["fundamentos_faltantes"])
        fracciones = Counter(
            f"{len(a['rectoras']) - len(a['rectoras_faltantes'])}/{len(a['rectoras'])}" for a in audits)
        conjuntos = Counter(tuple(sorted(a["parametros"].items())) for a in audits)
        salida[origen] = {
            "conversaciones": len(pares),
            "cobertura_evidencia": {
                "con_todas_las_rectoras": sum(not a["rectoras_faltantes"] for a in audits),
                "rectoras_recuperadas_sobre_requeridas": dict(sorted(fracciones.items())),
                "rectora_faltante_por_seccion": dict(sorted(faltan_rectoras.items())),
                "con_todos_los_fundamentos": sum(not a["fundamentos_faltantes"] for a in audits),
                "fundamento_faltante_por_seccion": dict(sorted(faltan_fundamentos.items())),
            },
            "curado": {
                "busquedas_insertadas": sum(len(a.get("busquedas_insertadas", [])) for a in audits),
                "conversaciones_con_fundamentos_quitados": sum(bool(a.get("fundamentos_quitados")) for a in audits),
                "conversaciones_sin_fundamentos": sum(
                    '"fundamentos"' not in r["messages"][-1]["content"] for r, _ in pares),
                "fundamentos_quitados_por_seccion": dict(sorted(Counter(
                    s for a in audits for s in a.get("fundamentos_quitados", [])).items())),
            },
            "parametros": {
                "conjuntos_distintos": len(conjuntos),
                "con_todos_los_valores_originales": sum(
                    all(v == originales[c] for c, v in a["parametros"].items()) for a in audits),
                "con_supuestos_originales": sum(bool(a["parametros_supuestos_originales"]) for a in audits),
                "valores": {c: dict(sorted(Counter(a["parametros"][c] for a in audits).items()))
                            for c in sorted(audits[0]["parametros"])},
            },
            "caracteres_por_conversacion": _resumen(
                sum(len(m["content"]) for m in r["messages"]) for r, _ in pares),
            "llamadas_por_conversacion": _resumen((len(r["messages"]) - 1) // 2 for r, _ in pares),
        }
    return salida


def _jsonl(registros) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in registros).encode("utf-8")


# --- punto de entrada ---------------------------------------------------------------------------

def generar_dataset(salida: Path, *, raiz: Path = RAIZ, ruta_prompt: Path = RUTA_BASELINE,
                    variante: str | None = None, semilla: int = SEMILLA, candidatos: int = CANDIDATOS) -> dict:
    salida, raiz = Path(salida), Path(raiz)
    if salida.exists() or salida.is_symlink():
        raise FileExistsError(f"la salida ya existe: {salida}")
    prompt = PromptSpec.cargar(ruta_prompt)
    fabrica = fabrica_buscador(variante)
    normas = raiz / "normas/original"
    with tempfile.TemporaryDirectory(prefix="normas-variadas-") as temporal:
        carpetas = CarpetasNormas(normas, Path(temporal), fabrica)
        _, buscador_original = carpetas.obtener({})
        unidades = {p["codigo"]: p["unidad"] for p in parametros_base(normas)}
        referencia = _leer_jsonl(raiz / "entrenamiento/trazas.jsonl")
        contexto = {
            "prompt": prompt, "carpetas": carpetas, "unidades": unidades,
            "candidatos": valores_observados(referencia, unidades),
            "costumbres": costumbres_por_tipo(json.loads((raiz / "historial_costumbres.json").read_text("utf-8"))),
            "practica": huellas_practica(raiz / "practica/casos.jsonl"),
        }
        ref, aud_ref, exc_ref = convertir_trazas(
            referencia, origen="referencia", referencia="entrenamiento/trazas.jsonl", contexto=contexto)

        filas = cargar_filas(raiz / "historial/historial.csv")
        ids, seleccion = seleccionar_ids(filas, cargar_clasificacion(raiz / "historial_clasificado.csv"),
                                         costumbre_aviso(raiz / "historial_costumbres.json"))
        elegidas = set(ids)
        historial = [t for t in _leer_jsonl(raiz / "historial/trazas.jsonl") if t["id"] in elegidas]
        csv_por_id = {f["id"]: {k: v for k, v in f.items() if k not in ("fecha", "operador")}
                      for f in filas if f["id"] in elegidas}
        his, aud_his, exc_his = convertir_trazas(
            historial, origen="historial", referencia="historial/trazas.jsonl", contexto=contexto,
            valores_fijos=csv_por_id)
        sin, aud_sin, exc_sin = generar_sinteticos(contexto, semilla, candidatos)
        carpetas_usadas = len(carpetas)
        busqueda = {"variante": getattr(buscador_original, "variante", None),
                    "clase": type(buscador_original).__name__,
                    "configuracion": asdict(buscador_original.config) if hasattr(buscador_original, "config") else None,
                    "k_entorno": getattr(buscador_original, "k_entorno", None),
                    "construido_con": "entrega.construir_buscador" if variante is None
                    else "admision_tp.buscador.BuscadorBM25(ruta, variante)"}

    registros, auditorias = ref + his + sin, aud_ref + aud_his + aud_sin
    datos, datos_auditoria = _jsonl(registros), _jsonl(auditorias)
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(tempfile.mkdtemp(dir=salida.parent, prefix=f".{salida.name}-"))
    try:
        (temporal / "conversaciones.jsonl").write_bytes(datos)
        (temporal / "auditoria.jsonl").write_bytes(datos_auditoria)
        validacion = validar_archivo(temporal / "conversaciones.jsonl", prompt.formato_observacion)
        if validacion["errores"]:
            raise ValueError(f"dataset inválido: {len(validacion['errores'])} errores")
        manifiesto = {
            "version": 2,
            "contrato_conversaciones": 1,
            "comando": ("python -m scripts.generar_conversaciones"
                        f" --prompt {_relativa(ruta_prompt, raiz)}"
                        + (f" --buscador-variante {variante}" if variante else "")
                        + f" --semilla {semilla} --candidatos {candidatos}"),
            "git": _git(raiz, salida),
            "entradas_sha256": {**{r: _sha256(raiz / r) for r in ENTRADAS},
                                **{f"normas/original/{p.name}": _sha256(p) for p in sorted(normas.iterdir())
                                   if p.is_file()}},
            "uso_de_practica": "solo practica/casos.jsonl, para excluir casos repetidos; nunca esperadas",
            "prompt": {"nombre": prompt.nombre, "version": prompt.version, "sha256": prompt.sha256,
                       "archivo": _relativa(ruta_prompt, raiz),
                       "textos_usados": ["system_prompt", "mensaje_inicial", "formato_observacion"],
                       "mensaje_json_invalido": "no aplica: las demostraciones no tienen respuestas inválidas"},
            "buscador": busqueda,
            "semilla": semilla,
            "candidatos_sinteticos": candidatos,
            "parametros": {
                "originales": dict(sorted(carpetas.originales.items())),
                "valores_de_la_referencia": contexto["candidatos"],
                "carpetas_de_normas_distintas": carpetas_usadas,
                "regla_no_observados": "valor original solo si la ficha no viola normas con ningún "
                                       "valor conocido; si no, la traza se excluye",
            },
            "curado_de_trazas": [
                "si NOR-COB-02 es rectora y no se recupera, se inserta la búsqueda «accidente de trabajo ART» "
                "después de la consulta combinada de ART (la original no cambia)",
                "se quitan de fundamentos las secciones no recuperadas en la conversación; "
                "si no queda ninguna, se quita la clave",
            ],
            "por_origen": dict(Counter(r["origen"] for r in registros)),
            "entradas_por_origen": {"referencia": len(referencia), "historial": len(historial),
                                    "sintetico": len(sin) + sum(exc_sin.values())},
            "seleccion_historial": seleccion,
            "exclusiones": {"referencia": dict(sorted(exc_ref.items())),
                            "historial": dict(sorted(exc_his.items())),
                            "sintetico": dict(sorted(exc_sin.items()))},
            "estadisticas": estadisticas(registros, auditorias, carpetas.originales),
            "longitudes": "caracteres, no tokens: la tokenización real no se ejecutó",
            "salida": {"conversaciones_sha256": validacion["sha256"], "registros": validacion["registros"],
                       "validos": validacion["validos"], "auditoria_sha256": hashlib.sha256(datos_auditoria).hexdigest()},
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


def _relativa(ruta: Path, raiz: Path) -> str:
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(Path(raiz).resolve()).as_posix()
    except ValueError:
        return ruta.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", type=Path,
                        help="carpeta nueva (por defecto datos/conversaciones/v2-<variante>)")
    parser.add_argument("--prompt", type=Path, default=RUTA_BASELINE, help="PromptSpec JSON")
    parser.add_argument("--buscador-variante", default=None,
                        help="por defecto, la de entrega.construir_buscador (BUSCADOR_VARIANTE o baseline)")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--candidatos", type=int, default=CANDIDATOS)
    args = parser.parse_args()
    salida = args.salida
    if salida is None:
        nombre = args.buscador_variante or fabrica_buscador(None)(RAIZ / "normas/original").variante
        salida = RAIZ / "datos/conversaciones" / f"v2-{nombre}"
    try:
        manifiesto = generar_dataset(salida, ruta_prompt=args.prompt, variante=args.buscador_variante,
                                     semilla=args.semilla, candidatos=args.candidatos)
    except FileExistsError as error:
        parser.exit(1, f"{error}\n")
    resumen = {k: manifiesto[k] for k in ("por_origen", "exclusiones", "salida")}
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
