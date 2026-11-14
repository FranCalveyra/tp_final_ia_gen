"""Análisis de errores por corrida (issue #14): qué controles fallan, en qué tipos de caso y por qué.

    python -m scripts.analizar_errores analizar --corrida corridas/kaggle/v22/baseline-original-n50-l4 \\
        --salida corridas/analisis/<nueva>
    python -m scripts.analizar_errores comparar --antes corridas/kaggle/v22/baseline-original-n50-l4 \\
        --despues corridas/kaggle/v22/prefijos-original-n50-l4 --salida corridas/analisis/<nueva>

Acepta una carpeta de `infra/kaggle_baseline.py` o de una corrida de `infra/kaggle_evaluacion.py` (`runner.json` +
`salida/trazas.jsonl`), una del runner de prompts del #13 (`manifiesto.json` + `trazas.jsonl` + `entradas/`) o una con
solo trazas (`--version` obligatorio, sin procedencia). Puntúa siempre con `admision.correr.puntuar` y, si la carpeta
ya trae `resultados.jsonl`/`resumen.json`, verifica que coincidan. Cada control fallado recibe una fuente (formato,
costumbre, buscador o modelo) según reglas explícitas: ver docs/informe/analisis-errores.md. Escribe en una carpeta nueva
`analisis.json` (o `comparacion.json`) con cada número y su procedencia, y el reporte Markdown.

Las esperadas y los casos se usan solo para diagnosticar después de jugar; nunca llegan al agente. Analizar la
práctica sirve para diagnosticar y reportar: no para elegir prompt, checkpoint ni hiperparámetros (eso va en el
conjunto de desarrollo).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
import re
import sys
import time
import unicodedata

from admision.casos import Caso, cargar_casos, cargar_esperadas, leer_jsonl
from admision.comparar import CONTROLES
from admision.correr import puntuar
from admision.ejemplo import extraer_json
from admision.entorno import MAX_LLAMADAS, nota_de_triage
from admision_tp.trazas_normativas import fila_triage, secciones_en
from scripts.clasificar_historial import cargar_parametros, circuito_esperado
from scripts.generar_conversaciones import secciones_rectoras
from scripts.tabla_evaluacion import RAIZ, _checkpoint, _git, _mostrar, _sha256

FUENTES = ("formato", "costumbre", "buscador", "modelo")
SUBTIPOS_MODELO = ("no_busco", "busco_y_decidio_mal")
# Familia normativa de cada control (None: lo decide una costumbre no escrita).
FAMILIA = {"registro": "ADM", "numero_documento": "ADM", "box_antes_del_registro": "ADM",
           "circuito": "TDE", "plazo_circuito_min": "TDE",
           "cobertura": "COB", "entidad_a_notificar": "COB", "verificacion": "COB",
           "aviso_al_voluntariado": None}
# Una búsqueda es relevante para una familia si su consulta tiene alguna de estas raíces (texto sin tildes)
# o si devolvió alguna sección de la familia. Heurística: ver docs/informe/analisis-errores.md.
PALABRAS = {
    "ADM": r"\b(registr|document|dni|pasaporte|cedula|box|identific|admision|mayor|edad|anos)",
    "TDE": r"\b(circuito|plazo|torac|ecg|electrocardio|acv|neurolog|sepsis|ambulancia|tiempo)",
    "COB": r"\b(cobertura|obra social|prepaga|art\b|afilia|credencial|verific|notific|entidad|servicio social"
           r"|accidente|trabaj|sin cobertura)",
}
# Herramientas cuyos argumentos inválidos impiden decidir los controles de la familia.
HERRAMIENTAS_FAMILIA = {"COB": ("preguntar", "verificar_afiliacion")}
PREFIJO_GUARDA = "[agente] Caso terminado sin registrar decisión"
MINIMO_GRUPO = 3


class CorridaInvalida(ValueError):
    """La carpeta no tiene lo necesario para analizarla sin adivinar."""


# --- texto ---------------------------------------------------------------------------------------

def _plano(texto) -> str:
    t = unicodedata.normalize("NFD", str(texto).lower())
    return " ".join("".join(c for c in t if unicodedata.category(c) != "Mn").split())


def _norm(texto) -> str:
    return _plano(texto).strip()


# --- carga de la corrida -------------------------------------------------------------------------

def _primera(carpeta: Path, *relativas: str) -> Path | None:
    return next((carpeta / r for r in relativas if (carpeta / r).is_file()), None)


def _leer_json(ruta: Path | None):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta else None


def procedencia_de(carpeta: Path) -> tuple[dict, list[str]]:
    """Procedencia registrada por el runner (runner.json o manifiesto.json) y avisos."""
    avisos = []
    if (runner := _leer_json(_primera(carpeta, "runner.json"))) is not None:
        conf, maquina = runner.get("configuracion") or {}, runner.get("maquina") or {}
        if not runner.get("prompt"):
            avisos.append("runner.json no registra el prompt (corrida anterior al registro de prompt en el runner)")
        if maquina.get("sucio") is not False:
            avisos.append("árbol Git sucio o sin dato: el commit no reconstruye solo el código")
        return {"origen": "runner.json", "version": runner.get("variante"), "estado": runner.get("estado"),
                "commit": maquina.get("commit"), "sucio": maquina.get("sucio"),
                "modelo": f"{conf.get('modelo')}@{conf.get('revision')}", "adapter": _checkpoint(conf),
                "prompt": runner.get("prompt"), "buscador": runner.get("buscador"),
                "lote": conf.get("lote"), "gpus": maquina.get("gpus"),
                "inicio": runner.get("inicio"), "fin": runner.get("fin")}, avisos
    if (manifiesto := _leer_json(_primera(carpeta, "manifiesto.json"))) is not None:
        conf, codigo = manifiesto.get("configuracion") or {}, manifiesto.get("codigo") or {}
        if codigo.get("dirty") is not False:
            avisos.append("árbol Git sucio o sin dato: el commit no reconstruye solo el código")
        if not manifiesto.get("comparable", True):
            avisos.append(f"manifiesto no comparable (propósito {manifiesto.get('proposito')})")
        adapter = conf.get("adapter_repo")
        return {"origen": "manifiesto.json", "version": None, "estado": manifiesto.get("estado"),
                "commit": codigo.get("commit"), "sucio": codigo.get("dirty"),
                "modelo": f"{conf.get('modelo')}@{conf.get('revision')}",
                "adapter": f"{adapter}@{conf.get('adapter_revision')}" if adapter else "sin entrenar",
                "prompt": manifiesto.get("prompt"), "buscador": {"variante": conf.get("buscador_variante")},
                "lote": conf.get("lote"), "proposito": manifiesto.get("proposito")}, avisos
    return {"origen": None}, ["sin runner.json ni manifiesto.json: no hay procedencia del código, prompt ni modelo"]


def cargar_corrida(carpeta: Path, *, version: str | None = None, casos: Path | None = None,
                   esperadas: Path | None = None, normas: Path | None = None) -> dict:
    """Trazas, casos, esperadas, parámetros y procedencia de una carpeta de corrida."""
    carpeta = Path(carpeta)
    ruta_trazas = _primera(carpeta, "salida/trazas.jsonl", "trazas.jsonl")
    if ruta_trazas is None:
        raise CorridaInvalida(f"{carpeta}: no hay trazas.jsonl ni salida/trazas.jsonl")
    procedencia, avisos = procedencia_de(carpeta)
    entradas = carpeta / "entradas"
    if procedencia["origen"] == "runner.json" and procedencia.get("estado") != "terminada":
        avisos.append(f"corrida en estado {procedencia.get('estado')}")
    version = version or procedencia.get("version")
    if (entradas / "casos.jsonl").is_file():
        casos = casos or entradas / "casos.jsonl"
        esperadas = esperadas or entradas / "esperadas.jsonl"
        normas = normas or entradas / "normas"
    if version is None and not (casos and esperadas and normas):
        raise CorridaInvalida(f"{carpeta}: sin procedencia ni entradas; pasá --version (o --casos/--esperadas/--normas)")
    casos = Path(casos or RAIZ / "practica/casos.jsonl")
    esperadas = Path(esperadas or RAIZ / f"practica/esperadas-{version}.jsonl")
    normas = Path(normas or RAIZ / f"normas/{version}")
    if procedencia["origen"] != "manifiesto.json" and not (entradas / "casos.jsonl").is_file():
        avisos.append(f"la corrida no guardó sus casos: se asume {_mostrar(casos)}")
    trazas = leer_jsonl(ruta_trazas)
    ids = [t["caso"] for t in trazas]
    if len(set(ids)) != len(ids):
        raise CorridaInvalida(f"{ruta_trazas}: casos repetidos")
    por_id = {c.id: c for c in cargar_casos(casos)}
    if faltan := [i for i in ids if i not in por_id]:
        raise CorridaInvalida(f"{casos}: faltan los casos {faltan[:5]} de las trazas")
    tiempos = _leer_json(_primera(carpeta, "salida/tiempos.json", "tiempos.json"))
    return {"carpeta": carpeta, "trazas": trazas, "ruta_trazas": ruta_trazas, "casos": por_id,
            "ruta_casos": casos, "esperadas": cargar_esperadas(esperadas), "ruta_esperadas": esperadas,
            "ruta_normas": normas, "parametros": cargar_parametros(normas), "tiempos": tiempos,
            "procedencia": procedencia, "avisos": avisos, "version": version,
            "resultados_previos": _primera(carpeta, "puntuacion/resultados.jsonl", "resultados.jsonl"),
            "resumen_previo": _primera(carpeta, "puntuacion/resumen.json", "resumen.json")}


# --- atributos y secciones rectoras --------------------------------------------------------------

def fila_del_caso(caso: Caso, esperada: dict) -> dict:
    """Fila con columnas del historial para reutilizar la lógica de #8/#10. La credencial elegida cuando hay dos
    sale de la verificación esperada: solo para diagnosticar, nunca para el agente."""
    fila = fila_triage(nota_de_triage(caso))
    puede_preguntar = caso.puede_responder or bool(caso.acompanante)
    declarada = caso.declarada if (puede_preguntar and not caso.credenciales) else None
    tipo = {"obra_social": "obra social", "prepaga": "prepaga"}
    requerida = (esperada.get("verificacion") or {}).get("requerida") or ["", ""]
    return {**fila,
            "art_declarada": f"{caso.art[0]} CUIL {caso.art[1]}" if caso.trabajo and caso.art else "",
            "cobertura_declarada": f"{declarada[1]} ({tipo[declarada[0]]}) {declarada[2]}" if declarada else "",
            "entidad_verificada": requerida[0], "numero_verificado": requerida[1]}


def _fuente_esperada(caso: Caso, esperada: dict) -> tuple[str, str, str] | None:
    """(tipo, entidad, número) de la cobertura que rige el caso según la verificación esperada."""
    if caso.trabajo:
        return ("art", *caso.art) if caso.art else None
    requerida = (esperada.get("verificacion") or {}).get("requerida")
    if not requerida:
        return None
    for tipo, entidad, numero in list(caso.credenciales) + ([caso.declarada] if caso.declarada else []):
        if (_norm(entidad), _norm(numero)) == (_norm(requerida[0]), _norm(requerida[1])):
            return tipo, entidad, numero
    return None


def secciones_por_control(caso: Caso, esperada: dict, parametros: dict) -> dict[str, list[str]]:
    """Secciones normativas que deciden cada control del caso.

    Base: `secciones_rectoras` del #8 (ADM-01 siempre; ADM-02 por edad; ADM-03 por pasaporte/cédula; la sección del
    circuito y TDE-04 por ambulancia; COB-02 por ART; COB-04 por terminación reservada), repartida por familia. El
    #8 no lista sección de cobertura en el caso común; acá se agrega la que decide: COB-02 (verificar y notificar)
    si hay cobertura que verificar, COB-01 si no presentó credencial y COB-03 si el resultado es sin cobertura.
    """
    rectoras = secciones_rectoras(fila_del_caso(caso, esperada), parametros)
    adm = [s for s in rectoras if s.startswith("NOR-ADM")]
    tde = [s for s in rectoras if s.startswith("NOR-TDE")]
    cob = [s for s in rectoras if s.startswith("NOR-COB")]
    if not cob:
        fuente = _fuente_esperada(caso, esperada)
        if not caso.credenciales:
            cob.append("NOR-COB-01")
        activa = fuente is not None and caso.afiliaciones.get(f"{fuente[1]}|{fuente[2]}", False)
        if fuente is not None:
            cob.append("NOR-COB-02")
        if not activa:
            cob.append("NOR-COB-03")
    return {
        "registro": [s for s in adm if s in ("NOR-ADM-01", "NOR-ADM-03")],
        "numero_documento": ["NOR-ADM-01"],
        "box_antes_del_registro": [s for s in adm if s in ("NOR-ADM-01", "NOR-ADM-02")],
        "circuito": [s for s in tde if s != "NOR-TDE-04"],
        "plazo_circuito_min": tde,
        "cobertura": cob, "entidad_a_notificar": cob, "verificacion": cob,
        "aviso_al_voluntariado": [],
    }


def atributos_del_caso(caso: Caso, parametros: dict) -> dict[str, str]:
    """Tipo de caso para agrupar, desde el archivo de casos que usó la corrida."""
    fila = fila_triage(nota_de_triage(caso))
    if not caso.credenciales:
        credenciales = "ninguna"
    elif len(caso.credenciales) == 2:
        credenciales = "dos"
    else:
        credenciales = caso.credenciales[0][0]
    reservada = any(n.endswith(str(parametros["P-COB-2"])) for _, _, n in caso.credenciales)
    return {
        "circuito": circuito_esperado(fila),
        "llegada": caso.llegada,
        "credenciales": credenciales,
        "accidente_trabajo": caso.trabajo or "no",
        "edad_mayor_P-ADM-2": "sí" if caso.edad >= parametros["P-ADM-2"] else "no",
        "acompanante": "sí" if caso.acompanante else "no",
        "puede_responder": "sí" if caso.puede_responder else "no",
        "documento": caso.documento_tipo or "ninguno",
        "nivel": str(caso.nivel),
        "terminacion_reservada": "sí" if reservada else "no",
    }


# --- fuentes -------------------------------------------------------------------------------------

def formato_del_caso(traza: dict) -> dict:
    """Incidentes de formato del caso y, si no hay ficha, por qué terminó sin ella."""
    mensajes, log = traza.get("mensajes") or [], traza.get("log") or []
    respuestas = [m for m in mensajes if isinstance(m, dict) and m.get("role") == "assistant"]
    sin_json = sum(extraer_json(str(m.get("content", ""))) is None for m in respuestas)
    invalidas = Counter(e["herramienta"] for e in log if not e.get("valida", True))
    causa = None
    if traza.get("decision") is None:
        ultimo = mensajes[-1] if mensajes else None
        if not mensajes:
            causa = "sin_conversacion"
        elif len(log) >= MAX_LLAMADAS:
            causa = "tope_llamadas"
        elif ultimo.get("role") == "user" and str(ultimo.get("content", "")).startswith(PREFIJO_GUARDA):
            causa = "guarda_contexto"
        elif ultimo.get("role") == "assistant" and extraer_json(str(ultimo.get("content", ""))) is None:
            causa = "sin_json"
        elif invalidas.get("registrar_decision"):
            causa = "registrar_decision_invalida"
        else:
            causa = "tope_turnos"
    return {"respuestas_sin_json": sin_json, "llamadas_invalidas": dict(invalidas), "sin_decision": causa}


def _busquedas(log: list[dict]) -> list[dict]:
    return [{"consulta": str((e.get("argumentos") or {}).get("consulta", "")),
             "secciones": secciones_en(str(e.get("observacion", "")))}
            for e in log if e["herramienta"] == "buscar_normativa" and e.get("valida", True)]


def _relevante(busqueda: dict, familia: str) -> bool:
    return (any(s.startswith(f"NOR-{familia}") for s in busqueda["secciones"])
            or re.search(PALABRAS[familia], _plano(busqueda["consulta"])) is not None)


def _verifico_otra_credencial(caso: Caso, esperada: dict, log: list[dict]) -> bool:
    """Con dos credenciales, el agente verificó una que no es la que elige la costumbre."""
    requerida = (esperada.get("verificacion") or {}).get("requerida")
    if caso.trabajo or len(caso.credenciales) != 2 or not requerida:
        return False
    objetivo = (_norm(requerida[0]), _norm(requerida[1]))
    otras = {(_norm(e), _norm(n)) for _, e, n in caso.credenciales} - {objetivo}
    hechas = {(_norm((e["argumentos"] or {}).get("entidad", "")), _norm((e["argumentos"] or {}).get("numero_afiliado", "")))
              for e in log if e["herramienta"] == "verificar_afiliacion" and isinstance(e.get("argumentos"), dict)}
    return bool(hechas & otras)


def clasificar_falla(control: str, *, traza: dict, caso: Caso, esperada: dict, rectoras: list[str],
                     formato: dict) -> dict:
    """Fuente de un control fallado, con precedencia formato > costumbre > buscador/modelo."""
    log = traza.get("log") or []
    familia = FAMILIA[control]
    if formato["sin_decision"]:
        return {"fuente": "formato", "subtipo": formato["sin_decision"]}
    for herramienta in HERRAMIENTAS_FAMILIA.get(familia, ()):
        usos = [e for e in log if e["herramienta"] == herramienta]
        if usos and not any(e.get("valida", True) for e in usos):
            return {"fuente": "formato", "subtipo": f"argumentos_invalidos:{herramienta}"}
    if familia is None:
        return {"fuente": "costumbre", "subtipo": "aviso_al_voluntariado"}
    if familia == "COB" and _verifico_otra_credencial(caso, esperada, log):
        return {"fuente": "costumbre", "subtipo": "eleccion_de_credencial"}
    busquedas = _busquedas(log)
    vistas = {s for b in busquedas for s in b["secciones"]}
    faltantes = [s for s in rectoras if s not in vistas]
    relevantes = sum(_relevante(b, familia) for b in busquedas)
    detalle = {"rectoras": rectoras, "faltantes": faltantes, "busquedas_relevantes": relevantes}
    if faltantes and relevantes:
        return {"fuente": "buscador", "subtipo": "rectora_no_recuperada", **detalle}
    return {"fuente": "modelo", "subtipo": "busco_y_decidio_mal" if relevantes else "no_busco", **detalle}


# --- análisis ------------------------------------------------------------------------------------

def _resultados_coinciden(propios: list[dict], ruta: Path | None) -> dict | None:
    if ruta is None:
        return None
    previos = {r["caso"]: r for r in leer_jsonl(ruta)}
    distintos = [r["caso"] for r in propios
                 if r["caso"] not in previos
                 or (previos[r["caso"]]["resuelto"], previos[r["caso"]]["controles"]) != (r["resuelto"], r["controles"])]
    return {"archivo": _mostrar(ruta), "casos_distintos": distintos, "coincide": not distintos}


def _resumen_coincide(propio: dict, ruta: Path | None) -> dict | None:
    if ruta is None:
        return None
    previo = json.loads(ruta.read_text(encoding="utf-8"))
    claves = ("casos", "resueltos", "parcial_medio", "por_control", "sin_decision")
    distintas = [k for k in claves if previo.get(k) != propio.get(k)]
    return {"archivo": _mostrar(ruta), "claves_distintas": distintas, "coincide": not distintas}


def _p_signos(ganados: int, perdidos: int) -> float | None:
    """Prueba de signos exacta, dos colas."""
    n = ganados + perdidos
    if not n:
        return None
    cola = sum(comb(n, i) for i in range(min(ganados, perdidos) + 1)) / 2 ** n
    return round(min(1.0, 2 * cola), 4)


def analizar(corrida: dict) -> dict:
    """Análisis completo de una corrida cargada con `cargar_corrida`."""
    trazas, casos, esperadas, parametros = (corrida[k] for k in ("trazas", "casos", "esperadas", "parametros"))
    resultados, resumen = puntuar(trazas, esperadas, None, corrida["tiempos"])  # La función del kit.
    por_caso = []
    for traza, resultado in zip(trazas, resultados, strict=True):
        caso, esperada = casos[traza["caso"]], esperadas[traza["caso"]]
        rectoras = secciones_por_control(caso, esperada, parametros)
        formato = formato_del_caso(traza)
        fallas = {}
        for control in CONTROLES:
            if not resultado["controles"][control]:
                fallas[control] = clasificar_falla(control, traza=traza, caso=caso, esperada=esperada,
                                                   rectoras=rectoras[control], formato=formato)
                if control != "verificacion":
                    fallas[control]["esperado"] = resultado["esperada"].get(control)
                    fallas[control]["obtenido"] = (resultado["obtenida"] or {}).get(control)
        por_caso.append({
            "caso": traza["caso"], "resuelto": resultado["resuelto"], "parcial": resultado["parcial"],
            "controles": resultado["controles"], "atributos": atributos_del_caso(caso, parametros),
            "fallas": fallas, "formato": formato,
            "secciones_recuperadas": sorted({s for b in _busquedas(traza.get("log") or []) for s in b["secciones"]}),
            "busquedas": resultado["busquedas"], "llamadas": resultado["llamadas"],
            "caracteres_de_contexto": resultado["caracteres_de_contexto"], "segundos": resultado["segundos"]})
    n = len(por_caso)
    por_control = {}
    for control in CONTROLES:
        fuentes = Counter(c["fallas"][control]["fuente"] for c in por_caso if control in c["fallas"])
        subtipos = Counter(f"{c['fallas'][control]['fuente']}/{c['fallas'][control]['subtipo']}"
                           for c in por_caso if control in c["fallas"])
        fallas = sum(fuentes.values())
        por_control[control] = {"fallas": fallas, "tasa_falla": round(fallas / n, 3),
                                "acierto": round((n - fallas) / n, 3),
                                "fuentes": {f: fuentes.get(f, 0) for f in FUENTES},
                                "subtipos": dict(sorted(subtipos.items()))}
    grupos = {}
    for atributo in por_caso[0]["atributos"] if por_caso else ():
        valores = {}
        for c in por_caso:
            valores.setdefault(c["atributos"][atributo], []).append(c)
        grupos[atributo] = {
            valor: {"casos": len(lista), "resueltos": sum(c["resuelto"] for c in lista),
                    "controles_fallados": sum(len(c["fallas"]) for c in lista),
                    "fallas_por_caso": round(sum(len(c["fallas"]) for c in lista) / len(lista), 2),
                    "por_control": {k: sum(k in c["fallas"] for c in lista) for k in CONTROLES
                                    if any(k in c["fallas"] for c in lista)},
                    "ids": [c["caso"] for c in lista]}
            for valor, lista in sorted(valores.items())}
    peores = sorted(((a, v, g) for a, vs in grupos.items() for v, g in vs.items() if g["casos"] >= MINIMO_GRUPO),
                    key=lambda x: (-(1 - x[2]["resueltos"] / x[2]["casos"]), -x[2]["fallas_por_caso"], x[0], x[1]))
    fuentes_totales = Counter(f["fuente"] for c in por_caso for f in c["fallas"].values())
    faltantes = Counter(s for c in por_caso for f in c["fallas"].values() if f["fuente"] == "buscador"
                        for s in f["faltantes"])
    por_control_kit = resumen["por_control"]
    cruce = {"por_control_igual_al_kit": all(
        por_control[k]["acierto"] == por_control_kit[k] for k in CONTROLES),
        "resultados_previos": _resultados_coinciden(resultados, corrida["resultados_previos"]),
        "resumen_previo": _resumen_coincide(resumen, corrida["resumen_previo"])}
    tiempos = corrida["tiempos"] or {}
    return {
        "procedencia": {**corrida["procedencia"], "carpeta": _mostrar(corrida["carpeta"]),
                        "version": corrida["version"],
                        "trazas": _mostrar(corrida["ruta_trazas"]), "trazas_sha256": _sha256(corrida["ruta_trazas"]),
                        "casos": _mostrar(corrida["ruta_casos"]), "casos_sha256": _sha256(corrida["ruta_casos"]),
                        "esperadas": _mostrar(corrida["ruta_esperadas"]),
                        "esperadas_sha256": _sha256(corrida["ruta_esperadas"]),
                        "parametros": _mostrar(corrida["ruta_normas"] / "parametros.yaml"),
                        "parametros_sha256": _sha256(corrida["ruta_normas"] / "parametros.yaml")},
        "avisos": corrida["avisos"],
        "resumen_kit": resumen,
        "cruce": cruce,
        "por_control": por_control,
        "fuentes_totales": {f: fuentes_totales.get(f, 0) for f in FUENTES},
        "secciones_faltantes_buscador": dict(faltantes.most_common()),
        "formato": {"sin_decision": dict(Counter(c["formato"]["sin_decision"] for c in por_caso
                                                 if c["formato"]["sin_decision"])),
                    "respuestas_sin_json": sum(c["formato"]["respuestas_sin_json"] for c in por_caso),
                    "llamadas_invalidas": dict(sum((Counter(c["formato"]["llamadas_invalidas"]) for c in por_caso),
                                                   Counter())),
                    "casos_con_incidentes": [c["caso"] for c in por_caso if c["formato"]["respuestas_sin_json"]
                                             or c["formato"]["llamadas_invalidas"]]},
        "grupos": grupos,
        "tipos_mas_fallidos": [{"atributo": a, "valor": v, "casos": g["casos"], "resueltos": g["resueltos"],
                                "fallas_por_caso": g["fallas_por_caso"]} for a, v, g in peores[:8]],
        "costo": {"caracteres_de_contexto_por_caso": resumen["caracteres_de_contexto_por_caso"],
                  "busquedas_por_caso": resumen["busquedas_por_caso"], "llamadas_medio": resumen["llamadas_medio"],
                  "segundos_por_caso": tiempos.get("jugar_lote_por_caso")},
        "casos": por_caso,
    }


def comparar_corridas(antes: dict, despues: dict, *, trazas_antes: list[dict], trazas_despues: list[dict],
                      permitir_interseccion: bool = False) -> dict:
    """Ganados/perdidos por caso, controles que cambian, cambios de fuente y de costo entre dos análisis."""
    a = {c["caso"]: c for c in antes["casos"]}
    b = {c["caso"]: c for c in despues["casos"]}
    if set(a) != set(b) and not permitir_interseccion:
        raise ValueError(f"las corridas no tienen los mismos casos ({len(a)} y {len(b)}); "
                         "usá --permitir-interseccion para comparar solo los comunes")
    ids = [i for i in a if i in b]
    if not ids:
        raise ValueError("las corridas no tienen casos en común")
    ta, tb = {t["caso"]: t for t in trazas_antes}, {t["caso"]: t for t in trazas_despues}
    ganados = [i for i in ids if b[i]["resuelto"] and not a[i]["resuelto"]]
    perdidos = [i for i in ids if a[i]["resuelto"] and not b[i]["resuelto"]]
    cambios, por_control, transiciones = [], {}, Counter()
    for control in CONTROLES:
        arreglados = [i for i in ids if b[i]["controles"][control] and not a[i]["controles"][control]]
        rotos = [i for i in ids if a[i]["controles"][control] and not b[i]["controles"][control]]
        fa = Counter(a[i]["fallas"][control]["fuente"] for i in ids if control in a[i]["fallas"])
        fb = Counter(b[i]["fallas"][control]["fuente"] for i in ids if control in b[i]["fallas"])
        por_control[control] = {"arreglados": arreglados, "rotos": rotos,
                                "fuentes_antes": {f: fa.get(f, 0) for f in FUENTES},
                                "fuentes_despues": {f: fb.get(f, 0) for f in FUENTES}}
        for i in ids:
            fuente_a = a[i]["fallas"].get(control, {}).get("fuente", "ok")
            fuente_b = b[i]["fallas"].get(control, {}).get("fuente", "ok")
            if fuente_a != fuente_b:
                transiciones[f"{fuente_a}→{fuente_b}"] += 1
                cambios.append({"caso": i, "control": control, "antes": fuente_a, "despues": fuente_b})
    distintos = [i for i in ids
                 if (ta[i].get("mensajes"), ta[i].get("log"), ta[i].get("decision"))
                 != (tb[i].get("mensajes"), tb[i].get("log"), tb[i].get("decision"))]

    def costo(analisis, campo):
        valores = [analisis[i][campo] for i in ids if analisis[i][campo] is not None]
        return {"total": round(sum(valores), 3), "por_caso": round(sum(valores) / len(valores), 3) if valores else None}

    return {
        "antes": antes["procedencia"], "despues": despues["procedencia"],
        "casos_comparados": len(ids),
        "interseccion": set(a) != set(b),
        "solo_antes": sorted(set(a) - set(b)), "solo_despues": sorted(set(b) - set(a)),
        "resueltos": {"antes": sum(a[i]["resuelto"] for i in ids), "despues": sum(b[i]["resuelto"] for i in ids)},
        "ganados": ganados, "perdidos": perdidos, "p_signos": _p_signos(len(ganados), len(perdidos)),
        "por_control": por_control,
        "cambios_de_fuente": cambios, "transiciones_de_fuente": dict(transiciones.most_common()),
        "conversaciones_distintas": distintos,
        "diferencias": len(distintos) + len(cambios) + len(ganados) + len(perdidos),
        "costo": {campo: {"antes": costo(a, campo), "despues": costo(b, campo)}
                  for campo in ("caracteres_de_contexto", "busquedas", "llamadas", "segundos")},
        "segundos_lote_por_caso": {"antes": antes["costo"]["segundos_por_caso"],
                                   "despues": despues["costo"]["segundos_por_caso"]},
    }


# --- reportes ------------------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def _fila_procedencia(p: dict) -> str:
    prompt = p.get("prompt")
    prompt = f"{prompt['nombre']} v{prompt['version']} ({prompt['sha256'][:12]})" if prompt else "no registrado"
    return (f"commit `{p.get('commit')}` (sucio={p.get('sucio')}), origen {p.get('origen')}; modelo `{p.get('modelo')}`; "
            f"adapter {p.get('adapter')}; prompt {prompt}; buscador `{(p.get('buscador') or {}).get('variante')}`; "
            f"normas {p.get('version') or 'entradas/'}; trazas `{p['trazas']}` (sha256 `{p['trazas_sha256']}`)")


def markdown_analisis(an: dict, comando: list[str], generado: str) -> str:
    r, p = an["resumen_kit"], an["procedencia"]
    lineas = [f"# Análisis de errores: `{p['carpeta']}`", "",
              f"- Procedencia: {_fila_procedencia(p)}.",
              f"- Casos `{p['casos']}` (sha256 `{p['casos_sha256'][:12]}`), esperadas `{p['esperadas']}` "
              f"(sha256 `{p['esperadas_sha256'][:12]}`).",
              *[f"- Aviso: {a}" for a in an["avisos"]], "",
              f"**{r['resueltos']}/{r['casos']} resueltos**, parcial medio {r['parcial_medio']}, "
              f"sin decisión {r['sin_decision']}; {an['costo']['caracteres_de_contexto_por_caso']} caracteres de "
              f"contexto/caso, {an['costo']['busquedas_por_caso']} búsquedas/caso, "
              f"{an['costo']['segundos_por_caso']} s/caso (lote).", "",
              "## Fallas por control y fuente", "",
              "| Control | Fallas | Tasa | Formato | Costumbre | Buscador | Modelo: no buscó | Modelo: buscó y decidió mal |",
              "|---|---|---|---|---|---|---|---|"]
    for control, d in an["por_control"].items():
        s = d["subtipos"]
        lineas.append(f"| {control} | {d['fallas']} | {_pct(d['tasa_falla'])} | {d['fuentes']['formato']} | "
                      f"{d['fuentes']['costumbre']} | {d['fuentes']['buscador']} | {s.get('modelo/no_busco', 0)} | "
                      f"{s.get('modelo/busco_y_decidio_mal', 0)} |")
    t = an["fuentes_totales"]
    lineas.append(f"| **total** | {sum(t.values())} | | {t['formato']} | {t['costumbre']} | {t['buscador']} | "
                  f"{sum(d['subtipos'].get('modelo/no_busco', 0) for d in an['por_control'].values())} | "
                  f"{sum(d['subtipos'].get('modelo/busco_y_decidio_mal', 0) for d in an['por_control'].values())} |")
    c = an["cruce"]
    lineas += ["", f"Cruce: tasas por control iguales a `resumen.por_control` del kit: "
               f"**{'sí' if c['por_control_igual_al_kit'] else 'NO'}**."]
    for clave in ("resultados_previos", "resumen_previo"):
        if c[clave]:
            lineas.append(f"Contra `{c[clave]['archivo']}`: {'coincide' if c[clave]['coincide'] else 'NO coincide'}.")
    lineas += ["", "## Secciones rectoras no recuperadas (fallas del buscador)", ""]
    lineas += [f"- {s}: {n}" for s, n in an["secciones_faltantes_buscador"].items()] or ["- Ninguna."]
    f = an["formato"]
    lineas += ["", "## Formato", "",
               f"- Casos sin decisión por causa: {f['sin_decision'] or 'ninguno'}.",
               f"- Respuestas sin JSON: {f['respuestas_sin_json']}; llamadas inválidas: "
               f"{f['llamadas_invalidas'] or 'ninguna'} (casos {', '.join(f['casos_con_incidentes']) or '—'})."]
    lineas += ["", "## Tipos de caso con más fallas", "",
               f"Grupos de al menos {MINIMO_GRUPO} casos, ordenados por casos sin resolver.", "",
               "| Atributo | Valor | Casos | Resueltos | Controles fallados/caso |", "|---|---|---|---|---|"]
    lineas += [f"| {x['atributo']} | {x['valor']} | {x['casos']} | {x['resueltos']} | {x['fallas_por_caso']} |"
               for x in an["tipos_mas_fallidos"]]
    lineas += ["", "## Fallas por tipo de caso", "", "| Atributo | Valor | Casos | Resueltos | Fallas por control |",
               "|---|---|---|---|---|"]
    for atributo, valores in an["grupos"].items():
        for valor, g in valores.items():
            detalle = ", ".join(f"{k} {v}" for k, v in g["por_control"].items()) or "—"
            lineas.append(f"| {atributo} | {valor} | {g['casos']} | {g['resueltos']} | {detalle} |")
    lineas += ["", f"Generado {generado} con `{' '.join(comando)}`. Reglas de fuente: docs/informe/analisis-errores.md. "
               "Analizar práctica es diagnóstico y reporte, no selección."]
    return "\n".join(lineas) + "\n"


def markdown_comparacion(cmp: dict, comando: list[str], generado: str) -> str:
    lineas = ["# Comparación de corridas", "",
              f"- Antes: `{cmp['antes']['carpeta']}`: {_fila_procedencia(cmp['antes'])}.",
              f"- Después: `{cmp['despues']['carpeta']}`: {_fila_procedencia(cmp['despues'])}.",
              f"- Casos comparados: {cmp['casos_comparados']}"
              + (f" (**solo la intersección**; solo antes {cmp['solo_antes']}, solo después {cmp['solo_despues']})"
                 if cmp["interseccion"] else " (mismo conjunto)") + ".", "",
              f"**Resueltos {cmp['resueltos']['antes']} → {cmp['resueltos']['despues']}.** Ganados "
              f"{len(cmp['ganados'])} ({', '.join(cmp['ganados']) or '—'}); perdidos {len(cmp['perdidos'])} "
              f"({', '.join(cmp['perdidos']) or '—'}); prueba de signos p={cmp['p_signos']}.",
              f"Conversaciones distintas: {len(cmp['conversaciones_distintas'])}; diferencias totales "
              f"(casos, controles, fuentes, conversaciones): {cmp['diferencias']}.", "",
              "## Por control", "",
              "| Control | Arreglados | Rotos | Fuentes antes (F/C/B/M) | Fuentes después (F/C/B/M) |",
              "|---|---|---|---|---|"]
    for control, d in cmp["por_control"].items():
        fa, fb = d["fuentes_antes"], d["fuentes_despues"]
        lineas.append(f"| {control} | {len(d['arreglados'])} | {len(d['rotos'])} | "
                      f"{'/'.join(str(fa[f]) for f in FUENTES)} | {'/'.join(str(fb[f]) for f in FUENTES)} |")
    lineas += ["", "## Cambios de fuente", ""]
    lineas += [f"- {k}: {v}" for k, v in cmp["transiciones_de_fuente"].items()] or ["- Ninguno."]
    lineas += ["", "## Costo", "", "| Medida | Antes (por caso) | Después (por caso) |", "|---|---|---|"]
    for campo, d in cmp["costo"].items():
        lineas.append(f"| {campo} | {d['antes']['por_caso']} | {d['despues']['por_caso']} |")
    s = cmp["segundos_lote_por_caso"]
    lineas += [f"| segundos de lote/caso (tiempos.json) | {s['antes']} | {s['despues']} |", "",
               "`segundos` por caso es el momento de la última llamada desde el inicio del lote: con lotes, los "
               "casos corren a la vez; T4 no es comparable con A100.", "",
               f"Generado {generado} con `{' '.join(comando)}`."]
    return "\n".join(lineas) + "\n"


# --- CLI -----------------------------------------------------------------------------------------

def _crear_salida(salida: Path) -> None:
    if salida.exists():
        raise FileExistsError(f"{salida} ya existe: elegí una salida nueva para no pisar un análisis")
    salida.mkdir(parents=True)


def _meta(comando: list[str]) -> dict:
    return {"generado": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "comando": comando,
            "git": {"commit": _git("rev-parse", "HEAD"), "sucio": bool(_git("status", "--porcelain"))}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    a = sub.add_parser("analizar", help="analiza una corrida")
    a.add_argument("--corrida", type=Path, required=True)
    for opcion in ("version", "casos", "esperadas", "normas"):
        a.add_argument(f"--{opcion}", type=None if opcion == "version" else Path)
    c = sub.add_parser("comparar", help="compara dos corridas sobre los mismos casos")
    c.add_argument("--antes", type=Path, required=True)
    c.add_argument("--despues", type=Path, required=True)
    c.add_argument("--version-antes")
    c.add_argument("--version-despues")
    c.add_argument("--permitir-interseccion", action="store_true")
    for p in (a, c):
        p.add_argument("--salida", type=Path, required=True, help="carpeta nueva")
    args = parser.parse_args(argv)
    comando = ["python", "-m", "scripts.analizar_errores", *(argv if argv is not None else sys.argv[1:])]
    if args.salida.exists():
        parser.error(f"{args.salida} ya existe: elegí una salida nueva para no pisar un análisis")
    if args.modo == "analizar":
        corrida = cargar_corrida(args.corrida, version=args.version, casos=args.casos,
                                 esperadas=args.esperadas, normas=args.normas)
        an = analizar(corrida)
        meta = _meta(comando)
        _crear_salida(args.salida)
        (args.salida / "analisis.json").write_text(
            json.dumps({**meta, **an}, ensure_ascii=False, indent=2), encoding="utf-8")
        texto = markdown_analisis(an, comando, meta["generado"])
        (args.salida / "analisis.md").write_text(texto, encoding="utf-8")
    else:
        ca = cargar_corrida(args.antes, version=args.version_antes)
        cb = cargar_corrida(args.despues, version=args.version_despues)
        cmp = comparar_corridas(analizar(ca), analizar(cb), trazas_antes=ca["trazas"], trazas_despues=cb["trazas"],
                                permitir_interseccion=args.permitir_interseccion)
        meta = _meta(comando)
        _crear_salida(args.salida)
        (args.salida / "comparacion.json").write_text(
            json.dumps({**meta, **cmp}, ensure_ascii=False, indent=2), encoding="utf-8")
        texto = markdown_comparacion(cmp, comando, meta["generado"])
        (args.salida / "comparacion.md").write_text(texto, encoding="utf-8")
    print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
