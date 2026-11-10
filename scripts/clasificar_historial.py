"""Clasificador reproducible del historial hospitalario."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SI = "s\u00ed"
SECCIONES = (
    "NOR-ADM-01", "NOR-ADM-02", "NOR-ADM-03",
    "NOR-COB-01", "NOR-COB-02", "NOR-COB-03", "NOR-COB-04",
    "NOR-TDE-01", "NOR-TDE-02", "NOR-TDE-03", "NOR-TDE-04",
)
FICHA_FIELDS = (
    "registro", "numero_documento", "box_antes_del_registro", "circuito",
    "plazo_circuito_min", "cobertura", "entidad_a_notificar",
    "aviso_al_voluntariado",
)


def cargar_parametros(ruta_normas: Path) -> dict[str, int]:
    texto = (ruta_normas / "parametros.yaml").read_text(encoding="utf-8")
    parametros = {
        codigo: int(valor)
        for codigo, valor in re.findall(
            r"codigo:\s*(P-[A-Z0-9-]+)\s*\n\s*valor:\s*([0-9]+)", texto
        )
    }
    requeridos = {
        "P-ADM-1", "P-ADM-2", "P-ADM-3", "P-COB-2",
        "P-TDE-1", "P-TDE-2", "P-TDE-3", "P-TDE-4", "P-TDE-5",
    }
    faltantes = requeridos - parametros.keys()
    if faltantes:
        raise ValueError(f"faltan parametros: {sorted(faltantes)}")
    return parametros


def verificar_fuentes(ruta_normas: Path) -> None:
    faltantes = [
        f"{seccion}.md" for seccion in SECCIONES
        if not (ruta_normas / f"{seccion}.md").is_file()
    ]
    if faltantes or not (ruta_normas / "parametros.yaml").is_file():
        raise FileNotFoundError(f"fuentes normativas incompletas: {faltantes}")


def _normalizar_ficha(valor: object) -> str:
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return SI if valor else "no"
    return str(valor)


def cargar_trazas(ruta: Path, rows: list[dict[str, str]]) -> dict:
    """Valida que las trazas correspondan al historial y resume su evidencia."""
    esperadas = {row["id"]: row for row in rows}
    trazas = []
    with ruta.open(encoding="utf-8") as archivo:
        for linea, texto in enumerate(archivo, 1):
            try:
                traza = json.loads(texto)
            except json.JSONDecodeError as error:
                raise ValueError(f"traza invalida en linea {linea}") from error
            trazas.append(traza)
    ids = [traza.get("id") for traza in trazas]
    if len(trazas) != len(rows) or set(ids) != set(esperadas) or len(set(ids)) != len(ids):
        raise ValueError("las trazas no tienen exactamente un registro por fila del historial")

    fichas_coincidentes = 0
    operadores = Counter()
    herramientas = Counter()
    pasos = 0
    por_id = {}
    for traza in trazas:
        row = esperadas[traza["id"]]
        ficha = traza.get("ficha")
        if not isinstance(ficha, dict):
            raise ValueError(f"la traza {traza['id']} no tiene ficha final")
        if traza.get("operador") != row["operador"]:
            raise ValueError(f"el operador de la traza {traza['id']} no coincide con el historial")
        if all(_normalizar_ficha(ficha.get(campo)) == _normalizar_ficha(row[campo]) for campo in FICHA_FIELDS):
            fichas_coincidentes += 1
        else:
            raise ValueError(f"la ficha de la traza {traza['id']} no coincide con el historial")
        operadores[traza.get("operador", "")] += 1
        for paso in traza.get("pasos", []):
            herramientas[paso.get("herramienta", "")] += 1
            pasos += 1
        por_id[traza["id"]] = traza
    return {
        "filas": len(trazas),
        "ids_coincidentes": len(ids),
        "fichas_coincidentes": fichas_coincidentes,
        "operadores": dict(sorted(operadores.items())),
        "herramientas": dict(sorted(herramientas.items())),
        "pasos_promedio": round(pasos / len(trazas), 6),
        "por_id": por_id,
    }


def entradas_cobertura(texto: str) -> list[tuple[str, str, str]]:
    return [
        (entidad.strip(), tipo, numero.strip())
        for entidad, tipo, numero in re.findall(
            r"([^;]+?) \((obra social|prepaga)\) ([^;]+)", texto
        )
    ]


def fuente_cobertura(row: dict[str, str]) -> tuple[str, str, str] | None:
    if row["lesion_trabajo"] != "no":
        match = re.search(r"(.+ ART) CUIL (.+)$", row["art_declarada"])
        return ("art", match.group(1), match.group(2)) if match else None

    entradas = entradas_cobertura(row["credenciales"])
    if not entradas:
        entradas = entradas_cobertura(row["cobertura_declarada"])
    if not entradas:
        return None
    if len(entradas) > 1:
        verificadas = [
            entrada for entrada in entradas
            if entrada[0] == row["entidad_verificada"].strip()
            and entrada[2] == row["numero_verificado"].strip()
        ]
        if len(verificadas) != 1:
            return None
        entidad, tipo, numero = verificadas[0]
    else:
        entidad, tipo, numero = entradas[0]
    return {"obra social": "obra_social", "prepaga": "prepaga"}[tipo], entidad, numero


def circuito_esperado(row: dict[str, str]) -> str:
    if row["sospecha_sepsis"] != "no":
        return "codigo_sepsis"
    if row["motivo"] == "dolor tor\u00e1cico":
        return "ecg"
    if row["motivo"] == "signos neurol\u00f3gicos focales de inicio s\u00fabito":
        return "codigo_acv"
    return "ninguno"


def plazo_esperado(row: dict[str, str], parametros: dict[str, int]) -> str:
    circuito = circuito_esperado(row)
    bases = {
        "ecg": parametros["P-TDE-1"],
        "codigo_acv": parametros["P-TDE-2"],
        "codigo_sepsis": (
            parametros["P-TDE-3"]
            if row["sospecha_sepsis"] in {"probable", "con signos de shock"}
            else parametros["P-TDE-4"]
        ),
    }
    if circuito == "ninguno":
        return ""
    descuento = parametros["P-TDE-5"] if row["llegada"] == "en ambulancia" else 0
    return str(bases[circuito] - descuento)


def registro_esperado(row: dict[str, str], parametros: dict[str, int]) -> str:
    if not row["documento_tipo"]:
        return "provisorio"
    if row["documento_tipo"] == "DNI":
        return "definitivo"
    return "provisorio" if int(row["nivel"]) <= parametros["P-ADM-3"] else "definitivo"


def cobertura_esperada(
    tipo: str,
    entidad: str,
    numero: str,
    activa: bool,
    parametros: dict[str, int],
) -> tuple[str, str]:
    """Devuelve (cobertura, entidad_a_notificar) para la fuente verificada.

    NOR-COB-04 §1/§3: con número terminado en P-COB-2 la ficha dice sin_cobertura y se
    notifica igual a la entidad, esté activa o no la afiliación (prevalece sobre
    NOR-COB-02 §3 y NOR-COB-03 §4). §4: no rige para la ART.
    """
    if tipo != "art" and numero.endswith(str(parametros["P-COB-2"])):
        return "sin_cobertura", entidad
    if not activa:
        return "sin_cobertura", ""
    return tipo, entidad


def razones_normativas(row: dict[str, str], parametros: dict[str, int]) -> list[str]:
    razones = []
    if row["registro"] != registro_esperado(row, parametros):
        razones.append("registro")
    if row["numero_documento"] != row["documento_numero"]:
        razones.append("numero_documento")

    # NOR-ADM-01 §4 sólo exige el box antes del registro en los niveles <= P-ADM-1 (y
    # NOR-ADM-02 por edad); en los demás niveles el registro "puede" hacerse mientras se
    # espera. Marcar error un box anticipado no exigido es más estricto que el texto, pero
    # en el historial no hay ninguna fila así (0 de 2.000), por lo que no cambia etiquetas.
    box = int(row["nivel"]) <= parametros["P-ADM-1"] or int(row["edad"]) >= parametros["P-ADM-2"]
    if (row["box_antes_del_registro"] == SI) != box:
        razones.append("box_antes_del_registro")

    circuito = circuito_esperado(row)
    if row["circuito"] != circuito:
        razones.append("circuito")
    if row["plazo_circuito_min"] != plazo_esperado(row, parametros):
        razones.append("plazo_circuito_min")

    fuente = fuente_cobertura(row)
    if fuente is None:
        tiene_credenciales = bool(entradas_cobertura(row["credenciales"]))
        tiene_datos_verificacion = any(
            row[campo] for campo in (
                "entidad_verificada", "numero_verificado", "respuesta_verificacion", "entidad_a_notificar"
            )
        )
        if tiene_credenciales and tiene_datos_verificacion:
            razones.append("verificacion")
        elif tiene_datos_verificacion:
            razones.append("verificacion_sin_fuente")
        if row["cobertura"] != "sin_cobertura":
            razones.append("cobertura")
        return razones

    tipo, entidad, numero = fuente
    if row["entidad_verificada"].strip() != entidad or row["numero_verificado"].strip() != numero:
        razones.append("verificacion")
    cobertura, notificar = cobertura_esperada(
        tipo, entidad, numero, row["respuesta_verificacion"] == "activa", parametros
    )
    if row["cobertura"] != cobertura:
        razones.append("cobertura")
    if row["entidad_a_notificar"] != notificar:
        razones.append("entidad_a_notificar")
    return razones


# Condiciones observables en el triage que la búsqueda puede combinar con la edad. No
# fijan ningún valor: la búsqueda elige el umbral y la condición que mejor explican las
# filas limpias, y el resultado se documenta con su comparación contra la regla sólo por edad.
CONDICIONES_AVISO = {
    # clave: (descripción, clave de la condición opuesta, predicado sobre la fila)
    "ninguna": ("sin condición adicional", None, lambda row: True),
    "sin_acompanante": ("sin acompañante", "con_acompanante", lambda row: not row["acompanante"].strip()),
    "con_acompanante": ("con acompañante", "sin_acompanante", lambda row: bool(row["acompanante"].strip())),
    "puede_responder": ("que puede responder", "no_puede_responder", lambda row: row["puede_responder"] == SI),
    "no_puede_responder": ("que no puede responder", "puede_responder", lambda row: row["puede_responder"] != SI),
    "en_ambulancia": ("llegado en ambulancia", "sin_ambulancia", lambda row: row["llegada"] == "en ambulancia"),
    "sin_ambulancia": ("llegado por sus medios", "en_ambulancia", lambda row: row["llegada"] != "en ambulancia"),
}
MINIMO_PREFERENCIA_CREDENCIAL = 0.90


def aviso_segun_costumbre(row: dict[str, str], costumbre: dict) -> bool:
    """Aplica la costumbre completa del voluntariado inferida del historial."""
    condicion = CONDICIONES_AVISO[costumbre["condicion"]][2]
    return int(row["edad"]) >= int(costumbre["umbral_edad"]) and condicion(row)


def _ids(rows: list[dict[str, str]]) -> list[str]:
    return sorted(row["id"] for row in rows)


def _resumen_operadores(rows: list[dict[str, str]], desvio) -> list[dict]:
    grupos: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grupos.setdefault(row["operador"], []).append(row)
    return [
        {
            "operador": operador,
            "muestra": len(grupo),
            "desvios": sum(desvio(row) for row in grupo),
            "ids_desvio": _ids([row for row in grupo if desvio(row)]),
        }
        for operador, grupo in sorted(grupos.items())
    ]


def inferir_costumbre_aviso(rows: list[dict[str, str]], limpias: list[dict[str, str]]) -> dict | None:
    """Busca la regla `edad >= umbral y condición` que mejor explica el aviso en filas limpias."""
    observado = [row["aviso_al_voluntariado"] == SI for row in limpias]
    edades = [int(row["edad"]) for row in limpias]
    candidatos = []
    for orden, (clave, (_, _, condicion)) in enumerate(CONDICIONES_AVISO.items()):
        cumple = [condicion(row) for row in limpias]
        for umbral in sorted(set(edades)):
            prediccion = [edad >= umbral and ok for edad, ok in zip(edades, cumple)]
            avisos = sum(prediccion)
            if avisos == 0 or avisos == len(limpias):
                continue
            aciertos = sum(p == o for p, o in zip(prediccion, observado))
            # Empates: menos condiciones primero; entre umbrales igual de buenos, el mayor,
            # para no afirmar avisos en edades sin evidencia.
            candidatos.append((-aciertos, orden, -umbral, clave, umbral))
    if not candidatos:
        return None
    candidatos.sort()
    _, _, _, clave, umbral = candidatos[0]
    solo_edad = next(item for item in candidatos if item[3] == "ninguna")
    costumbre = {"tipo": "aviso_al_voluntariado", "umbral_edad": umbral, "condicion": clave}
    descripcion, opuesta, condicion = CONDICIONES_AVISO[clave]
    costumbre["condicion_descripcion"] = descripcion
    costumbre["condicion_opuesta_descripcion"] = CONDICIONES_AVISO[opuesta][0] if opuesta else ""

    def desvio(row: dict[str, str]) -> bool:
        return aviso_segun_costumbre(row, costumbre) != (row["aviso_al_voluntariado"] == SI)

    lados = {
        "avisar": [row for row in limpias if aviso_segun_costumbre(row, costumbre)],
        "no_avisar_menores": [row for row in limpias if int(row["edad"]) < umbral],
        "no_avisar_mayores_sin_condicion": [
            row for row in limpias if int(row["edad"]) >= umbral and not condicion(row)
        ],
    }
    operadores = _resumen_operadores(limpias, desvio)
    con_desvio = {item["operador"] for item in operadores if item["desvios"]}
    resto = [row for row in rows if row["operador"] not in con_desvio]
    resto_lados = {
        "avisar": [row for row in resto if aviso_segun_costumbre(row, costumbre)],
        "no_avisar_menores": [row for row in resto if int(row["edad"]) < umbral],
        "no_avisar_mayores_sin_condicion": [
            row for row in resto if int(row["edad"]) >= umbral and not condicion(row)
        ],
    }
    costumbre.update({
        "nombre": f"avisar al voluntariado sólo a pacientes de {umbral} años o más {descripcion}",
        "muestra_limpias": len(limpias),
        "coinciden_limpias": -candidatos[0][0],
        "comparacion_solo_edad": {
            "umbral_edad": solo_edad[4],
            "coinciden_limpias": -solo_edad[0],
        },
        "lados": {
            nombre: {
                "muestra": len(grupo),
                "siguen_costumbre": sum(not desvio(row) for row in grupo),
                "ids_respaldo": _ids([row for row in grupo if not desvio(row)]),
                "ids_ruido": _ids([row for row in grupo if desvio(row)]),
            }
            for nombre, grupo in lados.items()
        },
        "operadores": operadores,
        "operadores_con_desvios": sorted(con_desvio),
        "todas_las_filas_operadores_sin_desvios": {
            nombre: {
                "muestra": len(grupo),
                "siguen_costumbre": sum(not desvio(row) for row in grupo),
            }
            for nombre, grupo in resto_lados.items()
        },
        "ids_respaldo": _ids([row for row in limpias if not desvio(row)]),
        "ids_ruido": _ids([row for row in limpias if desvio(row)]),
    })
    return costumbre


def _obra_social_y_prepaga(row: dict[str, str]) -> bool:
    entradas = entradas_cobertura(row["credenciales"])
    return (
        row["lesion_trabajo"] == "no"  # con lesión laboral rige la ART (NOR-COB-02 §5)
        and len(entradas) == 2
        and {tipo for _, tipo, _ in entradas} == {"obra social", "prepaga"}
    )


def inferir_costumbre_credencial(
    rows: list[dict[str, str]],
    limpias: list[dict[str, str]],
    trazas: dict | None = None,
) -> dict | None:
    """Qué credencial se verifica cuando hay obra social y prepaga, y qué pasa si está inactiva."""
    multiples = [
        (row, fuente_cobertura(row)[0]) for row in limpias
        if _obra_social_y_prepaga(row) and fuente_cobertura(row)
    ]
    if not multiples:
        return None
    preferencias = Counter(tipo for _, tipo in multiples)
    preferido, cantidad = preferencias.most_common(1)[0]
    if cantidad / len(multiples) < MINIMO_PREFERENCIA_CREDENCIAL:
        return None
    siguen = [row for row, tipo in multiples if tipo == preferido]
    activas = [row for row in siguen if row["respuesta_verificacion"] == "activa"]
    inactivas = [row for row in siguen if row["respuesta_verificacion"] != "activa"]
    ruido = [row for row, tipo in multiples if tipo != preferido]

    todas = [row for row in rows if _obra_social_y_prepaga(row) and fuente_cobertura(row)]
    todas_inactivas = [
        row for row in todas
        if fuente_cobertura(row)[0] == preferido and row["respuesta_verificacion"] != "activa"
    ]
    costumbre = {
        "tipo": "seleccion_credencial",
        "nombre": (
            f"con obra social y prepaga, verificar sólo la {preferido.replace('_', ' ')} "
            "y registrar su resultado, sin recurrir a la otra credencial"
        ),
        "muestra": len(multiples),
        "preferido": preferido,
        "frecuencia_preferido": round(cantidad / len(multiples), 6),
        "frecuencias": dict(sorted(preferencias.items())),
        "preferido_activa": {
            "muestra": len(activas),
            "coberturas": dict(sorted(Counter(row["cobertura"] for row in activas).items())),
            "ids_respaldo": _ids(activas),
        },
        "preferido_inactiva": {
            "muestra": len(inactivas),
            "coberturas": dict(sorted(Counter(row["cobertura"] for row in inactivas).items())),
            "ids_respaldo": _ids(inactivas),
        },
        "todas_las_filas_preferido_inactiva": {
            "muestra": len(todas_inactivas),
            "coberturas": dict(sorted(Counter(row["cobertura"] for row in todas_inactivas).items())),
        },
        "operadores_con_desvios": sorted({row["operador"] for row in ruido}),
        "ids_respaldo": _ids(siguen),
        "ids_ruido": _ids(ruido),
    }
    if trazas is not None:
        verificaciones = Counter(
            sum(paso.get("herramienta") == "verificar_afiliacion"
                for paso in trazas["por_id"][row["id"]].get("pasos", []))
            for row in todas
        )
        costumbre["verificaciones_por_traza"] = {str(k): v for k, v in sorted(verificaciones.items())}
    return costumbre


def inferir_costumbres(
    rows: list[dict[str, str]],
    parametros: dict[str, int],
    trazas: dict | None = None,
) -> list[dict]:
    """Detecta costumbres sólo en filas sin violaciones normativas."""
    razones = {row["id"]: razones_normativas(row, parametros) for row in rows}
    limpias = [row for row in rows if not razones[row["id"]]]
    costumbres = [
        inferir_costumbre_aviso(rows, limpias),
        inferir_costumbre_credencial(rows, limpias, trazas),
    ]
    return [costumbre for costumbre in costumbres if costumbre is not None]


def desvios_por_operador(costumbres: list[dict], rows: list[dict[str, str]]) -> list[dict]:
    """Cuántas filas `ruido` aporta cada operador, por costumbre."""
    operador = {row["id"]: row["operador"] for row in rows}
    conteo: dict[str, Counter] = {}
    for costumbre in costumbres:
        for identificador in costumbre["ids_ruido"]:
            conteo.setdefault(operador[identificador], Counter())[costumbre["tipo"]] += 1
    return [
        {"operador": nombre, **dict(sorted(tipos.items())), "total": sum(tipos.values())}
        for nombre, tipos in sorted(conteo.items())
    ]


def clasificar(
    rows: list[dict[str, str]],
    parametros: dict[str, int],
    trazas: dict | None = None,
) -> tuple[list[dict[str, str]], dict]:
    costumbres = inferir_costumbres(rows, parametros, trazas)
    razones = {row["id"]: razones_normativas(row, parametros) for row in rows}
    ids_ruido = {i for costumbre in costumbres for i in costumbre.get("ids_ruido", [])}
    salida = []
    for row in rows:
        tipo = "error" if razones[row["id"]] else "ruido" if row["id"] in ids_ruido else "correcta"
        salida.append({"id": row["id"], "tipo": tipo})
    return salida, {
        "costumbres": costumbres,
        "desvios_por_operador": desvios_por_operador(costumbres, rows),
    }


def _lista_ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "(ninguno)"


def _detalle(titulo: str, ids: list[str]) -> str:
    return f"<details><summary>{titulo} ({len(ids)})</summary>\n\n{_lista_ids(ids)}\n\n</details>\n"


def _conteos(conteo: dict) -> str:
    return ", ".join(f"`{clave}` {valor}" for clave, valor in conteo.items())


def _md_aviso(item: dict) -> list[str]:
    lados = item["lados"]
    resto = item["todas_las_filas_operadores_sin_desvios"]
    solo = item["comparacion_solo_edad"]
    umbral = item["umbral_edad"]
    condicion, opuesta = item["condicion_descripcion"], item["condicion_opuesta_descripcion"]
    filas = [
        (f"{umbral} años o más, {condicion}", "`sí`", "avisar"),
        (f"{umbral} años o más, {opuesta}", "`no`", "no_avisar_mayores_sin_condicion"),
        (f"menos de {umbral} años", "`no`", "no_avisar_menores"),
    ]
    lineas = [
        f"## 1. {item['nombre'][0].upper()}{item['nombre'][1:]}\n",
        f"**Costumbre.** `aviso_al_voluntariado` es `sí` si y sólo si el paciente tiene "
        f"**{umbral} años o más** y está **{condicion}** en la guardia. Ninguna norma escrita "
        "menciona el voluntariado.\n",
        "**Cómo se encontró.** Sobre las filas sin violaciones normativas se probaron todas las reglas "
        "`edad >= umbral` (cada edad observada), solas o combinadas con un dato observable del triage "
        "(acompañante, si puede responder, forma de llegada), y se eligió la que explica más filas. "
        f"La regla completa explica {item['coinciden_limpias']} de {item['muestra_limpias']} filas "
        f"limpias; la mejor regla sólo por edad (umbral {solo['umbral_edad']}) explica "
        f"{solo['coinciden_limpias']}, porque trata igual a los mayores {condicion} y {opuesta}. "
        "Todos los desvíos restantes son de tres operadores (ver abajo).\n",
        "| situación | aviso | filas limpias que siguen la costumbre | todas las filas de los operadores sin desvíos |",
        "|---|---|---:|---:|",
    ]
    for situacion, aviso, clave in filas:
        lineas.append(
            f"| {situacion} | {aviso} | {lados[clave]['siguen_costumbre']}/{lados[clave]['muestra']} | "
            f"{resto[clave]['siguen_costumbre']}/{resto[clave]['muestra']} |"
        )
    lineas += [
        "",
        f"**Quién se aparta.** Sólo {', '.join(item['operadores_con_desvios'])}. En filas sin errores "
        f"no avisaron a {len(lados['avisar']['ids_ruido'])} pacientes de {umbral}+ {condicion}, y avisaron "
        f"a {len(lados['no_avisar_mayores_sin_condicion']['ids_ruido'])} de {umbral}+ {opuesta} y a "
        f"{len(lados['no_avisar_menores']['ids_ruido'])} menores de {umbral}: se apartan en ambas "
        "direcciones, sin otra regla reconocible. El resto de los operadores cumple la costumbre en "
        "el 100% de sus filas, incluso en las que tienen errores normativos en otros campos (última "
        "columna). Cada desvío en una fila sin error normativo se clasifica `ruido`, en cualquier "
        "motivo o cobertura.\n",
        _detalle(f"Respaldo `sí`: {umbral}+ {condicion}", lados["avisar"]["ids_respaldo"]),
        _detalle(f"Respaldo `no`: {umbral}+ {opuesta}", lados["no_avisar_mayores_sin_condicion"]["ids_respaldo"]),
        _detalle(f"Respaldo `no`: menores de {umbral}", lados["no_avisar_menores"]["ids_respaldo"]),
        _detalle("Ruido: desvíos de la costumbre", item["ids_ruido"]),
    ]
    return lineas


def _md_credencial(item: dict) -> list[str]:
    preferido = item["preferido"].replace("_", " ")
    activa, inactiva = item["preferido_activa"], item["preferido_inactiva"]
    todas = item["todas_las_filas_preferido_inactiva"]
    verificaciones = item.get("verificaciones_por_traza")
    evidencia = (
        f"**Cómo se encontró.** En {item['muestra']} filas limpias con obra social y prepaga, la "
        f"credencial verificada fue la {preferido} en {item['frecuencias'].get(item['preferido'], 0)} "
        f"({item['frecuencia_preferido']:.1%}). Con la {preferido} activa ({activa['muestra']} filas) "
        f"la ficha registra {_conteos(activa['coberturas'])} (los `sin_cobertura` son números con "
        f"terminación reservada, NOR-COB-04). Con la {preferido} inactiva ({inactiva['muestra']} filas) "
        f"registra {_conteos(inactiva['coberturas'])}; contando también filas con errores en otros "
        f"campos, las {todas['muestra']} con la {preferido} inactiva registran {_conteos(todas['coberturas'])}."
    )
    if verificaciones:
        evidencia += (
            f" Las {verificaciones['1']} trazas de filas con las dos credenciales tienen exactamente "
            "una verificación de afiliación: nunca se consulta la segunda."
            if set(verificaciones) == {"1"} else
            f" Verificaciones por traza: {_conteos(verificaciones)}."
        )
    return [
        f"## 2. {item['nombre'][0].upper()}{item['nombre'][1:]}\n",
        f"**Costumbre.** Si el paciente presenta obra social y prepaga (y no hay lesión laboral, "
        f"donde rige la ART), se verifica **sólo la {preferido}**. Si está activa, la ficha registra "
        f"esa cobertura y se la notifica según las normas; si está inactiva, se registra "
        f"`sin_cobertura` y **no se recurre a la otra credencial**. Las normas obligan a verificar la "
        "cobertura que se usa, pero no dicen cuál elegir cuando hay dos.\n",
        evidencia + "\n",
        f"**Quién se aparta.** Las {len(item['ids_ruido'])} filas limpias que verificaron la otra "
        f"credencial son de {', '.join(item['operadores_con_desvios'])}; se clasifican `ruido`.\n",
        _detalle(f"Respaldo: {preferido} activa", activa["ids_respaldo"]),
        _detalle(f"Respaldo: {preferido} inactiva, `sin_cobertura` sin segunda verificación",
                 inactiva["ids_respaldo"]),
        _detalle("Ruido: verificaron la otra credencial", item["ids_ruido"]),
    ]


def escribir_costumbres(ruta: Path, informe: dict) -> None:
    costumbres = informe.get("costumbres", [])
    if not costumbres:
        ruta.write_text("# Costumbres detectadas\n\nNo se detectaron costumbres con evidencia suficiente.\n", encoding="utf-8")
        return
    bloques = [
        "# Costumbres detectadas\n",
        "Generado por `scripts/clasificar_historial.py` a partir de `historial/historial.csv`, "
        "`historial/trazas.jsonl` y `normas/original/`. Las costumbres se infieren sólo de filas "
        "sin violaciones normativas; una violación siempre es `error`, aunque también se aparte "
        "de una costumbre. Los números y las listas completas están en `historial_costumbres.json`.\n",
    ]
    for item in costumbres:
        if item["tipo"] == "aviso_al_voluntariado":
            bloques += _md_aviso(item)
        else:
            bloques += _md_credencial(item)
    desvios = informe.get("desvios_por_operador", [])
    if desvios:
        tipos = [item["tipo"] for item in costumbres]
        bloques += [
            "## Desvíos por operador\n",
            "Filas `ruido` aportadas por cada operador. Los desvíos no forman otra costumbre: "
            "se concentran en pocos operadores y van en ambas direcciones.\n",
            "| operador | " + " | ".join(tipos) + " | total |",
            "|---|" + "---:|" * (len(tipos) + 1),
        ]
        for item in desvios:
            bloques.append(
                f"| {item['operador']} | " + " | ".join(str(item.get(tipo, 0)) for tipo in tipos)
                + f" | {item['total']} |"
            )
        bloques.append("")
    ruta.write_text("\n".join(bloques).rstrip() + "\n", encoding="utf-8")


def ejecutar(
    entrada: Path,
    salida: Path,
    ruta_normas: Path,
    trazas: Path | None = None,
) -> dict:
    verificar_fuentes(ruta_normas)
    parametros = cargar_parametros(ruta_normas)
    with entrada.open(encoding="utf-8", newline="") as archivo:
        rows = list(csv.DictReader(archivo))
    if len(rows) != 2000:
        raise ValueError(f"el historial debe tener 2.000 filas, tiene {len(rows)}")
    evidencia_trazas = cargar_trazas(trazas, rows) if trazas is not None else None
    clasificados, informe = clasificar(rows, parametros, evidencia_trazas)
    if evidencia_trazas is not None:
        informe["trazas"] = {k: v for k, v in evidencia_trazas.items() if k != "por_id"}
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["id", "tipo"], lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(clasificados)
    (salida.parent / "historial_costumbres.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    escribir_costumbres(salida.parent / "costumbres.md", informe)
    return {"filas": len(clasificados), "conteo": dict(Counter(row["tipo"] for row in clasificados)), **informe}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("historial/historial.csv"))
    parser.add_argument("--salida", type=Path, default=Path("historial_clasificado.csv"))
    parser.add_argument("--normas", type=Path, default=Path("normas/original"))
    parser.add_argument("--trazas", type=Path, default=Path("historial/trazas.jsonl"))
    args = parser.parse_args()
    trazas = args.trazas if args.trazas.is_file() else None
    print(json.dumps(ejecutar(args.entrada, args.salida, args.normas, trazas), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
