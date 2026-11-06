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
SITUACION_FIELDS = ("motivo", "cobertura")
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


def razones_normativas(row: dict[str, str], parametros: dict[str, int]) -> list[str]:
    razones = []
    if row["registro"] != registro_esperado(row, parametros):
        razones.append("registro")
    if row["numero_documento"] != row["documento_numero"]:
        razones.append("numero_documento")

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
    activa = row["respuesta_verificacion"] == "activa"
    reservado = numero.endswith(str(parametros["P-COB-2"]))
    if not activa:
        cobertura, notificar = "sin_cobertura", ""
    elif reservado and tipo != "art":
        cobertura, notificar = "sin_cobertura", entidad
    else:
        cobertura, notificar = tipo, entidad
    if row["cobertura"] != cobertura:
        razones.append("cobertura")
    if row["entidad_a_notificar"] != notificar:
        razones.append("entidad_a_notificar")
    return razones


def analizar_variacion_operadores(
    limpias: list[dict[str, str]],
    umbral: int,
    trazas: dict | None = None,
) -> dict:
    """Describe una concentración por operador sin convertirla en regla."""
    menores = [row for row in limpias if int(row["edad"]) < umbral]
    grupos: dict[str, list[dict[str, str]]] = {}
    for row in menores:
        operador = row["operador"]
        if trazas is not None:
            operador = trazas["por_id"][row["id"]].get("operador", operador)
        grupos.setdefault(operador, []).append(row)

    evidencia = []
    for operador, grupo in sorted(grupos.items()):
        si = [row for row in grupo if row["aviso_al_voluntariado"] == SI]
        if len(grupo) < 30:
            continue
        evidencia.append({
            "operador": operador,
            "muestra": len(grupo),
            "si": len(si),
            "frecuencia_si": round(len(si) / len(grupo), 6),
            "ids_si": [row["id"] for row in si],
        })
    destacados = [item for item in evidencia if item["frecuencia_si"] >= 0.30]
    destacados_ids = {item["operador"] for item in destacados}
    resto = [row for row in menores if row["operador"] not in destacados_ids]
    resto_si = sum(row["aviso_al_voluntariado"] == SI for row in resto)
    return {
        "criterio": "al menos 30 filas limpias menores del umbral y frecuencia de aviso >= 30%",
        "grupos": evidencia,
        "grupos_destacados": [item["operador"] for item in destacados],
        "resto": {
            "muestra": len(resto),
            "si": resto_si,
            "frecuencia_si": round(resto_si / len(resto), 6) if resto else 0,
        },
        "confirmada_como_costumbre_hospital": False,
        "motivo_no_confirmacion": (
            "la señal está concentrada en un subgrupo de operadores y no describe una práctica mayoritaria del hospital"
        ),
    }


def inferir_costumbres(rows: list[dict[str, str]], parametros: dict[str, int]) -> list[dict]:
    """Detecta costumbres solo en filas limpias y en situaciones comparables."""
    razones = {row["id"]: razones_normativas(row, parametros) for row in rows}
    limpias = [row for row in rows if not razones[row["id"]]]
    costumbres = []

    por_edad: dict[int, list[dict[str, str]]] = {}
    for row in limpias:
        por_edad.setdefault(int(row["edad"]), []).append(row)
    candidatos = []
    for umbral in sorted(por_edad):
        actual = por_edad[umbral]
        anteriores = [row for edad in range(umbral - 3, umbral) for row in por_edad.get(edad, [])]
        if len(actual) < 10 or len(anteriores) < 10:
            continue
        actual_si = sum(row["aviso_al_voluntariado"] == SI for row in actual) / len(actual)
        anterior_si = sum(row["aviso_al_voluntariado"] == SI for row in anteriores) / len(anteriores)
        if actual_si < 0.50 or anterior_si >= 0.20:
            continue
        menores = [row for row in limpias if int(row["edad"]) < umbral]
        mayores = [row for row in limpias if int(row["edad"]) >= umbral]
        if len(menores) < 100 or len(mayores) < 100:
            continue
        no_menores = sum(row["aviso_al_voluntariado"] != SI for row in menores) / len(menores)
        if no_menores >= 0.90:
            candidatos.append((actual_si - anterior_si, umbral, menores, mayores))

    if candidatos:
        _, umbral, menores, mayores = max(candidatos)
        estratos = {}
        for row in limpias:
            clave = tuple(row[campo] for campo in SITUACION_FIELDS)
            grupo = estratos.setdefault(clave, {"menores": [], "mayores": []})
            grupo["menores" if int(row["edad"]) < umbral else "mayores"].append(row)
        comparables = []
        for clave, grupo in sorted(estratos.items()):
            if len(grupo["menores"]) < 20 or len(grupo["mayores"]) < 20:
                continue
            no = sum(row["aviso_al_voluntariado"] != SI for row in grupo["menores"])
            si = sum(row["aviso_al_voluntariado"] == SI for row in grupo["mayores"])
            if no / len(grupo["menores"]) >= 0.85:
                comparables.append({
                    "situacion": dict(zip(SITUACION_FIELDS, clave)),
                    "muestra_menores": len(grupo["menores"]),
                    "no_menores": no,
                    "frecuencia_no_menores": round(no / len(grupo["menores"]), 6),
                    "muestra_mayores": len(grupo["mayores"]),
                    "si_mayores": si,
                    "frecuencia_si_mayores": round(si / len(grupo["mayores"]), 6),
                    "ids_respaldo": [row["id"] for row in grupo["menores"] if row["aviso_al_voluntariado"] != SI],
                    "ids_ruido": [row["id"] for row in grupo["menores"] if row["aviso_al_voluntariado"] == SI],
                })
        costumbres.append({
            "tipo": "aviso_al_voluntariado",
            "nombre": "no avisar al voluntariado a pacientes menores del umbral observado",
            "umbral_edad": umbral,
            "situacion_fields": list(SITUACION_FIELDS),
            "muestra_menores_limpias": len(menores),
            "no_menores_limpias": sum(row["aviso_al_voluntariado"] != SI for row in menores),
            "muestra_mayores_limpias": len(mayores),
            "si_mayores_limpias": sum(row["aviso_al_voluntariado"] == SI for row in mayores),
            "estratos_comparables": comparables,
            "ids_respaldo": sorted({i for estrato in comparables for i in estrato["ids_respaldo"]}),
            "ids_ruido": sorted({i for estrato in comparables for i in estrato["ids_ruido"]}),
        })

    multiples = []
    for row in limpias:
        if len(entradas_cobertura(row["credenciales"])) > 1:
            seleccionada = fuente_cobertura(row)
            if seleccionada:
                multiples.append((row, seleccionada[0]))
    preferencias = Counter(tipo for _, tipo in multiples)
    if multiples:
        preferido, cantidad = preferencias.most_common(1)[0]
        if cantidad / len(multiples) >= 0.90:
            costumbres.append({
                "tipo": "seleccion_credencial",
                "nombre": f"priorizar {preferido} cuando se presentan dos credenciales",
                "muestra": len(multiples),
                "preferido": preferido,
                "frecuencia_preferido": round(cantidad / len(multiples), 6),
                "frecuencias": dict(sorted(preferencias.items())),
                "ids_respaldo": sorted(row["id"] for row, tipo in multiples if tipo == preferido),
                "ids_ruido": sorted(row["id"] for row, tipo in multiples if tipo != preferido),
            })
    return costumbres


def clasificar(
    rows: list[dict[str, str]],
    parametros: dict[str, int],
    trazas: dict | None = None,
) -> tuple[list[dict[str, str]], dict]:
    costumbres = inferir_costumbres(rows, parametros)
    razones = {row["id"]: razones_normativas(row, parametros) for row in rows}
    limpias = [row for row in rows if not razones[row["id"]]]
    edad = next((item for item in costumbres if item["tipo"] == "aviso_al_voluntariado"), None)
    patrones = []
    if edad is not None:
        patrones.append(analizar_variacion_operadores(limpias, edad["umbral_edad"], trazas))
    ids_ruido = {i for costumbre in costumbres for i in costumbre.get("ids_ruido", [])}
    salida = []
    for row in rows:
        tipo = "error" if razones[row["id"]] else "ruido" if row["id"] in ids_ruido else "correcta"
        salida.append({"id": row["id"], "tipo": tipo})
    return salida, {"costumbres": costumbres, "patrones_adicionales": patrones}


def escribir_costumbres(ruta: Path, informe: dict) -> None:
    costumbres = informe.get("costumbres", [])
    if not costumbres:
        ruta.write_text("# Costumbres detectadas\n\nNo se detectaron costumbres con evidencia suficiente.\n", encoding="utf-8")
        return
    bloques = ["# Costumbres detectadas\n"]
    for item in costumbres:
        bloques.append(f"## {item['nombre']}\n")
        if item["tipo"] == "aviso_al_voluntariado":
            bloques.append(
                f"Se compararon estratos por {', '.join(item['situacion_fields'])}, excluyendo errores normativos. "
                f"Umbral: **{item['umbral_edad']}**; estratos comparables: {len(item['estratos_comparables'])}. "
                f"En filas limpias menores, `no`: {item['no_menores_limpias']}/{item['muestra_menores_limpias']}; "
                f"en mayores, `si`: {item['si_mayores_limpias']}/{item['muestra_mayores_limpias']}.\n\n"
            )
        else:
            bloques.append(
                f"Frecuencias: {item['frecuencias']} sobre {item['muestra']} filas; "
                f"preferencia observada: {item['frecuencia_preferido']:.1%}.\n\n"
            )
        bloques.append(f"IDs de respaldo: {', '.join(item.get('ids_respaldo', []))}.\n\n")
        bloques.append(f"IDs clasificados como ruido: {', '.join(item.get('ids_ruido', []))}.\n\n")
    patrones = informe.get("patrones_adicionales", [])
    if patrones:
        bloques.append("## Patrones adicionales investigados\n")
        for patron in patrones:
            bloques.append(
                "Se examinó la variación del aviso entre operadores en pacientes menores del umbral, "
                "usando sólo filas sin violaciones normativas.\n\n"
            )
            for grupo in patron["grupos"]:
                bloques.append(
                    f"- **{grupo['operador']}**: {grupo['si']}/{grupo['muestra']} "
                    f"({grupo['frecuencia_si']:.1%}); IDs: {', '.join(grupo['ids_si'])}.\n"
                )
            resto = patron["resto"]
            bloques.append(
                f"\nLos otros operadores suman {resto['si']}/{resto['muestra']} "
                f"({resto['frecuencia_si']:.1%}). No se confirma como costumbre hospitalaria: "
                f"la señal es de un subgrupo y no de una práctica mayoritaria.\n\n"
            )
    ruta.write_text("\n".join(bloques), encoding="utf-8")


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
        escritor = csv.DictWriter(archivo, fieldnames=["id", "tipo"])
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
