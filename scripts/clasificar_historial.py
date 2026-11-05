"""Clasifica el historial del hospital de forma determinista.

La salida principal es ``historial_clasificado.csv``. También se generan
``costumbres.md`` y ``historial_costumbres.json`` para conservar el análisis
reutilizable en el curado de conversaciones del issue #11.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TIPOS = {"correcta", "error", "ruido"}
SI = "sí"
SECCIONES = (
    "NOR-ADM-01",
    "NOR-ADM-02",
    "NOR-ADM-03",
    "NOR-COB-01",
    "NOR-COB-02",
    "NOR-COB-03",
    "NOR-COB-04",
    "NOR-TDE-01",
    "NOR-TDE-02",
    "NOR-TDE-03",
    "NOR-TDE-04",
)


def cargar_parametros(ruta_normas: Path) -> dict[str, int]:
    """Lee los parámetros numéricos usados por las reglas observables."""
    texto = (ruta_normas / "parametros.yaml").read_text(encoding="utf-8")
    encontrados = {
        codigo: int(valor)
        for codigo, valor in re.findall(
            r"codigo:\s*(P-[A-Z0-9-]+)\s*\n\s*valor:\s*([0-9]+)", texto
        )
    }
    requeridos = {"P-ADM-1", "P-ADM-2", "P-ADM-3", "P-COB-2", "P-TDE-1", "P-TDE-2", "P-TDE-3", "P-TDE-4", "P-TDE-5"}
    faltantes = requeridos - encontrados.keys()
    if faltantes:
        raise ValueError(f"faltan parámetros en normas/original: {sorted(faltantes)}")
    return encontrados


def verificar_fuentes(ruta_normas: Path) -> None:
    faltantes = [f"{seccion}.md" for seccion in SECCIONES if not (ruta_normas / f"{seccion}.md").is_file()]
    if faltantes or not (ruta_normas / "parametros.yaml").is_file():
        raise FileNotFoundError(f"fuentes normativas incompletas: {faltantes}")


def entradas_cobertura(texto: str) -> list[tuple[str, str, str]]:
    return re.findall(r"([^;]+?) \((obra social|prepaga)\) ([^;]+)", texto)


def fuente_cobertura(row: dict[str, str]) -> tuple[str, str, str] | None:
    if row["lesion_trabajo"] != "no":
        match = re.search(r"(.+ ART) CUIL (.+)$", row["art_declarada"])
        return ("art", match.group(1), match.group(2)) if match else None

    entradas = entradas_cobertura(row["credenciales"])
    if not entradas:
        entradas = entradas_cobertura(row["cobertura_declarada"])
    if not entradas:
        return None
    entidad, tipo, numero = entradas[0]
    tipo_normalizado = {"obra social": "obra_social", "prepaga": "prepaga"}[tipo]
    return tipo_normalizado, entidad, numero


def circuito_esperado(row: dict[str, str]) -> str:
    if row["motivo"] == "dolor tor\u00e1cico":
        return "ecg"
    if row["motivo"] == "signos neurol\u00f3gicos focales de inicio s\u00fabito":
        return "codigo_acv"
    if row["sospecha_sepsis"] != "no":
        return "codigo_sepsis"
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


def registro_esperado(row: dict[str, str]) -> str:
    if not row["documento_tipo"]:
        return "provisorio"
    if row["documento_tipo"] == "DNI":
        return "definitivo"
    return "provisorio" if int(row["nivel"]) <= 2 else "definitivo"


def razones_normativas(row: dict[str, str], parametros: dict[str, int]) -> list[str]:
    razones: list[str] = []
    if row["registro"] != registro_esperado(row):
        razones.append("registro")

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
        if any(row[campo] for campo in ("entidad_verificada", "numero_verificado", "respuesta_verificacion", "entidad_a_notificar")):
            razones.append("verificacion_sin_fuente")
        if row["cobertura"] != "sin_cobertura":
            razones.append("cobertura")
        return razones

    tipo, entidad, numero = fuente
    if row["entidad_verificada"] != entidad or row["numero_verificado"] != numero:
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


def inferir_costumbre(rows: list[dict[str, str]], parametros: dict[str, int]) -> dict:
    """Encuentra el quiebre etario fuerte y conserva sus frecuencias e IDs.

    Se busca un salto local en la frecuencia por edad: la frecuencia de
    ``sí`` en una edad debe superar 50 %, la media de las tres edades
    anteriores debe ser menor a 20 % y debe haber al menos diez observaciones
    en cada edad. El umbral se aprende del historial, no de una norma.
    """
    por_edad: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        por_edad.setdefault(int(row["edad"]), []).append(row)
    candidatos = []
    edades = sorted({int(row["edad"]) for row in rows})
    for umbral in edades:
        actual = por_edad.get(umbral, [])
        anteriores = [
            row
            for edad in range(umbral - 3, umbral)
            for row in por_edad.get(edad, [])
        ]
        if len(actual) < 10 or len(anteriores) < 10:
            continue
        frecuencia_actual = sum(row["aviso_al_voluntariado"] == SI for row in actual) / len(actual)
        frecuencia_anterior = sum(row["aviso_al_voluntariado"] == SI for row in anteriores) / len(anteriores)
        if frecuencia_actual < 0.50 or frecuencia_anterior >= 0.20:
            continue
        menores = [row for row in rows if int(row["edad"]) < umbral]
        mayores = [row for row in rows if int(row["edad"]) >= umbral]
        if len(menores) < 100 or len(mayores) < 100:
            continue
        no_menores = sum(row["aviso_al_voluntariado"] != SI for row in menores) / len(menores)
        si_mayores = sum(row["aviso_al_voluntariado"] == SI for row in mayores) / len(mayores)
        if no_menores >= 0.90 and si_mayores >= 0.50:
            candidatos.append((frecuencia_actual - frecuencia_anterior, umbral, no_menores, si_mayores, menores, mayores))
    if not candidatos:
        return {"detectada": False, "motivo": "no hay una costumbre con evidencia suficiente"}

    _, umbral, no_menores, si_mayores, menores, mayores = max(candidatos)
    return {
        "detectada": True,
        "nombre": "no avisar al voluntariado a pacientes menores del umbral observado",
        "umbral_edad": umbral,
        "situacion": "edad menor al umbral, comparada con el mismo campo de salida en el resto del historial",
        "muestra_menores": len(menores),
        "no_menores": sum(row["aviso_al_voluntariado"] != SI for row in menores),
        "frecuencia_no_menores": round(no_menores, 6),
        "muestra_mayores": len(mayores),
        "si_mayores": sum(row["aviso_al_voluntariado"] == SI for row in mayores),
        "frecuencia_si_mayores": round(si_mayores, 6),
        "ids_respaldo": [row["id"] for row in menores if row["aviso_al_voluntariado"] != SI],
        "ids_ruido": [row["id"] for row in menores if row["aviso_al_voluntariado"] == SI],
    }


def clasificar(rows: list[dict[str, str]], parametros: dict[str, int]) -> tuple[list[dict[str, str]], dict]:
    costumbre = inferir_costumbre(rows, parametros)
    salida = []
    for row in rows:
        razones = razones_normativas(row, parametros)
        if razones:
            tipo = "error"
        elif costumbre.get("detectada") and int(row["edad"]) < costumbre["umbral_edad"] and row["aviso_al_voluntariado"] == SI:
            tipo = "ruido"
        else:
            tipo = "correcta"
        salida.append({"id": row["id"], "tipo": tipo})
    return salida, costumbre


def escribir_costumbres(ruta: Path, costumbre: dict) -> None:
    if not costumbre.get("detectada"):
        texto = "# Costumbres detectadas\n\nNo se detectó una costumbre con evidencia suficiente.\n"
    else:
        texto = (
            "# Costumbres detectadas\n\n"
            f"## {costumbre['nombre']}\n\n"
            f"En la situación `{costumbre['situacion']}`, `no` aparece en "
            f"{costumbre['no_menores']}/{costumbre['muestra_menores']} "
            f"({costumbre['frecuencia_no_menores']:.1%}) de las filas. "
            f"En el grupo de comparación (edad mayor o igual al umbral), `sí` aparece en "
            f"{costumbre['si_mayores']}/{costumbre['muestra_mayores']} "
            f"({costumbre['frecuencia_si_mayores']:.1%}); como el comportamiento allí es mixto, "
            "no se marcaron sus minorías como ruido.\n\n"
            f"Umbral observado: **{costumbre['umbral_edad']} años**. "
            f"IDs que sostienen la costumbre: {', '.join(costumbre['ids_respaldo'])}.\n\n"
            f"IDs clasificados como ruido por apartarse de ella: {', '.join(costumbre['ids_ruido'])}.\n"
        )
    ruta.write_text(texto, encoding="utf-8")


def ejecutar(entrada: Path, salida: Path, ruta_normas: Path) -> dict:
    verificar_fuentes(ruta_normas)
    parametros = cargar_parametros(ruta_normas)
    with entrada.open(encoding="utf-8", newline="") as archivo:
        rows = list(csv.DictReader(archivo))
    if len(rows) != 2000:
        raise ValueError(f"el historial debe tener 2.000 filas, tiene {len(rows)}")
    clasificados, costumbre = clasificar(rows, parametros)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["id", "tipo"])
        escritor.writeheader()
        escritor.writerows(clasificados)

    json_path = salida.parent / "historial_costumbres.json"
    json_path.write_text(json.dumps(costumbre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    escribir_costumbres(salida.parent / "costumbres.md", costumbre)
    return {"filas": len(clasificados), "conteo": dict(Counter(row["tipo"] for row in clasificados)), "costumbre": costumbre}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("historial/historial.csv"))
    parser.add_argument("--salida", type=Path, default=Path("historial_clasificado.csv"))
    parser.add_argument("--normas", type=Path, default=Path("normas/original"))
    args = parser.parse_args()
    print(json.dumps(ejecutar(args.entrada, args.salida, args.normas), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
