"""Selecciona filas correctas y convierte trazas reales al contrato de entrenamiento."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from admision.ejemplo import mensaje_de_sistema, primer_mensaje
from scripts.clasificar_historial import SI, aviso_segun_costumbre


def cargar_filas(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def cargar_clasificacion(ruta: Path) -> dict[str, str]:
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return {fila["id"]: fila["tipo"] for fila in csv.DictReader(archivo)}


def cargar_trazas(ruta: Path) -> dict[str, dict]:
    with ruta.open(encoding="utf-8") as archivo:
        trazas = [json.loads(linea) for linea in archivo if linea.strip()]
    ids = [traza.get("id") for traza in trazas]
    if len(ids) != len(set(ids)):
        raise ValueError("hay IDs de trazas duplicados")
    return {traza["id"]: traza for traza in trazas}


def costumbre_aviso(ruta: Path) -> dict:
    informe = json.loads(ruta.read_text(encoding="utf-8"))
    for costumbre in informe.get("costumbres", []):
        if costumbre.get("tipo") == "aviso_al_voluntariado":
            return costumbre
    raise ValueError("no se encontró la costumbre del aviso al voluntariado")


def seleccionar_ids(
    filas: list[dict[str, str]],
    clasificacion: dict[str, str],
    costumbre: dict,
) -> tuple[list[str], dict]:
    faltantes = {fila["id"] for fila in filas} - clasificacion.keys()
    if faltantes:
        raise ValueError(f"faltan clasificaciones para {len(faltantes)} filas")
    ids_error = sorted(i for i, tipo in clasificacion.items() if tipo == "error")
    ids_ruido = sorted(i for i, tipo in clasificacion.items() if tipo == "ruido")
    seleccionados = [fila["id"] for fila in filas if clasificacion[fila["id"]] == "correcta"]
    # Toda fila correcta sigue la costumbre completa; si no, la clasificación y el informe
    # de costumbres no corresponden entre sí y no se arma el dataset.
    inconsistentes = sorted(
        fila["id"] for fila in filas
        if clasificacion[fila["id"]] == "correcta"
        and aviso_segun_costumbre(fila, costumbre) != (fila["aviso_al_voluntariado"] == SI)
    )
    if inconsistentes:
        raise ValueError(f"filas correctas que contradicen la costumbre: {inconsistentes[:5]}")
    return seleccionados, {
        "total_historial": len(filas),
        "correctas_clasificadas": len(seleccionados),
        "excluidas_error": len(ids_error),
        "excluidas_ruido": len(ids_ruido),
        "seleccionadas": len(seleccionados),
    }


def _accion(herramienta: str, argumentos: dict) -> str:
    return json.dumps(
        {"herramienta": herramienta, "argumentos": argumentos},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def traza_a_conversacion(traza: dict) -> dict:
    pasos = traza.get("pasos", [])
    if not pasos or pasos[-1].get("herramienta") != "registrar_decision":
        raise ValueError(f"la traza {traza.get('id')} no termina en registrar_decision")
    mensajes = [
        {"role": "system", "content": mensaje_de_sistema()},
        {"role": "user", "content": primer_mensaje(traza["paciente_id"])},
    ]
    for paso in pasos:
        herramienta = paso.get("herramienta")
        argumentos = paso.get("argumentos")
        if not isinstance(herramienta, str) or not isinstance(argumentos, dict):
            raise ValueError(f"acción inválida en la traza {traza.get('id')}")
        mensajes.append({"role": "assistant", "content": _accion(herramienta, argumentos)})
        if herramienta != "registrar_decision":
            resultado = str(paso.get("resultado", "")).strip()
            if not resultado:
                raise ValueError(f"falta observación en la traza {traza.get('id')}")
            mensajes.append({
                "role": "user",
                "content": f"Resultado de {herramienta}:\n{resultado}",
            })
    return {
        "version": 1,
        "id": traza["id"],
        "origen": "historial",
        "referencias": [
            f"historial:id={traza['id']}",
            "fuente:historial/trazas.jsonl",
            "clasificacion:historial_clasificado.csv",
        ],
        "messages": mensajes,
    }


def convertir_historial(
    ruta_historial: Path,
    ruta_clasificacion: Path,
    ruta_costumbres: Path,
    ruta_trazas: Path,
) -> tuple[list[dict], dict]:
    filas = cargar_filas(ruta_historial)
    clasificacion = cargar_clasificacion(ruta_clasificacion)
    trazas = cargar_trazas(ruta_trazas)
    seleccionados, resumen = seleccionar_ids(filas, clasificacion, costumbre_aviso(ruta_costumbres))
    conversaciones = []
    for identificador in seleccionados:
        if identificador not in trazas:
            raise ValueError(f"falta la traza de {identificador}")
        conversaciones.append(traza_a_conversacion(trazas[identificador]))
    resumen["conversaciones_reales"] = len(conversaciones)
    resumen["ids_seleccionados"] = seleccionados
    return conversaciones, resumen
