"""Genera y resuelve conversaciones sintéticas difíciles sin usar casos de práctica."""

from __future__ import annotations

import json
import hashlib
import unicodedata
from collections import Counter
from dataclasses import replace
from pathlib import Path

from admision.entorno import Entorno
from admision.ejemplo import mensaje_de_sistema, primer_mensaje
from admision.generador import generar
from admision.normas import cargar_normas
from admision_tp.conversaciones import validar_registro
from scripts.clasificar_historial import (
    SI,
    cargar_parametros,
    circuito_esperado,
    entradas_cobertura,
    fuente_cobertura,
    plazo_esperado,
    razones_normativas,
    registro_esperado,
)


CIRCUITOS = ("ecg", "codigo_acv", "sepsis_probable", "sepsis_posible", "ninguno")
BUCKETS_ESPECIALES = ("art", "dos_credenciales", "acompanante", "sin_nadie")


def normalizar(texto: str) -> str:
    return "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto.casefold())
        if unicodedata.category(caracter) != "Mn"
    )


def circuito_caso(caso) -> str:
    motivo = normalizar(caso.motivo)
    if motivo == "dolor toracico":
        return "ecg"
    if motivo.startswith("signos neurologicos"):
        return "codigo_acv"
    if caso.sepsis in ("probable", "con signos de shock"):
        return "sepsis_probable"
    if caso.sepsis != "no":
        return "sepsis_posible"
    return "ninguno"


def dificultad_caso(caso) -> str:
    if caso.trabajo:
        return "art"
    if len(caso.credenciales) == 2:
        return "dos_credenciales"
    if not caso.puede_responder and caso.acompanante:
        return "acompanante"
    if not caso.puede_responder:
        return "sin_nadie"
    return "general"


def seleccionar_casos(casos: list, por_bucket: int = 60, por_circuito: int = 100) -> list:
    seleccionados = []
    usados = set()
    for bucket in BUCKETS_ESPECIALES:
        disponibles = [caso for caso in casos if dificultad_caso(caso) == bucket]
        if len(disponibles) < por_bucket:
            raise ValueError(f"no hay {por_bucket} casos para el bucket {bucket}")
        for caso in disponibles[:por_bucket]:
            seleccionados.append(caso)
            usados.add(caso.id)

    conteo_circuitos = Counter(circuito_caso(caso) for caso in seleccionados)
    for circuito in CIRCUITOS:
        disponibles = [
            caso for caso in casos
            if caso.id not in usados and circuito_caso(caso) == circuito
        ]
        faltan = por_circuito - conteo_circuitos[circuito]
        if len(disponibles) < faltan:
            raise ValueError(f"no hay suficientes casos para el circuito {circuito}")
        for caso in disponibles[:faltan]:
            seleccionados.append(caso)
            usados.add(caso.id)
        conteo_circuitos[circuito] += faltan
    return sorted(seleccionados, key=lambda caso: caso.id)


def serializar_credenciales(credenciales: tuple) -> str:
    return "; ".join(
        f"{entidad} ({'obra social' if tipo == 'obra_social' else 'prepaga'}) {numero}"
        for tipo, entidad, numero in credenciales
    )


def serializar_declarada(declarada: tuple | None) -> str:
    return serializar_credenciales((declarada,)) if declarada else ""


def serializar_art(art: tuple | None) -> str:
    return f"{art[0]} CUIL {art[1]}" if art else ""


def _fuente_sintetica(caso, fila_base: dict[str, str]) -> tuple[str, str, str] | None:
    if caso.trabajo:
        return "art", caso.art[0], caso.art[1]
    if len(caso.credenciales) > 1:
        prepaga = next(credencial for credencial in caso.credenciales if credencial[0] == "prepaga")
        return "prepaga", prepaga[1], prepaga[2]
    return fuente_cobertura(fila_base)


def deducir_ficha(
    caso,
    parametros: dict[str, int],
    umbral_edad: int,
    aviso_al_voluntariado: bool = False,
) -> tuple[dict, dict]:
    fila = {
        "id": caso.id,
        "edad": str(caso.edad),
        "nivel": str(caso.nivel),
        "motivo": caso.motivo,
        "sospecha_sepsis": caso.sepsis,
        "llegada": caso.llegada,
        "documento_tipo": caso.documento_tipo or "",
        "documento_numero": caso.documento_numero or "",
        "credenciales": serializar_credenciales(caso.credenciales),
        "lesion_trabajo": caso.trabajo or "no",
        "cobertura_declarada": serializar_declarada(caso.declarada),
        "art_declarada": serializar_art(caso.art),
        "entidad_verificada": "",
        "numero_verificado": "",
        "respuesta_verificacion": "",
    }
    fuente = _fuente_sintetica(caso, fila)
    if fuente:
        tipo, entidad, numero = fuente
        fila["entidad_verificada"] = entidad
        fila["numero_verificado"] = numero
        clave = f"{entidad}|{numero}"
        if caso.trabajo:
            clave = f"{caso.art[0]}|{caso.art[1]}"
        fila["respuesta_verificacion"] = "activa" if caso.afiliaciones.get(clave, False) else "inactiva"

    fila["registro"] = registro_esperado(fila, parametros)
    fila["numero_documento"] = fila["documento_numero"]
    fila["box_antes_del_registro"] = SI if (
        caso.nivel <= parametros["P-ADM-1"] or caso.edad >= parametros["P-ADM-2"]
    ) else "no"
    fila["circuito"] = circuito_esperado(fila)
    fila["plazo_circuito_min"] = plazo_esperado(fila, parametros)

    if fuente is None:
        fila["cobertura"] = "sin_cobertura"
        fila["entidad_a_notificar"] = ""
    else:
        tipo, entidad, numero = fuente
        activa = fila["respuesta_verificacion"] == "activa"
        reservado = numero.endswith(str(parametros["P-COB-2"]))
        if not activa:
            fila["cobertura"], fila["entidad_a_notificar"] = "sin_cobertura", ""
        elif reservado and tipo != "art":
            fila["cobertura"], fila["entidad_a_notificar"] = "sin_cobertura", entidad
        else:
            fila["cobertura"], fila["entidad_a_notificar"] = tipo, entidad

    if razones_normativas(fila, parametros):
        raise ValueError(f"la deducción sintética viola normas: {caso.id}")
    fundamentos = ["NOR-ADM-01"]
    if caso.edad >= parametros["P-ADM-2"]:
        fundamentos.append("NOR-ADM-02")
    if caso.documento_tipo and caso.documento_tipo != "DNI":
        fundamentos.append("NOR-ADM-03")
    if caso.credenciales or caso.declarada or caso.trabajo:
        fundamentos.extend(["NOR-COB-01", "NOR-COB-02"])
    if fila["cobertura"] == "sin_cobertura":
        fundamentos.append("NOR-COB-03")
    if fuente and fila["numero_verificado"].endswith(str(parametros["P-COB-2"])) and fuente[0] != "art":
        fundamentos.append("NOR-COB-04")
    fundamentos_por_circuito = {
        "ecg": "NOR-TDE-01",
        "codigo_acv": "NOR-TDE-02",
        "codigo_sepsis": "NOR-TDE-03",
    }
    if fila["circuito"] in fundamentos_por_circuito:
        fundamentos.append(fundamentos_por_circuito[fila["circuito"]])
        if caso.llegada == "en ambulancia":
            fundamentos.append("NOR-TDE-04")
    ficha = {
        "registro": fila["registro"],
        "numero_documento": fila["numero_documento"] or None,
        "box_antes_del_registro": fila["box_antes_del_registro"] == SI,
        "circuito": fila["circuito"],
        "plazo_circuito_min": int(fila["plazo_circuito_min"]) if fila["plazo_circuito_min"] else None,
        "cobertura": fila["cobertura"],
        "entidad_a_notificar": fila["entidad_a_notificar"] or None,
        "aviso_al_voluntariado": aviso_al_voluntariado,
        "fundamentos": list(dict.fromkeys(fundamentos)),
    }
    return fila, ficha


def probabilidad_aviso_mayores(costumbres: dict) -> tuple[int, int, float]:
    costumbre = next(
        item for item in costumbres["costumbres"]
        if item["tipo"] == "aviso_al_voluntariado"
    )
    muestra = int(costumbre["muestra_mayores_limpias"])
    avisos = int(costumbre["si_mayores_limpias"])
    if not muestra or not 0 <= avisos <= muestra:
        raise ValueError("evidencia inválida para el aviso en mayores")
    return avisos, muestra, avisos / muestra


def aviso_observado(
    edad: int,
    umbral_edad: int,
    probabilidad_mayores: float,
    semilla: int,
    caso_id: str,
) -> bool:
    """Aplica la costumbre confirmada y muestrea la señal mixta de mayores."""
    if edad < umbral_edad:
        return False
    digest = hashlib.sha256(f"{semilla}:{caso_id}".encode("ascii")).digest()
    valor = int.from_bytes(digest[:8], "big") / 2**64
    return valor < probabilidad_mayores


class BuscadorSintetico:
    """Buscador determinista sobre las normas originales para las observaciones."""

    k_entorno = 2

    def __init__(self, ruta_normas: Path):
        self.secciones = cargar_normas(ruta_normas)

    def recuperar(self, consulta: str, k: int) -> list:
        tokens = set(normalizar(consulta).split())
        puntajes = []
        for seccion in self.secciones:
            texto = normalizar(seccion.id + " " + seccion.texto)
            puntaje = sum(token in texto for token in tokens)
            puntajes.append((puntaje, seccion.id, seccion))
        puntajes.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in puntajes[:k]]


def _registrar(mensajes: list[dict], entorno: Entorno, herramienta: str, argumentos: dict) -> None:
    mensajes.append({
        "role": "assistant",
        "content": json.dumps(
            {"herramienta": herramienta, "argumentos": argumentos},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    })
    observacion, terminado = entorno.paso(herramienta, argumentos)
    if herramienta != "registrar_decision":
        if terminado:
            raise ValueError("el caso sintético terminó antes de registrar la ficha")
        mensajes.append({
            "role": "user",
            "content": f"Resultado de {herramienta}:\n{observacion}",
        })


def caso_a_conversacion(caso, ficha: dict, ruta_normas: Path) -> dict:
    entorno = Entorno(caso, BuscadorSintetico(ruta_normas))
    mensajes = [
        {"role": "system", "content": mensaje_de_sistema()},
        {"role": "user", "content": primer_mensaje(caso.paciente_id)},
    ]
    _registrar(mensajes, entorno, "leer_triage", {"paciente_id": caso.paciente_id})
    _registrar(mensajes, entorno, "buscar_normativa", {"consulta": "registro y ubicación en el box"})
    _registrar(mensajes, entorno, "buscar_normativa", {"consulta": f"circuito {circuito_caso(caso)} y plazo"})

    quien = "paciente" if caso.puede_responder else "acompanante"
    if not caso.credenciales:
        if caso.puede_responder or caso.acompanante:
            _registrar(mensajes, entorno, "buscar_normativa", {"consulta": "consulta de cobertura"})
            _registrar(mensajes, entorno, "preguntar", {
                "paciente_id": caso.paciente_id, "a_quien": quien, "tema": "cobertura",
            })
    if caso.trabajo and (caso.puede_responder or caso.acompanante):
        _registrar(mensajes, entorno, "buscar_normativa", {"consulta": "ART y accidente laboral"})
        _registrar(mensajes, entorno, "preguntar", {
            "paciente_id": caso.paciente_id, "a_quien": quien, "tema": "art",
        })
    if caso.credenciales or caso.declarada or caso.trabajo:
        _registrar(mensajes, entorno, "buscar_normativa", {"consulta": "verificación de afiliación"})
        if ficha["cobertura"] == "art" or caso.trabajo:
            entidad, numero = caso.art
        elif len(caso.credenciales) > 1:
            prepaga = next(c for c in caso.credenciales if c[0] == "prepaga")
            entidad, numero = prepaga[1], prepaga[2]
        elif caso.credenciales:
            entidad, numero = caso.credenciales[0][1], caso.credenciales[0][2]
        elif caso.declarada:
            entidad, numero = caso.declarada[1], caso.declarada[2]
        else:
            entidad = numero = ""
        if entidad:
            _registrar(mensajes, entorno, "verificar_afiliacion", {
                "entidad": entidad, "numero_afiliado": numero,
            })
    _registrar(mensajes, entorno, "registrar_decision", ficha)
    if not entorno.terminado or entorno.decision != ficha:
        raise ValueError(f"el entorno rechazó la ficha sintética {caso.id}")
    return {
        "version": 1,
        "id": caso.id,
        "origen": "sintetico",
        "referencias": [
            "generador:admision.generador",
            "configuracion:semilla-12012",
            "normas:normas/original",
            "costumbres:issue-11",
        ],
        "messages": mensajes,
    }


def generar_sinteticos(
    ruta_normas: Path,
    ruta_costumbres: Path,
    semilla: int = 12012,
    candidatos: int = 5000,
) -> tuple[list[dict], list[dict], dict]:
    costumbres = json.loads(ruta_costumbres.read_text(encoding="utf-8"))
    umbral = next(
        int(item["umbral_edad"])
        for item in costumbres["costumbres"]
        if item["tipo"] == "aviso_al_voluntariado"
    )
    avisos_mayores, muestra_mayores, probabilidad_mayores = probabilidad_aviso_mayores(costumbres)
    parametros = cargar_parametros(ruta_normas)
    casos = generar(semilla=semilla, n=candidatos, prefijo=f"cand-{semilla}")
    seleccionados = seleccionar_casos(casos)
    conversaciones = []
    auditoria = []
    for indice, caso in enumerate(seleccionados, 1):
        aviso = aviso_observado(
            caso.edad, umbral, probabilidad_mayores, semilla, caso.id
        )
        fila, ficha = deducir_ficha(caso, parametros, umbral, aviso)
        caso_final = replace(caso, id=f"S-{semilla}-{indice:04d}")
        conversacion = caso_a_conversacion(caso_final, ficha, ruta_normas)
        validar_registro(conversacion)
        conversaciones.append(conversacion)
        auditoria.append({
            "id": conversacion["id"],
            "caso_generado": caso.id,
            "circuito": circuito_caso(caso),
            "dificultad": dificultad_caso(caso),
            "edad": caso.edad,
            "cobertura": ficha["cobertura"],
            "aviso_mayores_observado": (
                caso.edad >= umbral and ficha["aviso_al_voluntariado"]
            ),
            "ficha": ficha,
            "normas_verificadas": not razones_normativas(fila, parametros),
        })
    resumen = {
        "semilla": semilla,
        "candidatos_generados": candidatos,
        "sinteticos_seleccionados": len(conversaciones),
        "por_circuito": dict(sorted(Counter(item["circuito"] for item in auditoria).items())),
        "por_dificultad": dict(sorted(Counter(item["dificultad"] for item in auditoria).items())),
        "fichas_normativamente_validas": sum(item["normas_verificadas"] for item in auditoria),
        "umbral_costumbre_edad": umbral,
        "avisos_mayores_observados": avisos_mayores,
        "muestra_mayores_observada": muestra_mayores,
        "probabilidad_aviso_mayores": round(probabilidad_mayores, 6),
    }
    return conversaciones, auditoria, resumen
