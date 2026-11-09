"""Medición y comparación reproducible de candidatos de prompt."""

from __future__ import annotations

import hashlib

from admision.comparar import CONTROLES
from admision.ejemplo import extraer_json

from .prompts import PromptSpec


def _ids_tokenizador(valor) -> list[int]:
    ids = valor.get("input_ids") if isinstance(valor, dict) else valor
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("tokenizador devolvió IDs inválidos o anidados")
    return ids


def medir_longitud_prompt(prompt: PromptSpec, tokenizer, paciente_id: str = "P-METRICA") -> dict:
    """Cuenta los textos configurables con el tokenizador recibido, sin aproximaciones.

    `contexto_inicial` usa además la plantilla de chat real y el identificador estable
    `paciente_id`. No descarga un tokenizer: producción debe inyectar el de la revisión
    fijada del modelo y los tests usan un doble explícito.
    """
    if not isinstance(prompt, PromptSpec):
        raise TypeError("prompt debe ser PromptSpec")
    if not isinstance(paciente_id, str) or not paciente_id.strip():
        raise ValueError("paciente_id de medición requiere texto no vacío")
    plantilla = getattr(tokenizer, "chat_template", None)
    if not isinstance(plantilla, str) or not plantilla:
        raise ValueError("tokenizador sin plantilla de chat")

    textos = {
        "system_prompt": prompt.system_prompt,
        "mensaje_inicial": prompt.mensaje_inicial,
        "formato_observacion": prompt.formato_observacion,
        "mensaje_json_invalido": prompt.mensaje_json_invalido,
    }
    por_texto = {}
    for nombre, texto in textos.items():
        try:
            ids = tokenizer.encode(texto, add_special_tokens=False)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("tokenizador no permite medir texto sin tokens especiales") from None
        por_texto[nombre] = len(_ids_tokenizador(ids))

    mensajes = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.mensaje_inicial_para(paciente_id)},
    ]
    try:
        salida = tokenizer.apply_chat_template(
            mensajes, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, padding=False, truncation=False, return_dict=True)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("tokenizador no permite medir el contexto inicial") from None
    contexto_inicial = len(_ids_tokenizador(salida))
    return {
        "version": 1,
        "metodo": "tokenizer_real_sin_aproximacion",
        "add_special_tokens_textos": False,
        "contexto_inicial_add_generation_prompt": True,
        "enable_thinking": False,
        "paciente_id_medicion": paciente_id,
        "por_texto": por_texto,
        "total_textos_configurables": sum(por_texto.values()),
        "contexto_inicial": contexto_inicial,
        "chat_template_sha256": hashlib.sha256(plantilla.encode("utf-8")).hexdigest(),
    }


def _redondear(valor: float, digitos: int = 3):
    return round(valor, digitos)


def calcular_metricas_candidato(resultados: list[dict], trazas: list[dict],
                                longitud_prompt: dict) -> dict:
    """Resume calidad y eficiencia; la clave lexicográfica prioriza casos 9/9.

    Los resultados deben provenir de ``admision.correr.puntuar``. Las trazas se
    usan solamente para contar respuestas que el mismo coordinador no reconoce
    como JSON; nunca se recalculan las fichas esperadas.
    """
    if not isinstance(resultados, list) or not resultados:
        raise ValueError("la métrica requiere resultados")
    if not isinstance(trazas, list) or len(trazas) != len(resultados):
        raise ValueError("resultados y trazas deben tener igual cantidad de casos")
    if (not isinstance(longitud_prompt, dict)
            or type(longitud_prompt.get("contexto_inicial")) is not int
            or type(longitud_prompt.get("por_texto", {}).get("system_prompt")) is not int):
        raise ValueError("medición de prompt inválida")

    por_traza = {}
    for traza in trazas:
        if not isinstance(traza, dict) or not isinstance(traza.get("caso"), str):
            raise ValueError("traza inválida")
        if traza["caso"] in por_traza:
            raise ValueError("caso repetido en trazas")
        por_traza[traza["caso"]] = traza

    controles_correctos = 0
    por_control = {control: 0 for control in CONTROLES}
    resueltos = sin_decision = invalidas = sin_json = llamadas = busquedas = 0
    caracteres = 0
    segundos = []
    casos_vistos = set()
    for resultado in resultados:
        caso = resultado.get("caso") if isinstance(resultado, dict) else None
        controles = resultado.get("controles") if isinstance(resultado, dict) else None
        if (not isinstance(caso, str) or caso in casos_vistos or caso not in por_traza
                or not isinstance(controles, dict) or set(controles) != set(CONTROLES)
                or any(type(controles[c]) is not bool for c in CONTROLES)):
            raise ValueError("resultado incompatible con los nueve controles")
        casos_vistos.add(caso)
        traza = por_traza[caso]
        mensajes, log = traza.get("mensajes"), traza.get("log")
        if not isinstance(mensajes, list) or not isinstance(log, list):
            raise ValueError("traza sin mensajes o log")
        correctos = sum(controles.values())
        controles_correctos += correctos
        for control in CONTROLES:
            por_control[control] += controles[control]
        resueltos += correctos == len(CONTROLES)
        sin_decision += resultado.get("obtenida") is None
        invalidas += int(resultado.get("invalidas", 0))
        llamadas += int(resultado.get("llamadas", len(log)))
        busquedas += int(resultado.get(
            "busquedas", sum(e.get("herramienta") == "buscar_normativa" for e in log if isinstance(e, dict))))
        caracteres += int(resultado.get("caracteres_de_contexto", 0))
        if isinstance(resultado.get("segundos"), (int, float)):
            segundos.append(float(resultado["segundos"]))
        sin_json += sum(
            extraer_json(m.get("content", "")) is None
            for m in mensajes if isinstance(m, dict) and m.get("role") == "assistant")

    if casos_vistos != set(por_traza):
        raise ValueError("resultados y trazas refieren a casos distintos")
    n, total_controles = len(resultados), len(CONTROLES) * len(resultados)
    tokens_system = longitud_prompt["por_texto"]["system_prompt"]
    # Un caso 9/9 adicional vale más que todos los controles parciales del conjunto.
    puntaje_calidad = resueltos * (total_controles + 1) + controles_correctos
    clave = [resueltos, controles_correctos, -sin_decision, -sin_json,
             -invalidas, -tokens_system, -longitud_prompt["contexto_inicial"], -caracteres]
    return {
        "version": 1,
        "casos": n,
        "resueltos_9_de_9": resueltos,
        "tasa_9_de_9": _redondear(resueltos / n),
        "controles_correctos": controles_correctos,
        "controles_totales": total_controles,
        "parcial_medio": _redondear(controles_correctos / total_controles),
        "por_control": {c: _redondear(por_control[c] / n) for c in CONTROLES},
        "sin_decision": sin_decision,
        "respuestas_sin_json": sin_json,
        "llamadas_invalidas": invalidas,
        "llamadas": {"total": llamadas, "medio": _redondear(llamadas / n)},
        "busquedas": {"total": busquedas, "medio": _redondear(busquedas / n)},
        "caracteres_de_contexto": {"total": caracteres, "medio": round(caracteres / n)},
        "segundos": {
            "casos_medidos": len(segundos),
            "total": _redondear(sum(segundos)),
            "medio": _redondear(sum(segundos) / len(segundos)) if segundos else None,
        },
        "longitud_prompt": longitud_prompt,
        "puntaje_calidad": puntaje_calidad,
        "criterio_comparacion": [
            "resueltos_9_de_9", "controles_correctos", "menos_sin_decision",
            "menos_respuestas_sin_json", "menos_llamadas_invalidas",
            "menos_tokens_system_prompt", "menos_tokens_contexto_inicial",
            "menos_caracteres_de_contexto",
        ],
        "clave_comparacion": clave,
    }


def clave_comparacion(metricas: dict) -> tuple:
    """Clave mayor-es-mejor para ordenar candidatos medidos sobre el mismo conjunto."""
    clave = metricas.get("clave_comparacion") if isinstance(metricas, dict) else None
    if (not isinstance(clave, list) or len(clave) != 8
            or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in clave)):
        raise ValueError("métricas sin clave de comparación válida")
    return tuple(clave)
