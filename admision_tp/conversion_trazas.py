"""Conversión fiel de trazas del kit; no cura decisiones ni elige prompt/procedencia."""

from dataclasses import replace
import json

from .conversaciones import validar_registro
from .prompts import PromptSpec, cargar_prompt_baseline


def prompt_desde_sistema(sistema):
    """Compatibilidad con `--sistema`: baseline con solo el mensaje de sistema reemplazado."""
    if not isinstance(sistema, str) or not sistema.strip():
        raise ValueError("identificadores, referencia y sistema requieren texto no vacío")
    return replace(cargar_prompt_baseline(), nombre="sistema-explicito", system_prompt=sistema)


def convertir_traza(traza, *, origen, referencia, prompt=None, sistema=None, buscar=None):
    """Devuelve v1 con acciones y observaciones originales, sin observación terminal.

    Usa todos los textos de `prompt` (PromptSpec); `sistema` es la forma anterior (solo
    mensaje de sistema sobre el baseline) y excluye a `prompt`. Si se pasa `buscar(consulta)`,
    las observaciones de `buscar_normativa` se reemplazan por lo que devuelve; el resto se conserva.
    """
    if (prompt is None) == (sistema is None):
        raise ValueError("indicar exactamente uno de prompt o sistema")
    prompt = prompt_desde_sistema(sistema) if prompt is None else prompt
    if not isinstance(prompt, PromptSpec):
        raise ValueError("prompt requiere un PromptSpec")
    campos = {"id", "paciente_id", "pasos", "ficha"}
    if not isinstance(traza, dict) or set(traza) not in (campos, campos | {"fecha", "operador"}):
        raise ValueError("campos de traza faltantes o desconocidos")
    for valor in [traza["id"], traza["paciente_id"], referencia,
                  *[traza[k] for k in ("fecha", "operador") if k in traza]]:
        if not isinstance(valor, str) or not valor.strip():
            raise ValueError("identificadores, referencia y sistema requieren texto no vacío")
    pasos = traza["pasos"]
    if not isinstance(pasos, list) or not pasos:
        raise ValueError("traza sin pasos")
    for orden, paso in enumerate(pasos, 1):
        if (not isinstance(paso, dict) or set(paso) != {"orden", "herramienta", "argumentos", "resultado"}
                or type(paso["orden"]) is not int or paso["orden"] != orden
                or not isinstance(paso["argumentos"], dict)
                or not isinstance(paso["resultado"], str) or not paso["resultado"].strip()):
            raise ValueError("paso incompleto, orden inválido u observación vacía")
        args = paso["argumentos"]
        if "paciente_id" in args and args["paciente_id"] != traza["paciente_id"]:
            raise ValueError("paciente de acción distinto del de la traza")
    if pasos[0]["herramienta"] != "leer_triage":
        raise ValueError("la traza debe comenzar con leer_triage")
    ficha = {k: v for k, v in pasos[-1]["argumentos"].items() if k != "fundamentos"}
    if (not isinstance(traza["ficha"], dict) or ficha != traza["ficha"]
            or any(type(v) is not type(traza["ficha"].get(k)) for k, v in ficha.items())):
        raise ValueError("ficha no coincide con la última acción")
    mensajes = [{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.mensaje_inicial_para(traza["paciente_id"])}]
    for paso in traza["pasos"]:
        mensajes.append({"role": "assistant", "content": json.dumps(
            {"herramienta": paso["herramienta"], "argumentos": paso["argumentos"]},
            ensure_ascii=False, allow_nan=False)})
        if paso["herramienta"] != "registrar_decision":
            resultado = paso["resultado"]
            if buscar is not None and paso["herramienta"] == "buscar_normativa":
                resultado = buscar(paso["argumentos"].get("consulta", ""))
            mensajes.append({"role": "user", "content":
                             prompt.observacion_para(paso["herramienta"], resultado)})
    registro = {"version": 1, "id": traza["id"], "origen": origen,
                "referencias": [f"{referencia}#{traza['id']}"], "messages": mensajes}
    validar_registro(registro, prompt.formato_observacion)
    return registro


def convertir_archivo(entrada, salida, *, origen, sistema=None, prompt=None):
    """Convierte un JSONL completo; publica solo si es válido y la salida no existe."""
    import hashlib
    import os
    from pathlib import Path
    import tempfile
    from .conversaciones import _json, validar_archivo

    entrada, salida = Path(entrada), Path(salida)
    if salida.exists() or salida.is_symlink():
        raise FileExistsError("la salida ya existe")
    contenido = entrada.read_bytes()
    try:
        lineas = contenido.decode("utf-8").splitlines()
    except UnicodeError:
        raise ValueError("entrada no es UTF-8") from None
    if not lineas:
        raise ValueError("archivo sin registros")
    registros = []
    for numero, linea in enumerate(lineas, 1):
        try:
            registros.append(convertir_traza(_json(linea), origen=origen, sistema=sistema,
                                             prompt=prompt, referencia=entrada.as_posix()))
        except (ValueError, TypeError, RecursionError):
            raise ValueError(f"línea {numero}: traza o contexto inválido") from None
    datos = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in registros).encode()
    # Temporal vecino: el enlace crea la salida completa de manera exclusiva, sin reemplazarla.
    with tempfile.NamedTemporaryFile(dir=salida.parent, prefix=".conversion-", suffix=".jsonl") as temporal:
        temporal.write(datos)
        temporal.flush()
        formato = (prompt or prompt_desde_sistema(sistema)).formato_observacion
        informe = validar_archivo(temporal.name, formato)
        if informe["errores"]:
            numero = informe["errores"][0]["linea"]
            raise ValueError(f"línea {numero}: conversación inválida o id duplicado")
        os.link(temporal.name, salida)
    return {"registros": len(registros), "entrada_sha256": hashlib.sha256(contenido).hexdigest(),
            "salida_sha256": informe["sha256"]}
