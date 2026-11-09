"""Conversión fiel de trazas del kit; no cura decisiones ni elige prompt/procedencia."""

import json

from admision.ejemplo import primer_mensaje
from .conversaciones import validar_registro


def convertir_traza(traza, *, origen, sistema, referencia):
    """Devuelve v1 con acciones y observaciones originales, sin observación terminal."""
    campos = {"id", "paciente_id", "pasos", "ficha"}
    if not isinstance(traza, dict) or set(traza) not in (campos, campos | {"fecha", "operador"}):
        raise ValueError("campos de traza faltantes o desconocidos")
    for valor in [traza["id"], traza["paciente_id"], referencia, sistema,
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
    mensajes = [{"role": "system", "content": sistema},
                {"role": "user", "content": primer_mensaje(traza["paciente_id"])}]
    for paso in traza["pasos"]:
        mensajes.append({"role": "assistant", "content": json.dumps(
            {"herramienta": paso["herramienta"], "argumentos": paso["argumentos"]},
            ensure_ascii=False, allow_nan=False)})
        if paso["herramienta"] != "registrar_decision":
            mensajes.append({"role": "user", "content":
                             f"Resultado de {paso['herramienta']}:\n{paso['resultado']}"})
    registro = {"version": 1, "id": traza["id"], "origen": origen,
                "referencias": [f"{referencia}#{traza['id']}"], "messages": mensajes}
    validar_registro(registro)
    return registro


def convertir_archivo(entrada, salida, *, origen, sistema):
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
                                             referencia=entrada.as_posix()))
        except (ValueError, TypeError, RecursionError):
            raise ValueError(f"línea {numero}: traza o contexto inválido") from None
    datos = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in registros).encode()
    # Temporal vecino: el enlace crea la salida completa de manera exclusiva, sin reemplazarla.
    with tempfile.NamedTemporaryFile(dir=salida.parent, prefix=".conversion-", suffix=".jsonl") as temporal:
        temporal.write(datos)
        temporal.flush()
        informe = validar_archivo(temporal.name)
        if informe["errores"]:
            numero = informe["errores"][0]["linea"]
            raise ValueError(f"línea {numero}: conversación inválida o id duplicado")
        os.link(temporal.name, salida)
    return {"registros": len(registros), "entrada_sha256": hashlib.sha256(contenido).hexdigest(),
            "salida_sha256": informe["sha256"]}
