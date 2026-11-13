"""Contrato propuesto v1 para demostraciones curadas; no evalúa su corrección."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from admision.entorno import CAMPOS, MAX_LLAMADAS, VALORES

ARGUMENTOS = {
    "leer_triage": {"paciente_id"},
    "preguntar": {"paciente_id", "a_quien", "tema"},
    "verificar_afiliacion": {"entidad", "numero_afiliado"},
    "buscar_normativa": {"consulta"},
    "registrar_decision": set(CAMPOS),
}
# `referencia`: las trazas con las que la cátedra entrenó su modelo (entrenamiento/trazas.jsonl).
ORIGENES = ("historial", "referencia", "sintetico")
FORMATO_OBSERVACION = "Resultado de {herramienta}:\n{observacion}"


def _json(texto):
    def objeto(pares):
        resultado = {}
        for clave, valor in pares:
            if clave in resultado:
                raise ValueError("clave JSON duplicada")
            resultado[clave] = valor
        return resultado
    return json.loads(texto, object_pairs_hook=objeto,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constante JSON inválida")))


def _cadena(valor):
    return isinstance(valor, str) and bool(valor.strip())


def _sin_prefijo(texto):
    """Separa el prefijo de hechos que antecede al JSON de la decisión final: texto sin llaves que termina en salto de línea."""
    corte = texto.find("{")
    prefijo = texto[:corte] if corte >= 0 else ""
    if not prefijo.strip() or not prefijo.endswith("\n") or "}" in prefijo:
        raise ValueError("prefijo inválido")
    return texto[corte:]


def _accion(texto, admite_prefijo=False):
    """Acción del mensaje. `admite_prefijo` permite texto antes del JSON, solo si la acción es `registrar_decision`."""
    if admite_prefijo and not texto.startswith("{"):
        accion = _json(_sin_prefijo(texto))
        if not isinstance(accion, dict) or accion.get("herramienta") != "registrar_decision":
            raise ValueError("solo registrar_decision admite prefijo")
        texto = texto[texto.find("{"):]
    return _accion_estricta(texto)


def _accion_estricta(texto):
    accion = _json(texto)
    if not isinstance(accion, dict) or set(accion) != {"herramienta", "argumentos"}:
        raise ValueError("acción requiere herramienta y argumentos")
    nombre, args = accion["herramienta"], accion["argumentos"]
    if not isinstance(nombre, str) or nombre not in ARGUMENTOS or not isinstance(args, dict):
        raise ValueError("herramienta o argumentos inválidos")
    requeridos = ARGUMENTOS[nombre]
    permitidos = requeridos | ({"fundamentos"} if nombre == "registrar_decision" else set())
    if not requeridos <= args.keys() or not args.keys() <= permitidos:
        raise ValueError("campos de argumentos faltantes o desconocidos")
    if nombre != "registrar_decision":
        if not all(_cadena(v) for v in args.values()):
            raise ValueError("argumentos requieren texto no vacío")
        if nombre == "preguntar" and (args["a_quien"] not in {"paciente", "acompanante"}
                                              or args["tema"] not in {"cobertura", "art"}):
            raise ValueError("destinatario o tema inválido")
    else:
        for campo, valores in VALORES.items():
            if not isinstance(args[campo], str) or args[campo] not in valores:
                raise ValueError("enumeración de ficha inválida")
        if any(type(args[k]) is not bool for k in ("box_antes_del_registro", "aviso_al_voluntariado")):
            raise ValueError("la ficha requiere booleanos JSON")
        if any(args[k] is not None and not _cadena(args[k])
               for k in ("numero_documento", "entidad_a_notificar")):
            raise ValueError("documento y entidad requieren texto o null")
        plazo = args["plazo_circuito_min"]
        if plazo is not None and (type(plazo) is not int or plazo < 0):
            raise ValueError("plazo requiere entero no negativo o null")
        fundamentos = args.get("fundamentos", [])
        if not isinstance(fundamentos, list) or not all(_cadena(x) for x in fundamentos):
            raise ValueError("fundamentos requiere lista de textos")
    return nombre


def _partes_observacion(formato, herramienta):
    """Prefijo y sufijo que el formato del prompt pone alrededor de la observación."""
    if not isinstance(formato, str) or formato.count("{observacion}") != 1:
        raise ValueError("formato de observación inválido")
    antes, despues = formato.split("{observacion}")
    return antes.format(herramienta=herramienta), despues.format(herramienta=herramienta)


def validar_registro(registro, formato_observacion=FORMATO_OBSERVACION):
    """Valida estructura sin modificar contenido; ValueError no incluye datos sensibles.

    `formato_observacion` es el del prompt usado (`PromptSpec.formato_observacion`).
    """
    campos = {"version", "id", "origen", "referencias", "messages"}
    if not isinstance(registro, dict) or set(registro) != campos:
        raise ValueError("campos del registro faltantes o desconocidos")
    if type(registro["version"]) is not int or registro["version"] != 1:
        raise ValueError("versión no soportada")
    if not _cadena(registro["id"]) or registro["origen"] not in ORIGENES:
        raise ValueError("id u origen inválido")
    refs = registro["referencias"]
    if not isinstance(refs, list) or not refs or not all(_cadena(x) for x in refs):
        raise ValueError("referencias requiere lista no vacía de textos")
    # Control explícito de procedencia, no detector universal de contaminación.
    if any(re.search(r"(?:^|[/\\:_-])(practica|test)(?:$|[/\\:_-])", x.casefold())
           for x in [registro["id"], *refs]):
        raise ValueError("procedencia de práctica/test no admitida")
    mensajes = registro["messages"]
    if not isinstance(mensajes, list) or len(mensajes) < 3 or len(mensajes) % 2 != 1:
        raise ValueError("conversación incompleta")
    if (len(mensajes) - 1) // 2 > MAX_LLAMADAS:
        raise ValueError("excede 40 llamadas")
    anterior = None
    for i, mensaje in enumerate(mensajes):
        rol = "system" if i == 0 else ("user" if i % 2 else "assistant")
        if (not isinstance(mensaje, dict) or set(mensaje) != {"role", "content"}
                or mensaje["role"] != rol or not _cadena(mensaje["content"])):
            raise ValueError(f"mensaje {i}: rol, campos o contenido inválidos")
        if rol == "assistant":
            try:
                anterior = _accion(mensaje["content"], admite_prefijo=i == len(mensajes) - 1)
            except (ValueError, TypeError) as e:
                raise ValueError(f"mensaje {i}: acción JSON inválida") from e
            if (anterior == "registrar_decision") != (i == len(mensajes) - 1):
                raise ValueError("registrar_decision debe ser la última acción")
        elif i > 1:
            prefijo, sufijo = _partes_observacion(formato_observacion, anterior)
            contenido = mensaje["content"]
            cuerpo = contenido[len(prefijo):len(contenido) - len(sufijo)]
            if (not contenido.startswith(prefijo) or not contenido.endswith(sufijo)
                    or len(contenido) < len(prefijo) + len(sufijo) or not cuerpo.strip()):
                raise ValueError(f"mensaje {i}: observación no corresponde a herramienta")


def validar_archivo(ruta, formato_observacion=FORMATO_OBSERVACION):
    """Informe determinista de formato y hash exacto; no carga modelo ni reescribe datos."""
    contenido = Path(ruta).read_bytes()
    informe = {"sha256": hashlib.sha256(contenido).hexdigest(), "registros": 0,
               "validos": 0, "por_origen": {}, "errores": []}
    vistos, origenes = set(), Counter()
    try:
        lineas = contenido.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        informe["errores"].append({"linea": 0, "error": "archivo no es UTF-8"})
        return informe
    for numero, linea in enumerate(lineas, 1):
        if not linea.strip():
            informe["errores"].append({"linea": numero, "error": "línea vacía"})
            continue
        informe["registros"] += 1
        try:
            registro = _json(linea)
            validar_registro(registro, formato_observacion)
            if registro["id"] in vistos:
                raise ValueError("id duplicado")
            vistos.add(registro["id"])
            origenes[registro["origen"]] += 1
            informe["validos"] += 1
        except (ValueError, TypeError, RecursionError) as error:
            # Solo errores propios: el parser nunca expone partes del documento.
            detalle = (str(error) if type(error) is ValueError
                       else "JSON o tipos inválidos; revisar contrato v1")
            informe["errores"].append({"linea": numero, "error": detalle})
    if not informe["registros"]:
        informe["errores"].append({"linea": 0, "error": "archivo sin registros"})
    informe["por_origen"] = dict(sorted(origenes.items()))
    return informe
