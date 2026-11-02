"""Tokenización SFT sin modelo: únicamente las respuestas generan pérdida."""

import hashlib
import json
from pathlib import Path
import tempfile

from .conversaciones import _json, validar_archivo, validar_registro
from .preparacion import _canonical, _codigo, _sha, validar_configuracion


def tokenizar_conversacion(registro, tokenizer, max_seq_length):
    """Incluye respuesta y cierre del turno; excluye encabezados y todo contexto.

    Solo acepta plantillas/tokenizadores que preserven exactamente sus prefijos.
    No pide máscaras nativas (Gemma fijado no tiene bloques generation).
    """
    validar_registro(registro)
    if type(max_seq_length) is not int or max_seq_length <= 0:
        raise ValueError("max_seq_length requiere entero positivo")
    mensajes = registro["messages"]

    def tokens(historia, generar=False):
        salida = tokenizer.apply_chat_template(
            historia, tokenize=True, add_generation_prompt=generar,
            enable_thinking=False, padding=False, truncation=False, return_dict=True)
        ids = salida["input_ids"]
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("tokenizador devolvió IDs inválidos o anidados")
        if salida.get("attention_mask", [1] * len(ids)) != [1] * len(ids):
            raise ValueError("tokenización individual no debe contener padding")
        return ids

    ids = tokens(mensajes)
    if len(ids) > max_seq_length:
        raise ValueError("conversación excede max_seq_length; no se recorta")
    labels = [-100] * len(ids)
    for i, mensaje in enumerate(mensajes):
        if mensaje["role"] != "assistant":
            continue
        inicio = tokens(mensajes[:i], True)
        fin = tokens(mensajes[:i + 1])
        if ids[:len(inicio)] != inicio or ids[:len(fin)] != fin or len(fin) <= len(inicio):
            raise ValueError("plantilla/tokenizador no preserva prefijos; máscara insegura")
        labels[len(inicio):len(fin)] = ids[len(inicio):len(fin)]
    if all(x == -100 for x in labels):
        raise ValueError("conversación sin tokens supervisados")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def tokenizar_preparacion(preparacion, salida, tokenizer=None):
    """Verifica snapshots y escribe nuevo artefacto; no modifica la preparación.

    tokenizer inyectable para tests; producción carga solo AutoTokenizer, nunca pesos.
    """
    origen, destino = Path(preparacion), Path(salida)
    if destino.exists() or destino.is_symlink():
        raise ValueError("la salida ya existe")
    if not destino.parent.is_dir():
        raise ValueError("la carpeta padre de salida debe existir")
    try:
        manifiesto_bytes = (origen / "manifiesto.json").read_bytes()
        manifiesto = _json(manifiesto_bytes.decode("utf-8"))
        datos = (origen / "conversaciones.jsonl").read_bytes()
        config_bytes = (origen / "configuracion.json").read_bytes()
        config = _json(config_bytes.decode("utf-8"))
        hashes = {"conversaciones.jsonl": _sha(datos), "configuracion.json": _sha(config_bytes)}
        if (manifiesto["version"] != 1 or manifiesto["estado"] != "preparada_no_entrenada"
                or manifiesto["archivos"] != hashes or manifiesto["configuracion"] != config):
            raise ValueError("snapshots no coinciden con manifiesto")
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise ValueError("preparación inválida o snapshots alterados") from None
    validar_configuracion(config)
    codigo = _codigo()
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["modelo"], revision=config["revision"])
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizador sin plantilla de chat única")
    # Se valida la misma copia exacta que se tokeniza; no se relee el origen.
    with tempfile.TemporaryDirectory(prefix=".tokens-", dir=destino.parent) as temporal:
        carpeta = Path(temporal)
        snapshot = carpeta / "conversaciones.jsonl"
        snapshot.write_bytes(datos)
        if validar_archivo(snapshot)["errores"]:
            raise ValueError("dataset inválido")
        ejemplos, supervisados = [], 0
        for linea in datos.decode("utf-8").splitlines():
            if not linea.strip():
                continue
            registro = _json(linea)
            ejemplo = tokenizar_conversacion(registro, tokenizer, config["max_seq_length"])
            supervisados += sum(x != -100 for x in ejemplo["labels"])
            ejemplos.append({"id": registro["id"], **ejemplo})
        contenido = ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ejemplos)).encode()
        resultado = {"version": 1, "estado": "tokenizada_no_entrenada", "codigo": codigo,
            "preparacion_sha256": _sha(manifiesto_bytes), "snapshots": hashes,
            "modelo": config["modelo"], "revision": config["revision"],
            "plantilla_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "mascara": "prefijos_verificados_assistant_con_cierre", "enable_thinking": False,
            "max_seq_length": config["max_seq_length"], "registros": len(ejemplos),
            "tokens": sum(len(x["input_ids"]) for x in ejemplos), "supervisados": supervisados,
            "archivos": {"tokens.jsonl": _sha(contenido)}}
        (carpeta / "tokens.jsonl").write_bytes(contenido)
        (carpeta / "manifiesto-tokens.json").write_bytes(_canonical(resultado))
        try:
            destino.mkdir()
        except FileExistsError:
            raise ValueError("la salida ya existe") from None
        try:
            for nombre in ("tokens.jsonl", "manifiesto-tokens.json"):
                (carpeta / nombre).rename(destino / nombre)
        except OSError:
            for archivo in destino.iterdir():
                archivo.unlink()
            destino.rmdir()
            raise
    return resultado
