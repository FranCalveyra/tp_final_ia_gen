"""SFT LoRA explícito: importar no carga pesos, entrena ni publica."""

import json
from pathlib import Path
import re
import uuid

from .conversaciones import _json, validar_registro
from .preparacion import _canonical, _codigo, _sha, validar_configuracion

LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"], "task_type": "CAUSAL_LM"}
POLITICA = {"save_steps": 50, "bf16": True, "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False}, "save_only_model": False}


def cargar_entradas(preparacion, tokenizacion):
    """Valida snapshots y estructura antes de importar bibliotecas de entrenamiento."""
    p, t = Path(preparacion), Path(tokenizacion)
    archivos = {n: (p / n).read_bytes() for n in
                ("conversaciones.jsonl", "configuracion.json", "manifiesto.json")}
    archivos.update({n: (t / n).read_bytes() for n in ("tokens.jsonl", "manifiesto-tokens.json")})
    config = _json(archivos["configuracion.json"].decode())
    validar_configuracion(config)
    preparado = _json(archivos["manifiesto.json"].decode())
    tokens = _json(archivos["manifiesto-tokens.json"].decode())
    hashes = {n: _sha(archivos[n]) for n in ("conversaciones.jsonl", "configuracion.json")}
    if (preparado.get("estado") != "preparada_no_entrenada" or preparado.get("version") != 1
            or preparado.get("archivos") != hashes or preparado.get("configuracion") != config
            or tokens.get("version") != 1 or tokens.get("estado") != "tokenizada_no_entrenada"
            or tokens.get("preparacion_sha256") != _sha(archivos["manifiesto.json"])
            or tokens.get("snapshots") != hashes
            or tokens.get("archivos") != {"tokens.jsonl": _sha(archivos["tokens.jsonl"])}
            or (tokens.get("modelo"), tokens.get("revision"), tokens.get("max_seq_length")) !=
               (config["modelo"], config["revision"], config["max_seq_length"])
            or tokens.get("mascara") != "prefijos_verificados_assistant_con_cierre"
            or tokens.get("enable_thinking") is not False):
        raise ValueError("snapshots o manifiestos incompatibles")
    registros = [_json(x) for x in archivos["conversaciones.jsonl"].decode().splitlines() if x.strip()]
    for registro in registros:
        validar_registro(registro)
    ejemplos = [_json(x) for x in archivos["tokens.jsonl"].decode().splitlines() if x.strip()]
    if not ejemplos or [x.get("id") for x in ejemplos] != [x["id"] for x in registros]:
        raise ValueError("orden o IDs tokenizados incompatibles")
    for x in ejemplos:
        if set(x) != {"id", "input_ids", "attention_mask", "labels"}:
            raise ValueError("campos de tokens inválidos")
        ids, mask, labels = x["input_ids"], x["attention_mask"], x["labels"]
        if (not all(isinstance(v, list) for v in (ids, mask, labels))
                or not 0 < len(ids) <= config["max_seq_length"]
                or len(ids) != len(labels) or mask != [1] * len(ids)
                or any(type(i) is not int or i < 0 for i in ids)
                or any(type(v) is not int or v not in (-100, i) for i, v in zip(ids, labels))
                or all(v == -100 for v in labels)):
            raise ValueError("IDs, atención o etiquetas inválidos")
    if (tokens.get("registros") != len(ejemplos)
            or tokens.get("tokens") != sum(len(x["input_ids"]) for x in ejemplos)
            or tokens.get("supervisados") != sum(v != -100 for x in ejemplos for v in x["labels"])):
        raise ValueError("conteos tokenizados incompatibles")
    identidad = {"archivos": {n: _sha(b) for n, b in archivos.items()},
                 "configuracion": config, "lora": LORA, "politica": POLITICA,
                 "uv_lock_sha256": _sha((Path(__file__).resolve().parents[1] / "uv.lock").read_bytes())}
    return config, registros, ejemplos, tokens, identidad, archivos


def formar_lote(ejemplos, pad_token_id):
    """Padding a derecha; nunca transforma contexto o padding en objetivos."""
    largo = max(len(x["input_ids"]) for x in ejemplos)
    return {campo: [x[campo] + [relleno] * (largo - len(x[campo])) for x in ejemplos]
            for campo, relleno in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def validar_reanudacion(salida, checkpoint, identidad):
    raiz, checkpoint = Path(salida).resolve(), Path(checkpoint).resolve()
    if checkpoint.parent != raiz or not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        raise ValueError("checkpoint debe pertenecer a la misma corrida")
    auditoria = _json((raiz / "corrida.json").read_text())
    if (auditoria["identidad"] != identidad
            or _json((checkpoint / "corrida.json").read_text()) != auditoria):
        raise ValueError("checkpoint de datos/configuración distintos")
    requeridos = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                  "adapter_config.json", "adapter_model.safetensors")
    if not all((checkpoint / n).is_file() for n in requeridos):
        raise ValueError("checkpoint incompleto: faltan pesos o estado de entrenamiento")
    hashes = _json((checkpoint / "integridad.json").read_text())
    if any(hashes.get(n) != _sha((checkpoint / n).read_bytes()) for n in requeridos):
        raise ValueError("checkpoint alterado")
    return auditoria


def publicar_checkpoint(carpeta, nombre, api, repo, run):
    """La privacidad se comprueba antes de cada upload; no cambia visibilidad."""
    if api.repo_info(repo_id=repo, repo_type="model").private is not True:
        raise ValueError("el repositorio HF debe existir y ser privado")
    commit = api.upload_folder(repo_id=repo, folder_path=str(carpeta), path_in_repo=nombre,
                              commit_message=f"checkpoint {nombre}", ignore_patterns=["publicacion-hf.json"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit.oid):
        raise ValueError("HF no devolvió una revisión inmutable")
    dato = {"checkpoint": nombre, "hf_repo": repo, "hf_revision": commit.oid}
    # Conservar el recibo aunque el envío posterior a W&B falle.
    (Path(carpeta) / "publicacion-hf.json").write_bytes(_canonical(dato))
    run.log(dato)
    return dato


def entrenar(preparacion, tokenizacion, salida, *, checkpoint=None, online=False,
             publicar=False, backend=None, codigo=None, persistir=None):
    """Ejecuta solo por llamada explícita. backend inyectable para tests sin entrenar."""
    config, registros, ejemplos, tokens, identidad, archivos = cargar_entradas(preparacion, tokenizacion)
    destino = Path(salida)
    codigo = _codigo() if codigo is None else codigo
    if (not isinstance(codigo, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(codigo.get("commit")))
            or type(codigo.get("dirty")) is not bool):
        raise ValueError("procedencia de código inválida")
    if publicar and not online:
        raise ValueError("publicar requiere registro online explícito")
    if checkpoint:
        auditoria = validar_reanudacion(destino, checkpoint, identidad)
        if auditoria["online"] != online or auditoria["publicar"] != publicar:
            raise ValueError("no cambiar modo al reanudar")
        if auditoria["codigo"] != codigo:
            raise ValueError("no cambiar código al reanudar")
    else:
        if destino.exists() or destino.is_symlink():
            raise ValueError("salida existente; usá reanudación explícita")
        auditoria = {"version": 1, "run_id": uuid.uuid4().hex[:16], "identidad": identidad,
                     "codigo": codigo, "online": online, "publicar": publicar}
    # Backend comprueba CUDA y plantilla/máscaras reales antes de cargar pesos.
    ejecutar = backend or _ejecutar_cuda
    destino.mkdir(exist_ok=bool(checkpoint), parents=False)
    (destino / "corrida.json").write_bytes(_canonical(auditoria))
    entradas = destino / "entradas"
    entradas.mkdir(exist_ok=True)
    for n, contenido in archivos.items():
        (entradas / n).write_bytes(contenido)
    try:
        if persistir:
            persistir()
        argumentos = (config, registros, ejemplos, tokens, destino, auditoria, checkpoint)
        resultado = (ejecutar(*argumentos) if backend else
                     ejecutar(*argumentos, persistir=persistir))
        estado = {"estado": "finalizada", **resultado}
        (destino / "estado.json").write_bytes(_canonical(estado))
        if persistir:
            persistir()
    except Exception as error:
        # No persistir mensajes de terceros que pudieran contener credenciales.
        (destino / "estado.json").write_bytes(_canonical({"estado": "fallida", "error_tipo": type(error).__name__}))
        if persistir:
            try:
                persistir()
            except Exception:
                pass  # No reemplazar el error original por otro de persistencia.
        raise
    return estado


def _ejecutar_cuda(config, registros, ejemplos, tokens, destino, auditoria, checkpoint, persistir=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("entrenamiento requiere CUDA; no se usa CPU automáticamente")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model
    from .tokenizacion import tokenizar_conversacion
    import wandb

    set_seed(config["semilla"])
    tokenizer = AutoTokenizer.from_pretrained(config["modelo"], revision=config["revision"])
    if _sha(tokenizer.chat_template.encode()) != tokens["plantilla_sha256"]:
        raise ValueError("plantilla distinta de la tokenización")
    for registro, ejemplo in zip(registros, ejemplos, strict=True):
        if tokenizar_conversacion(registro, tokenizer, config["max_seq_length"]) != {
                k: ejemplo[k] for k in ("input_ids", "attention_mask", "labels")}:
            raise ValueError("tokens o máscaras no reproducibles")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizador sin token de padding")
    run = wandb.init(entity=config["wandb_entity"], project=config["wandb_project"],
        id=auditoria["run_id"], resume="must" if checkpoint and auditoria["online"] else None,
        mode="online" if auditoria["online"] else "offline", dir=str(destino),
        config=auditoria, save_code=False, settings=wandb.Settings(disable_git=True))
    exitoso = False
    publicaciones = (_json((destino / "publicaciones.json").read_text())
                     if (destino / "publicaciones.json").exists() else [])
    try:
        if auditoria["publicar"]:
            from huggingface_hub import HfApi
            api = HfApi()
            if api.repo_info(repo_id=config["hf_repo"], repo_type="model").private is not True:
                raise ValueError("repositorio HF no privado")
        modelo = AutoModelForCausalLM.from_pretrained(config["modelo"], revision=config["revision"],
                                                     dtype=torch.bfloat16)
        # Sin attn_implementation: forzar "sdpa" rompe la torre de audio de Gemma 4 (ver modelo.py).
        modelo.config.use_cache = False
        modelo = get_peft_model(modelo, LoraConfig(**LORA, revision=config["revision"]))
        modelo.enable_input_require_grads()

        class Auditor(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                run.log({**(logs or {}), "trainer_step": state.global_step})

            def guardar(self, carpeta, nombre):
                (carpeta / "corrida.json").write_bytes(_canonical(auditoria))
                hashes = {p.name: _sha(p.read_bytes()) for p in carpeta.iterdir()
                          if p.is_file() and p.name != "integridad.json"}
                (carpeta / "integridad.json").write_bytes(_canonical(hashes))
                if persistir:
                    persistir()
                if auditoria["publicar"]:
                    publicaciones.append(publicar_checkpoint(carpeta, nombre, api, config["hf_repo"], run))
                    (destino / "publicaciones.json").write_bytes(_canonical(publicaciones))
                    if persistir:
                        persistir()

            def on_save(self, args, state, control, **kwargs):
                self.guardar(destino / f"checkpoint-{state.global_step}", f"checkpoint-{state.global_step}")

        callback = Auditor()
        args = TrainingArguments(output_dir=str(destino), seed=config["semilla"], data_seed=config["semilla"],
            learning_rate=config["learning_rate"], num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            **POLITICA, save_strategy="steps", logging_steps=1,
            report_to="none", push_to_hub=False, remove_unused_columns=False)
        def collator(filas):
            return {k: torch.tensor(v, dtype=torch.long) for k, v in formar_lote(filas, tokenizer.pad_token_id).items()}
        trainer = Trainer(model=modelo, args=args, train_dataset=ejemplos,
                          data_collator=collator, processing_class=tokenizer, callbacks=[callback])
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(destino / "final"))
        tokenizer.save_pretrained(destino / "final")
        callback.guardar(destino / "final", "final")
        exitoso = True
        return {"run_id": run.id, "wandb_url": run.url, "publicaciones": publicaciones}
    finally:
        run.finish(exit_code=0 if exitoso else 1)
