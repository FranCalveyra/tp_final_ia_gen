"""Orquestación con dobles: estos tests NO entrenan ningún modelo."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from admision_tp.entrenamiento import (EJECUCION, POLITICA, cargar_entradas, entrenar, formar_lote,
                                      posiciones_supervisadas, publicar_checkpoint, validar_reanudacion)
from admision_tp.preparacion import _canonical, _sha, preparar_corrida
from admision_tp.tokenizacion import tokenizar_preparacion
from test_conversaciones import registro
from test_tokenizacion import TokenizadorArtificial


class EntrenamientoTests(unittest.TestCase):
    def entradas(self, raiz):
        datos, config = raiz / "datos.jsonl", raiz / "config.json"
        datos.write_text(json.dumps(registro()) + "\n")
        config.write_bytes(Path("configuraciones/sft-ejemplo.json").read_bytes())
        preparar_corrida(datos, config, raiz / "preparada")
        tokenizar_preparacion(raiz / "preparada", raiz / "tokens", TokenizadorArtificial())
        return raiz / "preparada", raiz / "tokens"

    def test_padding_preserva_mascaras(self):
        filas = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                 {"input_ids": [3], "attention_mask": [1], "labels": [3]}]
        antes = copy.deepcopy(filas)
        lote = formar_lote(filas, 0)
        self.assertEqual(lote, {"input_ids": [[1, 2], [3, 0]],
                               "attention_mask": [[1, 1], [1, 0]], "labels": [[-100, 2], [3, -100]]})
        self.assertEqual(filas, antes)

    def test_exito_copia_entradas_no_sobrescribe(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            vistos = []
            def backend(config, registros, ejemplos, tokens, destino, auditoria, checkpoint):
                vistos.append(auditoria)
                self.assertFalse(auditoria["online"])
                self.assertFalse(auditoria["publicar"])
                self.assertEqual(config["revision"], tokens["revision"])
                self.assertEqual(registros[0]["id"], ejemplos[0]["id"])
                return {"run_id": auditoria["run_id"]}
            salida = raiz / "salida"
            resultado = entrenar(p, t, salida, backend=backend)
            self.assertEqual(resultado["estado"], "finalizada")
            self.assertEqual((salida / "entradas/tokens.jsonl").read_bytes(), (t / "tokens.jsonl").read_bytes())
            self.assertEqual(len(vistos), 1)
            with self.assertRaises(ValueError):
                entrenar(p, t, salida, backend=backend)
            with self.assertRaises(ValueError):
                entrenar(p, t, raiz / "otra", publicar=True, backend=backend)

    def test_fallo_registrado_sin_exponer_mensaje(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            def falla(*args):
                raise RuntimeError("mensaje sensible del proveedor")
            with self.assertRaises(RuntimeError):
                entrenar(p, t, raiz / "salida", backend=falla)
            estado = json.loads((raiz / "salida/estado.json").read_text())
            self.assertEqual(estado, {"estado": "fallida", "error_tipo": "RuntimeError"})

    def test_tokens_alterados_fallan_antes_de_backend(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            (t / "tokens.jsonl").write_text("alterado")
            with self.assertRaises(ValueError):
                entrenar(p, t, raiz / "no-creada", backend=lambda *a: self.fail("backend invocado"))
            self.assertFalse((raiz / "no-creada").exists())

    def test_reanuda_identidad_y_estado_completo(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            salida = raiz / "salida"
            entrenar(p, t, salida, backend=lambda *a: {})
            auditoria = json.loads((salida / "corrida.json").read_text())
            cp = salida / "checkpoint-50"
            cp.mkdir()
            for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                           "adapter_config.json", "adapter_model.safetensors"):
                (cp / nombre).write_bytes(b"estado artificial no ejecutable")
            (cp / "corrida.json").write_bytes(_canonical(auditoria))
            hashes = {f.name: _sha(f.read_bytes()) for f in cp.iterdir()}
            (cp / "integridad.json").write_bytes(_canonical(hashes))
            identidad = cargar_entradas(p, t)[4]
            self.assertEqual(validar_reanudacion(salida, cp, identidad), auditoria)
            def reanudar(*args):
                self.assertEqual(args[-1], cp)
                self.assertEqual(args[-2]["run_id"], auditoria["run_id"])
                return {}
            entrenar(p, t, salida, checkpoint=cp, backend=reanudar)
            otra = copy.deepcopy(identidad)
            otra["lora"]["r"] = 16
            with self.assertRaises(ValueError):
                validar_reanudacion(salida, cp, otra)
            (cp / "optimizer.pt").write_bytes(b"alterado")
            with self.assertRaises(ValueError):
                validar_reanudacion(salida, cp, identidad)

    def test_publicacion_privada_revision_y_fallo(self):
        class API:
            private = False
            subidas = 0
            def repo_info(self, **kwargs):
                return SimpleNamespace(private=self.private)
            def upload_folder(self, **kwargs):
                self.subidas += 1
                return SimpleNamespace(oid="a" * 40)
        api, logs = API(), []
        run = SimpleNamespace(log=logs.append)
        with self.assertRaises(ValueError):
            publicar_checkpoint(Path("cp"), "checkpoint-50", api, "u/repo", run)
        self.assertEqual(api.subidas, 0)
        api.private = True
        with tempfile.TemporaryDirectory() as temporal:
            dato = publicar_checkpoint(Path(temporal), "checkpoint-50", api, "u/repo", run)
            def fallo_log(dato):
                raise RuntimeError("fallo W&B")
            with self.assertRaises(RuntimeError):
                publicar_checkpoint(Path(temporal), "checkpoint-50", api, "u/repo",
                                    SimpleNamespace(log=fallo_log))
            self.assertEqual(json.loads((Path(temporal) / "publicacion-hf.json").read_text()), dato)
        self.assertEqual(dato["hf_revision"], "a" * 40)
        self.assertEqual(logs, [dato])

    def dobles(self):
        """Reemplaza torch/transformers/peft/wandb/HF; devuelve los módulos y lo que observan."""
        tokenizer = TokenizadorArtificial()
        tokenizer.pad_token_id = 0
        tokenizer.save_pretrained = lambda ruta: (ruta / "tokenizer.json").write_text("artificial")
        obs = SimpleNamespace(finales=[], logs=[], opciones=[], carga={})
        run = SimpleNamespace(id="run", url="https://example.invalid/run", log=obs.logs.append,
                              finish=lambda **kw: obs.finales.append(kw))
        modelo = SimpleNamespace(config=SimpleNamespace(), enable_input_require_grads=lambda: None)
        class API:
            def repo_info(self, **kw):
                return SimpleNamespace(private=True)
            def upload_folder(self, **kw):
                return SimpleNamespace(oid="b" * 40)
        class Trainer:
            falla = False
            def __init__(self, **kw):
                self.kw = kw
                obs.opciones.append(kw)
            def train(self, **kw):
                if self.falla:
                    raise RuntimeError("simulado")
                self.kw["data_collator"](self.kw["train_dataset"])
                cp = Path(self.kw["args"].output_dir) / "checkpoint-50"
                cp.mkdir()
                for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                               "adapter_config.json", "adapter_model.safetensors", "scaler.pt"):
                    (cp / nombre).write_text("simulado")
                callback = self.kw["callbacks"][0]
                callback.on_log(None, SimpleNamespace(global_step=50), None, logs={"loss": 1.0})
                callback.on_save(None, SimpleNamespace(global_step=50), None)
            def save_model(self, ruta):
                ruta = Path(ruta)
                ruta.mkdir()
                (ruta / "adapter_model.safetensors").write_text("simulado")
        obs.Trainer = Trainer
        modulos = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                    max_memory_allocated=lambda i: (9.5e9, 7.25e9)[i], max_memory_reserved=lambda i: 1.2e10),
                bfloat16="bf16", float16="fp16", long="long", tensor=lambda x, **kw: x),
            "transformers": SimpleNamespace(Trainer=Trainer, TrainerCallback=object,
                TrainingArguments=lambda **kw: SimpleNamespace(**kw), set_seed=lambda x: None,
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
                AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=SimpleNamespace(
                    from_pretrained=lambda *a, **kw: (obs.carga.update(kw), modelo)[1])),
            "peft": SimpleNamespace(LoraConfig=lambda **kw: kw, get_peft_model=lambda m, c: m),
            "wandb": SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw),
            "huggingface_hub": SimpleNamespace(HfApi=API),
        }
        return modulos, obs

    def test_trainer_simulado_checkpoint_publicacion_y_cierre_fallido(self):
        """Recorre el backend real, reemplazando todas las bibliotecas pesadas."""
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            modulos, obs = self.dobles()
            with patch.dict("sys.modules", modulos):
                commits = []
                estado = entrenar(p, t, raiz / "exito", online=True, publicar=True,
                                  persistir=lambda: commits.append("commit"))
                self.assertEqual(len(estado["publicaciones"]), 2)
                # VRAM para el #5: sumada sobre GPU, en estado.json y en cada log de W&B.
                vram = {"vram_pico_gb": 16.75, "vram_reservada_gb": 24.0, "vram_pico_por_gpu_gb": [9.5, 7.25]}
                self.assertEqual({k: estado[k] for k in vram}, vram)
                guardado = json.loads((raiz / "exito/estado.json").read_text())
                self.assertEqual(guardado["vram_pico_gb"], 16.75)
                self.assertEqual(obs.logs[0]["vram_pico_gb"], 16.75)
                self.assertNotIn("attn_implementation", obs.carga)  # Forzar SDPA rompe el audio de Gemma 4.
                self.assertEqual(len(commits), 6)  # Entradas, cp+recibo, final+recibo, cierre.
                args = obs.opciones[0]["args"]
                self.assertFalse(args.save_only_model)
                self.assertFalse(args.push_to_hub)
                # Defaults de la entrega (A100): bf16, una GPU, sin device_map, checkpoint cada 50.
                self.assertEqual((obs.carga["dtype"], "device_map" in obs.carga), ("bf16", False))
                self.assertEqual((args.bf16, hasattr(args, "fp16"), args.save_steps), (True, False, 50))
                identidad = cargar_entradas(p, t)[4]
                self.assertEqual(identidad["politica"], POLITICA)
                validar_reanudacion(raiz / "exito", raiz / "exito/checkpoint-50", identidad)
                auditoria = json.loads((raiz / "exito/corrida.json").read_text())
                self.assertEqual(auditoria["ejecucion"], EJECUCION)
                obs.Trainer.falla = True
                with self.assertRaises(RuntimeError):
                    entrenar(p, t, raiz / "fallo", online=True)
            self.assertEqual(obs.finales, [{"exit_code": 0}, {"exit_code": 1}])

    def test_fp16_y_reparto_automatico_llegan_a_modelo_y_trainer(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            modulos, obs = self.dobles()
            with patch.dict("sys.modules", modulos):
                entrenar(p, t, raiz / "t4", online=True, precision="fp16", dispositivos="auto",
                         guardar_cada=2)
            args = obs.opciones[0]["args"]
            self.assertEqual((obs.carga["dtype"], obs.carga["device_map"]), ("fp16", "auto"))
            self.assertEqual((args.fp16, args.bf16, args.save_steps), (True, False, 2))
            self.assertTrue(args.gradient_checkpointing)
            self.assertFalse(args.save_only_model)  # Optimizador y scaler van en cada checkpoint.
            ejecucion = {"precision": "fp16", "dispositivos": "auto", "guardar_cada": 2}
            auditoria = json.loads((raiz / "t4/corrida.json").read_text())
            self.assertEqual(auditoria["ejecucion"], ejecucion)
            self.assertEqual(auditoria["identidad"]["politica"]["fp16"], True)
            self.assertNotEqual(auditoria["identidad"], cargar_entradas(p, t)[4])  # No es la identidad A100.

    def test_reanudacion_exige_misma_precision_dispositivos_y_scaler(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            modulos, obs = self.dobles()
            with patch.dict("sys.modules", modulos):
                entrenar(p, t, raiz / "t4", online=True, precision="fp16", dispositivos="auto")
            salida, cp = raiz / "t4", raiz / "t4/checkpoint-50"
            ejecucion = {"precision": "fp16", "dispositivos": "auto", "guardar_cada": 50}
            identidad = cargar_entradas(p, t, ejecucion)[4]
            validar_reanudacion(salida, cp, identidad)
            for cambio in ({"precision": "bf16"}, {"dispositivos": "unico"}, {"guardar_cada": 10}):
                with self.subTest(cambio=cambio), self.assertRaises(ValueError):
                    entrenar(p, t, salida, checkpoint=cp, online=True, backend=lambda *a: {},
                             **{"precision": "fp16", "dispositivos": "auto", **cambio})
            # fp16 exige scaler.pt en el checkpoint, con hash íntegro.
            (cp / "scaler.pt").unlink()
            with self.assertRaises(ValueError):
                validar_reanudacion(salida, cp, identidad)

    def test_valores_de_ejecucion_invalidos_fallan_antes_de_crear_salida(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            for kw in ({"precision": "fp8"}, {"dispositivos": "cuda:1"}, {"guardar_cada": 0}):
                with self.subTest(kw=kw), self.assertRaises(ValueError):
                    entrenar(p, t, raiz / "no", backend=lambda *a: self.fail("backend"), **kw)
            self.assertFalse((raiz / "no").exists())

    def test_auditoria_anterior_sin_ejecucion_se_lee_como_default_a100(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            salida = raiz / "vieja"
            entrenar(p, t, salida, backend=lambda *a: {})
            auditoria = json.loads((salida / "corrida.json").read_text())
            del auditoria["ejecucion"]
            (salida / "corrida.json").write_bytes(_canonical(auditoria))
            cp = salida / "checkpoint-50"
            cp.mkdir()
            for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                           "adapter_config.json", "adapter_model.safetensors"):
                (cp / nombre).write_bytes(b"x")
            (cp / "corrida.json").write_bytes(_canonical(auditoria))
            (cp / "integridad.json").write_bytes(_canonical({f.name: _sha(f.read_bytes()) for f in cp.iterdir()}))
            entrenar(p, t, salida, checkpoint=cp, backend=lambda *a: {})


if __name__ == "__main__":
    unittest.main()


class PosicionesSupervisadasTest(unittest.TestCase):
    """La pérdida solo proyecta al vocabulario las posiciones que predicen un token del asistente."""

    def test_cada_posicion_predice_el_token_siguiente(self):
        # Contexto (-100), respuesta [5, 6], cierre fuera de la máscara y padding.
        posiciones, objetivos = posiciones_supervisadas([[-100, -100, 5, 6, -100, -100]])
        self.assertEqual(posiciones, [1, 2])
        self.assertEqual(objetivos, [[5, 6]])

    def test_lote_usa_la_union_y_marca_lo_ajeno_como_ignorado(self):
        posiciones, objetivos = posiciones_supervisadas([[-100, 7, -100, 8], [-100, -100, 9, -100]])
        self.assertEqual(posiciones, [0, 1, 2])
        self.assertEqual(objetivos, [[7, -100, 8], [-100, 9, -100]])

    def test_ultima_posicion_nunca_se_conserva(self):
        # La última posición no tiene token siguiente aunque su etiqueta esté supervisada.
        posiciones, objetivos = posiciones_supervisadas([[-100, 3, 4]])
        self.assertEqual(posiciones, [0, 1])
        self.assertEqual(objetivos, [[3, 4]])

    def test_posiciones_son_enteros_de_python(self):
        # accelerate mueve los tensores del forward a la primera GPU (Kaggle v31); una lista no se mueve.
        posiciones, _ = posiciones_supervisadas([[-100, 3, 4]])
        self.assertIs(type(posiciones), list)
        self.assertTrue(all(type(p) is int for p in posiciones))


class ObjetivoLoraTest(unittest.TestCase):
    def test_solo_atencion_del_modelo_de_lenguaje(self):
        import re
        from admision_tp.entrenamiento import LORA
        patron = LORA["target_modules"]
        for nombre in ("model.language_model.layers.0.self_attn.q_proj",
                       "model.language_model.layers.41.self_attn.v_proj"):
            self.assertTrue(re.fullmatch(patron, nombre), nombre)
        for nombre in ("model.vision_tower.encoder.layers.0.self_attn.q_proj",
                       "model.audio_tower.layers.3.self_attn.v_proj",
                       "model.language_model.layers.0.self_attn.k_proj",
                       "model.language_model.layers.0.self_attn.q_proj.linear"):
            self.assertIsNone(re.fullmatch(patron, nombre), nombre)


class AdapterInicialTest(unittest.TestCase):
    """Sin `adapter_inicial`, LoRA nuevo; con él, sigue entrenando el adapter publicado."""

    def peft(self, base=("google/gemma-4-E4B-it", "e" * 40)):
        llamadas = []
        config_hf = SimpleNamespace(base_model_name_or_path=base[0], revision=base[1])
        peft = SimpleNamespace(
            LoraConfig=lambda **kw: kw,
            get_peft_model=lambda m, c: llamadas.append(("nuevo", c)) or "nuevo",
            PeftConfig=SimpleNamespace(from_pretrained=lambda repo, **kw: config_hf),
            PeftModel=SimpleNamespace(from_pretrained=lambda m, repo, **kw: llamadas.append(("inicial", repo, kw))
                                      or "inicial"))
        return peft, llamadas

    def test_sin_adapter_inicial_crea_lora_nuevo(self):
        from admision_tp.entrenamiento import LORA, aplicar_lora
        peft, llamadas = self.peft()
        self.assertEqual(aplicar_lora(peft, "modelo", {"modelo": "google/gemma-4-E4B-it", "revision": "e" * 40}),
                         "nuevo")
        self.assertEqual(llamadas, [("nuevo", {**LORA, "revision": "e" * 40})])

    def test_con_adapter_inicial_lo_carga_entrenable_en_su_revision(self):
        from admision_tp.entrenamiento import aplicar_lora
        peft, llamadas = self.peft()
        config = {"modelo": "google/gemma-4-E4B-it", "revision": "e" * 40,
                  "adapter_inicial": {"repo": "FranCalveyra/tp-admision", "revision": "d" * 40, "subcarpeta": "final"}}
        self.assertEqual(aplicar_lora(peft, "modelo", config), "inicial")
        (_, repo, kw), = llamadas
        self.assertEqual(repo, "FranCalveyra/tp-admision")
        self.assertEqual((kw["revision"], kw["subfolder"], kw["is_trainable"]), ("d" * 40, "final", True))

    def test_adapter_inicial_de_otra_base_falla(self):
        from admision_tp.entrenamiento import aplicar_lora
        peft, llamadas = self.peft(base=("google/gemma-4-E4B-it", "f" * 40))
        config = {"modelo": "google/gemma-4-E4B-it", "revision": "e" * 40,
                  "adapter_inicial": {"repo": "FranCalveyra/tp-admision", "revision": "d" * 40, "subcarpeta": "final"}}
        with self.assertRaisesRegex(ValueError, "otra base"):
            aplicar_lora(peft, "modelo", config)
        self.assertEqual(llamadas, [])
