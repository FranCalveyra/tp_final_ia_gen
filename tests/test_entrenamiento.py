"""Orquestación con dobles: estos tests NO entrenan ningún modelo."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from admision_tp.entrenamiento import (cargar_entradas, entrenar, formar_lote,
                                      publicar_checkpoint, validar_reanudacion)
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

    def test_trainer_simulado_checkpoint_publicacion_y_cierre_fallido(self):
        """Recorre el backend real, reemplazando todas las bibliotecas pesadas."""
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            tokenizer = TokenizadorArtificial()
            tokenizer.pad_token_id = 0
            tokenizer.save_pretrained = lambda ruta: (ruta / "tokenizer.json").write_text("artificial")
            finales, logs, opciones = [], [], []
            run = SimpleNamespace(id="run", url="https://example.invalid/run", log=logs.append,
                                  finish=lambda **kw: finales.append(kw))
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
                    opciones.append(kw)
                def train(self, **kw):
                    if self.falla:
                        raise RuntimeError("simulado")
                    self.kw["data_collator"](self.kw["train_dataset"])
                    cp = Path(self.kw["args"].output_dir) / "checkpoint-50"
                    cp.mkdir()
                    for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                                   "adapter_config.json", "adapter_model.safetensors"):
                        (cp / nombre).write_text("simulado")
                    callback = self.kw["callbacks"][0]
                    callback.on_log(None, SimpleNamespace(global_step=50), None, logs={"loss": 1.0})
                    callback.on_save(None, SimpleNamespace(global_step=50), None)
                def save_model(self, ruta):
                    ruta = Path(ruta)
                    ruta.mkdir()
                    (ruta / "adapter_model.safetensors").write_text("simulado")
            carga = {}
            modulos = {
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                    bfloat16="bf16", long="long", tensor=lambda x, **kw: x),
                "transformers": SimpleNamespace(Trainer=Trainer, TrainerCallback=object,
                    TrainingArguments=lambda **kw: SimpleNamespace(**kw), set_seed=lambda x: None,
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: (carga.update(kw), modelo)[1])),
                "peft": SimpleNamespace(LoraConfig=lambda **kw: kw, get_peft_model=lambda m, c: m),
                "wandb": SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw),
                "huggingface_hub": SimpleNamespace(HfApi=API),
            }
            with patch.dict("sys.modules", modulos):
                commits = []
                estado = entrenar(p, t, raiz / "exito", online=True, publicar=True,
                                  persistir=lambda: commits.append("commit"))
                self.assertEqual(len(estado["publicaciones"]), 2)
                self.assertNotIn("attn_implementation", carga)  # Forzar SDPA rompe el audio de Gemma 4.
                self.assertEqual(len(commits), 6)  # Entradas, cp+recibo, final+recibo, cierre.
                args = opciones[0]["args"]
                self.assertFalse(args.save_only_model)
                self.assertFalse(args.push_to_hub)
                identidad = cargar_entradas(p, t)[4]
                validar_reanudacion(raiz / "exito", raiz / "exito/checkpoint-50", identidad)
                Trainer.falla = True
                with self.assertRaises(RuntimeError):
                    entrenar(p, t, raiz / "fallo", online=True)
            self.assertEqual(finales, [{"exit_code": 0}, {"exit_code": 1}])


if __name__ == "__main__":
    unittest.main()
