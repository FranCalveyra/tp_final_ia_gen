"""Runner de entrenamiento en Kaggle con dobles: no entrena, no descarga ni sube nada."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from admision_tp.conversaciones import validar_archivo
from admision_tp.entrenamiento import EJECUCION, entrenar
from admision_tp.preparacion import _canonical, _sha, preparar_corrida
from admision_tp.tokenizacion import tokenizar_preparacion
from infra import kaggle_entrenamiento as ke
from scripts import datos_humo
from test_tokenizacion import TokenizadorArtificial

HF, WANDB = "hf_secreto_escritura", "wandb_secreto"
T4 = {"precision": "fp16", "dispositivos": "auto", "guardar_cada": 2}


class KaggleEntrenamientoTest(unittest.TestCase):
    def entradas(self, raiz, **kw):
        datos_humo.generar(raiz / "humo", **kw)
        preparar_corrida(raiz / "humo/conversaciones.jsonl", raiz / "humo/configuracion.json", raiz / "preparada")
        tokenizar_preparacion(raiz / "preparada", raiz / "tokens", TokenizadorArtificial())
        return raiz / "preparada", raiz / "tokens"

    def correr(self, raiz, destino="corrida", hf=HF, wandb=WANDB, **kw):
        p, t = self.entradas(raiz) if not (raiz / "preparada").exists() else (raiz / "preparada", raiz / "tokens")
        with patch.object(ke.kb, "describir_maquina", return_value={"commit": "a" * 40, "sucio": False}), \
                patch.object(ke.kb, "ejecutar_con_log", return_value=(0, "")) as ejecutar:
            codigo = ke.correr(p, t, raiz / destino, T4, hf, wandb, **kw)
        return codigo, ejecutar

    def test_datos_de_humo_son_sinteticos_validos_y_no_pisan(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz, registros=3)
            self.assertEqual(validar_archivo(p / "conversaciones.jsonl")["por_origen"], {"sintetico": 3})
            config = json.loads((p / "configuracion.json").read_text())
            self.assertEqual((config["batch_size"], config["gradient_accumulation_steps"]), (1, 1))
            self.assertEqual(config["hf_repo"], "FranCalveyra/tp-admision")
            texto = (p / "conversaciones.jsonl").read_text().lower()
            self.assertNotIn("practica", texto)
            with self.assertRaises(FileExistsError):
                datos_humo.generar(raiz / "humo")

    def test_secretos_viajan_por_entorno_y_nunca_por_argv_ni_evidencia(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            codigo, ejecutar = self.correr(raiz)
            comando, entorno, log, secretos = ejecutar.call_args.args
            runner = (raiz / "corrida/runner.json").read_text()
        self.assertEqual(codigo, 0)
        self.assertEqual((entorno["HF_TOKEN"], entorno["WANDB_API_KEY"]), (HF, WANDB))
        self.assertEqual(secretos, [HF, WANDB])
        self.assertFalse(any(HF in c or WANDB in c for c in comando))
        self.assertNotIn(HF, runner)
        self.assertNotIn(WANDB, runner)
        self.assertEqual(log, raiz / "corrida/runner.log")

    def test_log_tapa_ambos_secretos(self):
        programa = f"print('a {HF} b {WANDB}')"
        with tempfile.TemporaryDirectory() as temporal:
            log = Path(temporal) / "runner.log"
            ke.kb.ejecutar_con_log([sys.executable, "-c", programa], {}, log, [HF, WANDB])
            texto = log.read_text()
        self.assertEqual(texto.strip(), "a *** b ***")

    def test_publicacion_forzada_por_defecto_y_comando_reusa_entrenar(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            _, ejecutar = self.correr(raiz)
            comando = ejecutar.call_args.args[0]
            runner = json.loads((raiz / "corrida/runner.json").read_text())
        self.assertEqual(comando[1:3], ["-m", "scripts.entrenar"])
        for bandera in ("--online", "--publicar"):
            self.assertIn(bandera, comando)
        self.assertEqual(comando[comando.index("--precision") + 1], "fp16")
        self.assertEqual(comando[comando.index("--dispositivos") + 1], "auto")
        self.assertEqual(comando[comando.index("--guardar-cada") + 1], "2")
        self.assertEqual(comando[comando.index("--salida") + 1], str(raiz / "corrida/entrenamiento"))
        self.assertTrue(runner["publicar"])
        self.assertEqual((runner["estado"], runner["returncode"], runner["wandb"]), ("terminada", 0, "online"))
        self.assertEqual((runner["hf_repo"], runner["ejecucion"], runner["commit"]),
                         ("FranCalveyra/tp-admision", T4, "a" * 40))

    def test_runner_copia_la_vram_del_estado_del_entrenamiento(self):
        def entrenar_con_estado(comando, env, log, secretos):
            salida = Path(comando[comando.index("--salida") + 1])
            salida.mkdir(parents=True, exist_ok=True)
            (salida / "estado.json").write_text(json.dumps(
                {"estado": "finalizada", "vram_pico_gb": 18.2, "vram_reservada_gb": 22.0,
                 "vram_pico_por_gpu_gb": [10.1, 8.1], "run_id": "r"}))
            return 0, ""
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            with patch.object(ke.kb, "describir_maquina", return_value={"commit": "a" * 40, "sucio": False}), \
                    patch.object(ke.kb, "ejecutar_con_log", side_effect=entrenar_con_estado):
                self.assertEqual(ke.correr(p, t, raiz / "corrida", T4, HF, WANDB), 0)
            runner = json.loads((raiz / "corrida/runner.json").read_text())
        self.assertEqual(runner["vram"], {"vram_pico_gb": 18.2, "vram_reservada_gb": 22.0,
                                          "vram_pico_por_gpu_gb": [10.1, 8.1]})

    def test_main_publica_salvo_pedido_explicito(self):
        with patch.object(ke, "secreto", side_effect=lambda n: {"HF_WRITE_TOKEN": HF, "WANDB_API_KEY": WANDB}[n]), \
                patch.object(ke, "correr", return_value=0) as correr:
            ke.main(["--preparacion", "p", "--tokens", "t", "--salida", "s"])
            self.assertTrue(correr.call_args.args[6])
            self.assertEqual(correr.call_args.args[3], {"precision": "fp16", "dispositivos": "auto", "guardar_cada": 50})
            ke.main(["--preparacion", "p", "--tokens", "t", "--salida", "s", "--sin-publicar"])
            self.assertFalse(correr.call_args.args[6])

    def test_falla_temprano_sin_token_o_sin_wandb_y_no_crea_salida(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            for hf, wandb, mensaje in ((None, WANDB, "HF_WRITE_TOKEN"), ("", WANDB, "HF_WRITE_TOKEN"),
                                       (HF, None, "WANDB_API_KEY")):
                with self.subTest(hf=hf, wandb=wandb), self.assertRaisesRegex(RuntimeError, mensaje):
                    self.correr(raiz, "no", hf=hf, wandb=wandb)
            self.assertFalse((raiz / "no").exists())

    def test_sin_publicar_y_sin_wandb_queda_offline_en_runner_json(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            _, ejecutar = self.correr(raiz, hf=None, wandb=None, publicar=False)
            comando, entorno, _, _ = ejecutar.call_args.args
            runner = json.loads((raiz / "corrida/runner.json").read_text())
        self.assertNotIn("--online", comando)
        self.assertNotIn("--publicar", comando)
        self.assertNotIn("HF_TOKEN", entorno)
        self.assertEqual(runner["wandb"], "offline (sin WANDB_API_KEY)")

    def test_no_pisa_salida_existente(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            (raiz / "corrida").mkdir()
            with self.assertRaises(FileExistsError):
                self.correr(raiz)
            self.assertEqual(list((raiz / "corrida").iterdir()), [])

    def test_error_del_entrenamiento_marca_fallida_y_corte_deja_en_curso(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            with patch.object(ke.kb, "describir_maquina", return_value={}), \
                    patch.object(ke.kb, "ejecutar_con_log", return_value=(1, "boom")):
                self.assertEqual(ke.correr(p, t, raiz / "mala", T4, HF, WANDB), 1)
            with patch.object(ke.kb, "describir_maquina", return_value={}), \
                    patch.object(ke.kb, "ejecutar_con_log", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    ke.correr(p, t, raiz / "cortada", T4, HF, WANDB)
            mala = json.loads((raiz / "mala/runner.json").read_text())
            cortada = json.loads((raiz / "cortada/runner.json").read_text())
        self.assertEqual((mala["estado"], mala["returncode"], mala["log_final"]), ("fallida", 1, "boom"))
        self.assertEqual(cortada["estado"], "en_curso")

    def test_secreto_prefiere_entorno_luego_kaggle_y_none_si_falta(self):
        cliente = MagicMock()
        cliente.UserSecretsClient.return_value.get_secret.return_value = "del_secret"
        with patch.dict(sys.modules, {"kaggle_secrets": cliente}), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(ke.secreto("HF_WRITE_TOKEN"), "del_secret")
            with patch.dict("os.environ", {"HF_WRITE_TOKEN": "del_entorno"}):
                self.assertEqual(ke.secreto("HF_WRITE_TOKEN"), "del_entorno")
            cliente.UserSecretsClient.return_value.get_secret.side_effect = RuntimeError(f"no existe {HF}")
            self.assertIsNone(ke.secreto("WANDB_API_KEY"))

    def publicar_falso(self, raiz, p, t):
        """Corre `entrenar` con dobles hasta tener un checkpoint íntegro y lo deja como si viniera de HF."""
        origen = raiz / "origen"
        entrenar(p, t, origen, online=True, publicar=True, precision="fp16", dispositivos="auto",
                 guardar_cada=2, codigo={"commit": "a" * 40, "dirty": False}, backend=lambda *a: {})
        cp = origen / "checkpoint-2"
        cp.mkdir()
        for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                       "adapter_config.json", "adapter_model.safetensors", "scaler.pt"):
            (cp / nombre).write_bytes(b"estado")
        (cp / "corrida.json").write_bytes((origen / "corrida.json").read_bytes())
        (cp / "integridad.json").write_bytes(_canonical({f.name: _sha(f.read_bytes()) for f in cp.iterdir()}))
        return cp

    def test_reanudar_descarga_checkpoint_fijado_y_valida_integridad(self):
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            cp = self.publicar_falso(raiz, p, t)
            llamadas = []
            def descargar(**kw):
                llamadas.append(kw)
                destino = Path(kw["local_dir"]) / "checkpoint-2"
                destino.mkdir(parents=True)
                for f in cp.iterdir():
                    (destino / f.name).write_bytes(f.read_bytes())
            identidad = ke.cargar_entradas(p, t, T4)[4]
            salida = raiz / "nueva/entrenamiento"
            ruta = ke.descargar_checkpoint("FranCalveyra/tp-admision", revision, "checkpoint-2", salida,
                                           HF, identidad, descargar=descargar)
            self.assertEqual(ruta, salida / "checkpoint-2")
            self.assertTrue((salida / "corrida.json").is_file())
            self.assertEqual(llamadas, [{"repo_id": "FranCalveyra/tp-admision", "repo_type": "model",
                "revision": revision, "allow_patterns": ["checkpoint-2/*"], "local_dir": str(salida),
                "token": HF}])
            # Un checkpoint alterado tras la descarga no pasa la validación.
            def descargar_alterado(**kw):
                descargar(**kw)
                (Path(kw["local_dir"]) / "checkpoint-2/optimizer.pt").write_bytes(b"alterado")
            with self.assertRaises(ValueError):
                ke.descargar_checkpoint("FranCalveyra/tp-admision", revision, "checkpoint-2",
                                        raiz / "otra/entrenamiento", HF, identidad, descargar=descargar_alterado)

    def test_reanudar_en_correr_pasa_flags_y_exige_revision_inmutable(self):
        revision = "d" * 40
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            p, t = self.entradas(raiz)
            with patch.object(ke, "descargar_checkpoint", return_value=raiz / "x/checkpoint-2") as bajar:
                _, ejecutar = self.correr(raiz, reanudar=("checkpoint-2", revision))
                comando = ejecutar.call_args.args[0]
                self.assertEqual(comando[comando.index("--reanudar") + 1], str(raiz / "x/checkpoint-2"))
                self.assertEqual(bajar.call_args.args[:4],
                                 ("FranCalveyra/tp-admision", revision, "checkpoint-2", raiz / "corrida/entrenamiento"))
                runner = json.loads((raiz / "corrida/runner.json").read_text())
                self.assertEqual(runner["modo"], "reanudacion")
                self.assertEqual(runner["reanudacion"]["revision"], revision)
                for malo in (("checkpoint-2", "main"), ("../x", revision), ("checkpoint-2", "d" * 39)):
                    with self.subTest(malo=malo), self.assertRaises(ValueError):
                        self.correr(raiz, f"malo{malo[0][-1]}", reanudar=malo)
                    self.assertFalse((raiz / f"malo{malo[0][-1]}").exists())

    def test_ejecucion_default_del_runner_no_es_la_de_a100(self):
        self.assertEqual(EJECUCION, {"precision": "bf16", "dispositivos": "unico", "guardar_cada": 50})


if __name__ == "__main__":
    unittest.main()
