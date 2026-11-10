import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from infra import kaggle_baseline as kb


class KaggleTest(unittest.TestCase):
    def test_entorno_t4_y_token_sin_imprimirse(self):
        entorno = kb.preparar_entorno({"A": "1"}, "float16", "auto", Path("/x/m.jsonl"), "hf_secreto")
        self.assertEqual({k: entorno[k] for k in ("A", "MODELO_DTYPE", "MODELO_DEVICE_MAP", "MODELO_METRICAS")},
                         {"A": "1", "MODELO_DTYPE": "float16", "MODELO_DEVICE_MAP": "auto",
                          "MODELO_METRICAS": "/x/m.jsonl"})
        self.assertEqual(entorno["HF_TOKEN"], "hf_secreto")
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            kb.preparar_entorno({}, "float16", "auto", Path("m"), None)

    def test_secreto_viene_de_kaggle_secrets(self):
        cliente = MagicMock()
        cliente.UserSecretsClient.return_value.get_secret.return_value = "tok"
        with patch.dict(sys.modules, {"kaggle_secrets": cliente}):
            self.assertEqual(kb.obtener_secreto("HF_TOKEN"), "tok")
        cliente.UserSecretsClient.return_value.get_secret.assert_called_once_with("HF_TOKEN")

    def test_no_pisa_salida_existente(self):
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertRaises(FileExistsError):
                kb.crear_salida(Path(carpeta))
            self.assertTrue(kb.crear_salida(Path(carpeta) / "nueva").is_dir())

    def test_insumos_sin_esperadas_y_cantidad_validada(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz, destino = Path(carpeta) / "repo", Path(carpeta) / "out"
            (raiz / "practica").mkdir(parents=True)
            (raiz / "normas/b").mkdir(parents=True)
            (raiz / "practica/casos.jsonl").write_text("1\n2\n3\n")
            for nombre in ("parametros.yaml", "NOR-1.md", "otro.txt"):
                (raiz / "normas/b" / nombre).write_text("x")
            destino.mkdir()
            casos, normas = kb.preparar_insumos("b", 2, destino, raiz)
            self.assertEqual(casos.read_text(), "1\n2\n")
            self.assertEqual(sorted(p.name for p in normas.iterdir()), ["NOR-1.md", "parametros.yaml"])
            for args in (("z", 1), ("b", 0), ("b", 51)):
                with self.assertRaises(ValueError):
                    kb.preparar_insumos(*args, destino, raiz)

    def test_correr_guarda_runner_sin_token_y_no_gasta_con_dtype_invalido(self):
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log", return_value=(3, "falló con ***")) as ejecutar, \
                patch.object(kb, "describir_maquina", return_value={"gpus": ["T4", "T4"]}):
            destino = Path(carpeta) / "corrida"
            codigo = kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            runner = json.loads((destino / "runner.json").read_text())
            comando, env, log, token = ejecutar.call_args.args
            with self.assertRaises(ValueError):
                kb.correr("original", 1, Path(carpeta) / "otra", "int8", "auto", "hf_secreto")
            self.assertFalse((Path(carpeta) / "otra").exists())
        self.assertEqual(codigo, 3)
        self.assertEqual((runner["estado"], runner["returncode"]), ("fallida", 3))
        self.assertEqual(runner["configuracion"]["dtype"], "float16")
        self.assertNotIn("hf_secreto", json.dumps(runner))
        self.assertEqual(env["MODELO_METRICAS"], str(destino / "metricas.jsonl"))
        self.assertEqual(comando[1:4], ["-m", "admision.correr", "jugar"])
        self.assertEqual((log, token), (destino / "runner.log", "hf_secreto"))

    def test_runner_registra_el_lote_del_entorno(self):
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.dict("os.environ", {"MODELO_LOTE": "4"}), \
                patch.object(kb, "ejecutar_con_log", return_value=(0, "")), \
                patch.object(kb, "describir_maquina", return_value={}):
            destino = Path(carpeta) / "corrida"
            kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            config = json.loads((destino / "runner.json").read_text())["configuracion"]
            self.assertEqual(os.environ.get("MODELO_METRICAS"), None)  # No contamina el proceso.
        self.assertEqual((config["lote"], config["dtype"], config["device_map"]), (4, "float16", "auto"))
        self.assertEqual(config["metricas"], str(destino / "metricas.jsonl"))

    def _buscador_registrado(self, variante: str | None) -> dict:
        with tempfile.TemporaryDirectory() as carpeta, patch.dict("os.environ"), \
                patch.object(kb, "ejecutar_con_log", return_value=(0, "")), \
                patch.object(kb, "describir_maquina", return_value={}):
            os.environ.pop("BUSCADOR_VARIANTE", None)
            if variante:
                os.environ["BUSCADOR_VARIANTE"] = variante
            destino = Path(carpeta) / "corrida"
            kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            return json.loads((destino / "runner.json").read_text())["buscador"]

    def test_runner_registra_la_variante_del_buscador(self):
        self.assertEqual(self._buscador_registrado("prefijos"),
                         {"variante": "prefijos", "sin_comunes": True, "prefijo": 4,
                          "umbral_relativo": 0.6, "k_entorno": 3})

    def test_runner_registra_baseline_si_no_hay_variante(self):
        self.assertEqual(self._buscador_registrado(None),
                         {"variante": "baseline", "sin_comunes": False, "prefijo": None,
                          "umbral_relativo": 0.0, "k_entorno": 2})

    def _prompt_registrado(self, prompt_spec: str | None) -> dict:
        with tempfile.TemporaryDirectory() as carpeta, patch.dict("os.environ"), \
                patch.object(kb, "ejecutar_con_log", return_value=(0, "")), \
                patch.object(kb, "describir_maquina", return_value={}):
            os.environ.pop("PROMPT_SPEC", None)
            if prompt_spec is not None:
                os.environ["PROMPT_SPEC"] = prompt_spec
            destino = Path(carpeta) / "corrida"
            kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            return json.loads((destino / "runner.json").read_text())["prompt"]

    def test_runner_registra_el_prompt_baseline_por_defecto(self):
        from admision_tp.prompts import PromptSpec, RUTA_BASELINE
        esperado = PromptSpec.cargar(RUTA_BASELINE)
        for valor in (None, ""):
            self.assertEqual(self._prompt_registrado(valor),
                             {"nombre": esperado.nombre, "version": esperado.version,
                              "sha256": esperado.sha256, "archivo": "configuraciones/prompts/baseline.json"})

    def test_runner_registra_el_prompt_de_prompt_spec(self):
        from admision_tp.prompts import PromptSpec, RUTA_BASELINE
        datos = json.loads(RUTA_BASELINE.read_text(encoding="utf-8"))
        datos["nombre"] = "candidato-prueba"
        datos["system_prompt"] = datos["system_prompt"] + " Respondé breve."
        datos.pop("sha256", None)
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "candidato.json"
            spec = PromptSpec(**{k: datos[k] for k in ("version", "nombre", "system_prompt", "mensaje_inicial",
                                                       "formato_observacion", "mensaje_json_invalido")})
            ruta.write_text(json.dumps({**datos, "sha256": spec.sha256}, ensure_ascii=False), encoding="utf-8")
            registrado = self._prompt_registrado(str(ruta))
        self.assertEqual(registrado["nombre"], "candidato-prueba")
        self.assertEqual(registrado["sha256"], spec.sha256)
        self.assertNotEqual(registrado["sha256"], PromptSpec.cargar(RUTA_BASELINE).sha256)

    def test_prompt_alterado_falla_antes_de_crear_la_salida(self):
        from admision_tp.prompts import RUTA_BASELINE
        datos = json.loads(RUTA_BASELINE.read_text(encoding="utf-8"))
        datos["system_prompt"] = datos["system_prompt"] + " (alterado sin recalcular el hash)"
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log") as ejecutar:
            ruta = Path(carpeta) / "alterado.json"
            ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
            destino = Path(carpeta) / "corrida"
            with patch.dict("os.environ", {"PROMPT_SPEC": str(ruta)}), self.assertRaises(ValueError):
                kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            self.assertFalse(destino.exists())
        ejecutar.assert_not_called()

    def test_variante_de_buscador_desconocida_falla_antes_de_crear_la_salida(self):
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.dict("os.environ", {"BUSCADOR_VARIANTE": "inexistente"}), \
                patch.object(kb, "ejecutar_con_log") as ejecutar:
            destino = Path(carpeta) / "corrida"
            with self.assertRaisesRegex(ValueError, "BUSCADOR_VARIANTE"):
                kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            self.assertFalse(destino.exists())
        ejecutar.assert_not_called()

    def test_error_atrapado_por_el_kit_marca_la_corrida_fallida(self):
        def jugar_con_error(comando, env, log, token):
            salida = Path(comando[comando.index("--salida") + 1])
            salida.mkdir()
            (salida / "tiempos.json").write_text(json.dumps({"error": "AcceleratorError: CUDA"}))
            return 0, ""
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log", side_effect=jugar_con_error), \
                patch.object(kb, "describir_maquina", return_value={}):
            destino = Path(carpeta) / "corrida"
            codigo = kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            runner = json.loads((destino / "runner.json").read_text())
        self.assertEqual(codigo, 1)
        self.assertEqual((runner["estado"], runner["returncode"]), ("fallida", 1))
        self.assertIn("CUDA", runner["error_agente"])

    def test_corrida_sin_error_del_kit_queda_terminada(self):
        def jugar_bien(comando, env, log, token):
            salida = Path(comando[comando.index("--salida") + 1])
            salida.mkdir()
            (salida / "tiempos.json").write_text(json.dumps({"error": None}))
            return 0, ""
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log", side_effect=jugar_bien), \
                patch.object(kb, "describir_maquina", return_value={}):
            destino = Path(carpeta) / "corrida"
            self.assertEqual(kb.correr("original", 1, destino, "float16", "auto", "x"), 0)
            runner = json.loads((destino / "runner.json").read_text())
        self.assertEqual((runner["estado"], runner["error_agente"]), ("terminada", None))

    def test_corrida_cortada_queda_en_curso(self):
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log", side_effect=KeyboardInterrupt), \
                patch.object(kb, "describir_maquina", return_value={}):
            destino = Path(carpeta) / "corrida"
            with self.assertRaises(KeyboardInterrupt):
                kb.correr("original", 1, destino, "float16", "auto", "hf_secreto")
            runner = json.loads((destino / "runner.json").read_text())
        self.assertEqual(runner["estado"], "en_curso")
        self.assertNotIn("returncode", runner)

    def test_log_incremental_tapa_el_token(self):
        import sys
        programa = "import sys; print('cargando hf_secreto'); print('error', file=sys.stderr); sys.exit(2)"
        with tempfile.TemporaryDirectory() as carpeta:
            log = Path(carpeta) / "runner.log"
            codigo, cola = kb.ejecutar_con_log([sys.executable, "-c", programa], {}, log, "hf_secreto")
            texto = log.read_text()
        self.assertEqual(codigo, 2)
        self.assertIn("cargando ***", texto)
        self.assertIn("error", texto)
        self.assertNotIn("hf_secreto", texto + cola)


if __name__ == "__main__":
    unittest.main()
