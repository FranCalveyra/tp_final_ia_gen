import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from admision_tp.configuracion import Configuracion
from admision_tp.modelo import ModeloLocal


class ModeloTest(unittest.TestCase):
    def setUp(self):
        self.torch = patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)})
        self.torch.start()
        self.addCleanup(self.torch.stop)
        self.procesador = MagicMock()
        self.modelo = MagicMock(device="cuda:0")
        self.entradas = MagicMock()
        self.entradas.__getitem__.return_value.shape = (1, 10)
        self.entradas.to.return_value = {"input_ids": "ids", "attention_mask": "mask"}
        self.procesador.apply_chat_template.return_value = self.entradas
        self.procesador.batch_decode.side_effect = [["primera"], ["segunda"]]
        self.adaptador = ModeloLocal(self.modelo, self.procesador, Configuracion())

    def test_vacio_no_genera(self):
        self.assertEqual(self.adaptador([]), [])
        self.modelo.generate.assert_not_called()

    def test_micro_lotes_preservan_orden_y_no_mutan_conversaciones(self):
        conversaciones = [[{"role": "user", "content": "uno"}], [{"role": "user", "content": "dos"}]]
        self.assertEqual(self.adaptador(conversaciones), ["primera", "segunda"])
        self.assertEqual(len(conversaciones[0]), 1)
        self.assertEqual(self.modelo.generate.call_count, 2)
        kwargs = self.procesador.apply_chat_template.call_args.kwargs
        self.assertFalse(kwargs["truncation"])
        self.assertFalse(kwargs["enable_thinking"])
        self.assertTrue(kwargs["padding"])
        self.assertFalse(self.modelo.generate.call_args.kwargs["do_sample"])
        self.modelo.generate.return_value.__getitem__.assert_called_with((slice(None), slice(10, None)))

    def test_lote_real_devuelve_dos_respuestas(self):
        self.adaptador.configuracion = Configuracion(lote=2)
        self.procesador.batch_decode.side_effect = [["uno", "dos"]]
        self.assertEqual(self.adaptador([[{}], [{}]]), ["uno", "dos"])
        self.modelo.generate.assert_called_once()

    def test_contexto_excesivo_falla_sin_generar(self):
        self.entradas.__getitem__.return_value.shape = (1, 8192)
        with self.assertRaisesRegex(ValueError, "Contexto excedido"):
            self.adaptador([[{}]])
        self.modelo.generate.assert_not_called()

    def test_cantidad_incorrecta_falla(self):
        self.procesador.batch_decode.side_effect = [[]]
        with self.assertRaises(RuntimeError):
            self.adaptador([[{}]])

    def test_revision_y_limites_validos(self):
        for kwargs in ({"revision": "main"}, {"lote": 0}):
            with self.assertRaises(ValueError):
                Configuracion(**kwargs)

    def test_carga_sin_cuda_falla_antes_de_descargar(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}):
            with self.assertRaisesRegex(RuntimeError, "GPU CUDA"):
                ModeloLocal.cargar(Configuracion())

    def test_carga_usa_revision_padding_y_bf16(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        procesador, modelo = MagicMock(), MagicMock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(
                AutoTokenizer=procesador, AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=modelo)}):
            adaptador = ModeloLocal.cargar(Configuracion())
        self.assertIs(adaptador.modelo, modelo.from_pretrained.return_value)
        self.assertEqual(procesador.from_pretrained.call_args.kwargs["padding_side"], "left")
        self.assertEqual(modelo.from_pretrained.call_args.kwargs["revision"], Configuracion.revision)
        self.assertEqual(modelo.from_pretrained.call_args.kwargs["dtype"], "bf16")
        self.assertNotIn("attn_implementation", modelo.from_pretrained.call_args.kwargs)
        modelo.from_pretrained.return_value.eval.assert_called_once()

    def _cargar_con_adapter(self, base=None, revision_base=None):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        transformers = SimpleNamespace(AutoTokenizer=MagicMock(), AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=MagicMock())
        peft = SimpleNamespace(PeftConfig=MagicMock(), PeftModel=MagicMock())
        peft.PeftConfig.from_pretrained.return_value = SimpleNamespace(
            base_model_name_or_path=base or Configuracion.modelo,
            revision=revision_base or Configuracion.revision)
        config = Configuracion(adapter_repo="grupo/lora", adapter_revision="a" * 40)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": peft}):
            return ModeloLocal.cargar(config), transformers, peft

    def test_carga_adapter_fijado_y_lo_fusiona(self):
        adaptador, transformers, peft = self._cargar_con_adapter()
        base = transformers.AutoModelForCausalLM.from_pretrained.return_value
        origen = {"revision": "a" * 40, "subfolder": "final"}
        peft.PeftConfig.from_pretrained.assert_called_once_with("grupo/lora", **origen)
        args, kwargs = peft.PeftModel.from_pretrained.call_args
        self.assertEqual(args, (base, "grupo/lora"))
        self.assertIs(kwargs["config"], peft.PeftConfig.from_pretrained.return_value)
        self.assertEqual({k: kwargs[k] for k in origen}, origen)
        fusionado = peft.PeftModel.from_pretrained.return_value.merge_and_unload.return_value
        self.assertIs(adaptador.modelo, fusionado)
        fusionado.eval.assert_called_once()

    def test_adapter_de_otra_base_falla_antes_de_aplicarse(self):
        for kwargs in ({"base": "otro/modelo"}, {"revision_base": "b" * 40}):
            with self.assertRaisesRegex(ValueError, "otra base"):
                self._cargar_con_adapter(**kwargs)

    def test_sin_adapter_no_importa_peft(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        with patch.dict(sys.modules, {"torch": torch, "peft": None, "transformers": SimpleNamespace(
                AutoTokenizer=MagicMock(), AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=MagicMock())}):
            ModeloLocal.cargar(Configuracion())

    def test_adapter_exige_revision_inmutable(self):
        for kwargs in ({"adapter_repo": "grupo/lora"}, {"adapter_repo": "grupo/lora", "adapter_revision": "main"},
                       {"adapter_revision": "a" * 40}):
            with self.assertRaises(ValueError):
                Configuracion(**kwargs)

    def test_entorno_configura_adapter(self):
        entorno = {"ADAPTER_REPO": "grupo/lora", "ADAPTER_REVISION": "c" * 40, "ADAPTER_SUBCARPETA": "checkpoint-50"}
        with patch.dict("os.environ", entorno):
            config = Configuracion.desde_entorno()
        self.assertEqual((config.adapter_repo, config.adapter_revision, config.adapter_subcarpeta),
                         ("grupo/lora", "c" * 40, "checkpoint-50"))

    def test_entorno_sin_adapter_usa_el_de_la_entrega(self):
        with patch.dict("os.environ", {}, clear=True):
            config = Configuracion.desde_entorno()
        self.assertEqual((config.adapter_repo, config.adapter_revision, config.adapter_subcarpeta),
                         ("FranCalveyra/tp-admision", "dc41674a8a2561a724884be2170ff6808efa6e96", "final"))

    def test_entorno_sin_lote_usa_el_de_los_numeros_reportados(self):
        # Todas las evaluaciones reportadas (evaluaciones/) corrieron con lote 4.
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(Configuracion.desde_entorno().lote, 4)
        with patch.dict("os.environ", {"MODELO_LOTE": "1"}, clear=True):
            self.assertEqual(Configuracion.desde_entorno().lote, 1)

    def test_revision_de_adapter_sin_repo_no_pisa_el_de_la_entrega(self):
        # Sin ADAPTER_REPO, una ADAPTER_REVISION suelta cambiaría en silencio la revisión del adapter de la entrega.
        for variable in ("ADAPTER_REVISION", "ADAPTER_SUBCARPETA"):
            with self.subTest(variable=variable), patch.dict("os.environ", {variable: "a" * 40}, clear=True), \
                    self.assertRaisesRegex(ValueError, "ADAPTER_REPO"):
                Configuracion.desde_entorno()

    def test_referencia_de_adapter_validada_en_un_solo_lugar(self):
        from admision_tp.configuracion import validar_adapter
        bueno = {"repo": "FranCalveyra/tp-admision", "revision": "a" * 40, "subcarpeta": "final"}
        self.assertEqual(validar_adapter(bueno), bueno)
        for malo in ({**bueno, "revision": "main"}, {**bueno, "repo": "../x"}, {**bueno, "subcarpeta": ""},
                     {**bueno, "subcarpeta": "../x"}, {**bueno, "extra": 1}, {"repo": "x/y"}, "x/y", None):
            with self.subTest(adapter=malo), self.assertRaises(ValueError):
                validar_adapter(malo)

    def test_entorno_con_adapter_vacio_usa_la_base_sola(self):
        with patch.dict("os.environ", {"ADAPTER_REPO": ""}, clear=True):
            config = Configuracion.desde_entorno()
        self.assertEqual((config.adapter_repo, config.adapter_revision), (None, None))

    def test_sin_ruta_de_metricas_no_mide_ni_toca_cuda(self):
        self.adaptador([[{}]])  # El torch doble no tiene `cuda`: fallaría si se midiera.

    def test_metricas_registran_una_linea_por_lote(self):
        import json
        import tempfile
        from pathlib import Path

        cuda = MagicMock()
        cuda.max_memory_allocated.return_value = 12.5e9
        cuda.max_memory_reserved.return_value = 14e9
        cuda.device_count.return_value = 1
        torch = SimpleNamespace(inference_mode=nullcontext, cuda=cuda)
        self.entradas.to.return_value = {"input_ids": "ids", "attention_mask": MagicMock()}
        self.entradas.to.return_value["attention_mask"].sum.return_value.item.return_value = 9
        generados = MagicMock(shape=(1, 4))
        generados.__ne__.return_value.sum.return_value.item.return_value = 3
        self.modelo.generate.return_value.__getitem__.return_value = generados
        self.procesador.pad_token_id = 0
        with tempfile.TemporaryDirectory() as carpeta, patch.dict(sys.modules, {"torch": torch}):
            ruta = Path(carpeta) / "metricas.jsonl"
            self.adaptador.configuracion = Configuracion(metricas=str(ruta))
            self.adaptador([[{}], [{}]])
            lineas = [json.loads(l) for l in ruta.read_text().splitlines()]
        self.assertEqual(len(lineas), 2)
        self.assertEqual(cuda.reset_peak_memory_stats.call_count, 2)
        generados.__ne__.assert_called_with(0)
        dato = lineas[0]
        self.assertEqual({k: dato[k] for k in ("conversaciones", "ancho_entrada", "tokens_entrada",
                                              "tokens_generados", "vram_pico_gb", "vram_reservada_gb")},
                         {"conversaciones": 1, "ancho_entrada": 10, "tokens_entrada": 9,
                          "tokens_generados": 3, "vram_pico_gb": 12.5, "vram_reservada_gb": 14.0})
        self.assertGreaterEqual(dato["segundos"], 0)

    def test_metricas_suman_vram_de_todas_las_gpu(self):
        import json
        import tempfile
        from pathlib import Path

        cuda = MagicMock()
        cuda.device_count.return_value = 2
        cuda.max_memory_allocated.side_effect = lambda g: (6e9, 4e9)[g]
        cuda.max_memory_reserved.side_effect = lambda g: (7e9, 5e9)[g]
        torch = SimpleNamespace(inference_mode=nullcontext, cuda=cuda)
        self.procesador.pad_token_id = 0
        self.procesador.batch_decode.side_effect = [["x"]]
        self.entradas.to.return_value = {"input_ids": "ids", "attention_mask": MagicMock()}
        self.entradas.to.return_value["attention_mask"].sum.return_value.item.return_value = 9
        generados = MagicMock(shape=(1, 4))
        generados.__ne__.return_value.sum.return_value.item.return_value = 3
        self.modelo.generate.return_value.__getitem__.return_value = generados
        with tempfile.TemporaryDirectory() as carpeta, patch.dict(sys.modules, {"torch": torch}):
            ruta = Path(carpeta) / "m.jsonl"
            self.adaptador.configuracion = Configuracion(metricas=str(ruta))
            self.adaptador([[{}]])
            dato = json.loads(ruta.read_text())
        self.assertEqual((dato["vram_pico_gb"], dato["vram_reservada_gb"]), (10.0, 12.0))
        self.assertEqual(dato["vram_pico_por_gpu_gb"], [6.0, 4.0])
        self.assertEqual([c.args for c in cuda.reset_peak_memory_stats.call_args_list], [(0,), (1,)])

    def test_defaults_a100_no_cambian(self):
        config = Configuracion()
        self.assertEqual((config.dtype, config.device_map), ("bfloat16", "cuda:0"))
        # Adapter y lote de la entrega fijados aparte (ver los tests de entorno): acá, el resto de los defaults.
        with patch.dict("os.environ", {"ADAPTER_REPO": "", "MODELO_LOTE": "1"}, clear=True):
            self.assertEqual(Configuracion.desde_entorno(), config)

    def test_dtype_y_device_map_validados(self):
        for kwargs in ({"dtype": "int8"}, {"dtype": "bf16"}, {"device_map": "cpu"}, {"device_map": "cuda:x"}):
            with self.assertRaises(ValueError):
                Configuracion(**kwargs)
        for mapa in ("auto", "cuda", "cuda:1"):
            Configuracion(device_map=mapa)

    def test_entorno_configura_dtype_y_device_map(self):
        with patch.dict("os.environ", {"MODELO_DTYPE": "float16", "MODELO_DEVICE_MAP": "auto"}):
            config = Configuracion.desde_entorno()
        self.assertEqual((config.dtype, config.device_map), ("float16", "auto"))

    def test_carga_propaga_dtype_y_device_map(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16", float16="fp16")
        modelo = MagicMock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(
                AutoTokenizer=MagicMock(), AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=modelo)}):
            ModeloLocal.cargar(Configuracion(dtype="float16", device_map="auto"))
        kwargs = modelo.from_pretrained.call_args.kwargs
        self.assertEqual((kwargs["dtype"], kwargs["device_map"]), ("fp16", "auto"))

    def test_carga_default_usa_cuda0(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        modelo = MagicMock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(
                AutoTokenizer=MagicMock(), AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(vision_config={}, audio_config={})), AutoModelForCausalLM=modelo)}):
            ModeloLocal.cargar(Configuracion())
        self.assertEqual(modelo.from_pretrained.call_args.kwargs["device_map"], "cuda:0")

    def test_entorno_activa_metricas(self):
        with patch.dict("os.environ", {"MODELO_METRICAS": "/tmp/m.jsonl"}):
            self.assertEqual(Configuracion.desde_entorno().metricas, "/tmp/m.jsonl")
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(Configuracion.desde_entorno().metricas)

    def test_factory_carga_una_vez_y_reutiliza_coordinador(self):
        import entrega
        from admision_tp.agente_prompt import AgentePrompt
        with patch.object(ModeloLocal, "cargar", return_value=self.adaptador) as cargar:
            agente = entrega.construir_agente()
        self.assertIsInstance(agente, AgentePrompt)
        self.assertIs(agente.modelo_lote, self.adaptador)
        cargar.assert_called_once()

    def test_medir_contexto_usa_el_template_de_la_generacion_sin_tocar_torch(self):
        self.procesador.apply_chat_template.return_value = {"input_ids": [[1] * 300]}
        conversacion = [{"role": "user", "content": "hola"}]
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(self.adaptador.medir_contexto(conversacion), (300, 8192 - 512))
        args, kwargs = self.procesador.apply_chat_template.call_args
        self.assertEqual(args, ([conversacion],))
        self.assertEqual({k: kwargs[k] for k in ("tokenize", "add_generation_prompt", "enable_thinking", "truncation")},
                         {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
                          "truncation": False})
        self.modelo.generate.assert_not_called()

    def test_limite_medido_coincide_con_el_corte_de_la_generacion(self):
        self.procesador.apply_chat_template.return_value = {"input_ids": [[]]}
        _, limite = self.adaptador.medir_contexto([])
        self.procesador.apply_chat_template.return_value = self.entradas
        self.entradas.__getitem__.return_value.shape = (1, limite)
        self.assertEqual(self.adaptador([[{}]]), ["primera"])  # justo en el límite: genera
        self.entradas.__getitem__.return_value.shape = (1, limite + 1)
        with self.assertRaisesRegex(ValueError, "Contexto excedido"):
            self.adaptador([[{}]])


if __name__ == "__main__":
    unittest.main()
