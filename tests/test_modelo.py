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
                AutoProcessor=procesador, AutoModelForCausalLM=modelo)}):
            adaptador = ModeloLocal.cargar(Configuracion())
        self.assertIs(adaptador.modelo, modelo.from_pretrained.return_value)
        self.assertEqual(procesador.from_pretrained.call_args.kwargs["padding_side"], "left")
        self.assertEqual(modelo.from_pretrained.call_args.kwargs["revision"], Configuracion.revision)
        self.assertEqual(modelo.from_pretrained.call_args.kwargs["dtype"], "bf16")
        modelo.from_pretrained.return_value.eval.assert_called_once()

    def test_factory_carga_una_vez_y_reutiliza_coordinador(self):
        import entrega
        from admision.ejemplo import AgenteJSON
        with patch.object(ModeloLocal, "cargar", return_value=self.adaptador) as cargar:
            agente = entrega.construir_agente()
        self.assertIsInstance(agente, AgenteJSON)
        self.assertIs(agente.modelo_lote, self.adaptador)
        cargar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
