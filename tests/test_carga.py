import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from admision_tp.carga import cargar_solo_texto


class CargaSoloTextoTest(unittest.TestCase):
    def test_saca_vision_y_audio_y_conserva_el_resto(self):
        config = SimpleNamespace(vision_config={"x": 1}, audio_config={"y": 2}, text_config={"z": 3})
        transformers = SimpleNamespace(AutoConfig=MagicMock(), AutoModelForCausalLM=MagicMock())
        transformers.AutoConfig.from_pretrained.return_value = config
        modelo = cargar_solo_texto(transformers, "google/gemma", "a" * 40, dtype="fp16", device_map="auto")
        transformers.AutoConfig.from_pretrained.assert_called_once_with("google/gemma", revision="a" * 40)
        self.assertEqual((config.vision_config, config.audio_config, config.text_config), (None, None, {"z": 3}))
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "google/gemma", revision="a" * 40, config=config, dtype="fp16", device_map="auto")
        self.assertIs(modelo, transformers.AutoModelForCausalLM.from_pretrained.return_value)

    def test_modelo_sin_torres_no_falla(self):
        config = SimpleNamespace(text_config={})
        transformers = SimpleNamespace(AutoConfig=MagicMock(), AutoModelForCausalLM=MagicMock())
        transformers.AutoConfig.from_pretrained.return_value = config
        cargar_solo_texto(transformers, "otro", "b" * 40)
        self.assertFalse(hasattr(config, "vision_config"))


if __name__ == "__main__":
    unittest.main()
