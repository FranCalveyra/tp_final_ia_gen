"""Instrumentación aislada: sin autenticación, red ni modelo."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.probar_wandb import probar


class TrackingTests(unittest.TestCase):
    def test_offline_por_defecto_registra_y_cierra(self):
        sdk = MagicMock()
        sdk.init.return_value.__enter__.return_value.id = "simulada"
        with tempfile.TemporaryDirectory() as temporal, patch.dict(sys.modules, {"wandb": sdk}), \
                patch.dict(os.environ, {}, clear=True):
            resultado = probar(Path(temporal))
        self.assertEqual(resultado, {"id": "simulada", "modo": "offline"})
        parametros = sdk.init.call_args.kwargs
        self.assertEqual(parametros["mode"], "offline")
        self.assertEqual(parametros["entity"], "francalveyra-ia-gen")
        self.assertEqual(parametros["project"], "tp-admision")
        sdk.init.return_value.__enter__.return_value.log.assert_called_once_with({"setup_check": 1})
        sdk.init.return_value.__exit__.assert_called_once()

    def test_online_sin_clave_falla_antes_de_importar_sdk(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "WANDB_API_KEY"):
                probar(Path("no-se-crea"), online=True)

    def test_secret_y_separacion_de_inferencia(self):
        raiz = Path(__file__).resolve().parents[1]
        tracking = (raiz / "infra/wandb_app.py").read_text()
        self.assertIn('modal.Secret.from_name("wandb-secret")', tracking)
        self.assertNotIn('gpu=', tracking)
        self.assertLess(tracking.index('.env('), tracking.index('.add_local_file('))
        self.assertNotIn('wandb', (raiz / "infra/modal_app.py").read_text())


if __name__ == "__main__":
    unittest.main()
