import io
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts import entrenar as cli


class CliEntrenarTest(unittest.TestCase):
    def test_fallo_muestra_el_traceback_y_sale_con_1(self):
        argv = ["entrenar", "--preparacion", "p", "--tokens", "t", "--salida", "s"]
        errores = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch.object(cli, "entrenar", side_effect=ValueError("detalle del fallo")), \
                redirect_stderr(errores), self.assertRaises(SystemExit) as salida:
            cli.main()
        self.assertEqual(salida.exception.code, 1)
        self.assertIn("ValueError: detalle del fallo", errores.getvalue())
        self.assertIn("Traceback", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
