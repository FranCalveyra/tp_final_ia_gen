import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.explorar_buscadores import explorar, main

RAIZ = Path(__file__).resolve().parents[1]


class ExplorarBuscadoresTests(unittest.TestCase):
    def test_reproduce_baseline_y_un_intento_rechazado(self):
        informe = explorar(RAIZ / "normas/original", RAIZ / "rag/consultas_dev.jsonl",
                           ["baseline", "baseline_exploratorio", "parrafos_top2"])
        r = informe["resultados"]
        self.assertEqual(r["baseline"]["metricas"]["consultas_con_toda_la_evidencia"], 48)
        self.assertEqual(r["baseline"]["metricas"]["tokens_por_consulta"], 618)
        # La clase exploratoria con valores por defecto equivale al buscador real.
        self.assertEqual(r["baseline_exploratorio"]["metricas"], r["baseline"]["metricas"])
        self.assertEqual(r["baseline_exploratorio"]["completas"], r["baseline"]["completas"])
        self.assertEqual(r["parrafos_top2"]["metricas"]["consultas_con_toda_la_evidencia"], 39)
        self.assertEqual(len(informe["configuracion"]["normas_sha256"]), 21)

    def test_no_pisa_salida(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp) / "exploracion.json"
            salida.write_text("previo")
            argv = ["explorar", "--normas", str(RAIZ / "normas/original"),
                    "--consultas", str(RAIZ / "rag/consultas_dev.jsonl"), "--salida", str(salida)]
            with patch("sys.argv", argv), patch("scripts.explorar_buscadores.explorar", return_value={}):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(salida.read_text(), "previo")


if __name__ == "__main__":
    unittest.main()
