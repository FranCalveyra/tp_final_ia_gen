import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from admision_tp.buscador import BuscadorBM25
from entrega import construir_buscador
from scripts.comparar_buscadores import FusionLexica, comparar, ngramas

RAIZ = Path(__file__).resolve().parents[1]


class VariantesTests(unittest.TestCase):
    def test_seleccion_explicita_y_validacion(self):
        ruta = RAIZ / "normas/original"
        self.assertEqual(BuscadorBM25(ruta).variante, "baseline")
        for nombre in ("baseline", "sin_comunes"):
            with patch.dict("os.environ", {"BUSCADOR_VARIANTE": nombre}):
                self.assertEqual(construir_buscador(ruta).variante, nombre)
        with patch.dict("os.environ", {"BUSCADOR_VARIANTE": "desconocida"}):
            with self.assertRaises(ValueError):
                construir_buscador(ruta)

    def test_filtrado_simetrico_sin_alterar_texto_ni_negaciones(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp)
            (ruta / "parametros.yaml").write_text("parametros: []\n")
            texto = "EL no sin ni AFILIACIÓN 42."
            (ruta / "NOR-A-01.md").write_text(texto)
            b = BuscadorBM25(ruta, "sin_comunes")
            self.assertEqual(b.recuperar("el la le", 2), [])
            for consulta in ("NO", "sin", "ni", "afiliacion", "42"):
                self.assertEqual(b.recuperar(consulta, 2)[0].texto, texto)
            self.assertTrue(BuscadorBM25(ruta).recuperar("el", 2))

    def test_ngramas_respetan_bordes_y_normalizacion(self):
        self.assertEqual(ngramas("Á"), {" a ": 1})
        self.assertEqual(ngramas("ÉCG"), ngramas("ecg"))
        self.assertNotIn("a b", ngramas("a b"))

    def test_fusion_vacia_y_limite(self):
        b = FusionLexica(RAIZ / "normas/original")
        self.assertEqual(b.recuperar("", 2), [])
        self.assertEqual(b.recuperar("registro", 0), [])
        self.assertEqual(len(b.recuperar("registro", 1)), 1)

    def test_reproduce_experimento_dev_y_expone_regresiones(self):
        r = comparar(RAIZ / "normas/original", RAIZ / "rag/consultas_dev.jsonl")["resultados"]
        self.assertEqual(r["baseline"]["metricas"]["consultas_con_toda_la_evidencia"], 48)
        self.assertEqual(r["sin_comunes"]["metricas"]["consultas_con_toda_la_evidencia"], 50)
        self.assertEqual(r["sin_comunes"]["ganadas_vs_baseline"], ["d022", "d026", "d028"])
        self.assertEqual(r["sin_comunes"]["perdidas_vs_baseline"], ["d011"])
        self.assertEqual(r["fusion_lexica"]["metricas"]["consultas_con_toda_la_evidencia"], 48)
        self.assertEqual(r["fusion_lexica"]["perdidas_vs_baseline"], ["d011", "d025", "d027"])


if __name__ == "__main__":
    unittest.main()
