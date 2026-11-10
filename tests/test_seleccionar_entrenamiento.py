import tempfile
import unittest
from pathlib import Path

from admision_tp.conversaciones import validar_registro
from scripts.seleccionar_entrenamiento import convertir_historial


RAIZ = Path(__file__).resolve().parents[1]


class SeleccionEntrenamientoTest(unittest.TestCase):
    def convertir(self):
        return convertir_historial(
            RAIZ / "historial/historial.csv",
            RAIZ / "historial_clasificado.csv",
            RAIZ / "historial_costumbres.json",
            RAIZ / "historial/trazas.jsonl",
        )

    def test_selecciona_solo_casos_reales_curados(self):
        conversaciones, resumen = self.convertir()
        self.assertEqual(len(conversaciones), 1595)
        self.assertEqual(resumen["excluidas_error"], 279)
        self.assertEqual(resumen["excluidas_ruido"], 90)
        self.assertEqual(resumen["excluidas_ambiguas"], 36)
        self.assertEqual(resumen["conversaciones_reales"], 1595)
        for conversacion in conversaciones:
            validar_registro(conversacion)
            self.assertEqual(conversacion["origen"], "historial")
            self.assertNotIn("practica", " ".join(conversacion["referencias"]))
            self.assertNotIn("test", " ".join(conversacion["referencias"]))

    def test_conversion_real_es_determinista(self):
        primera, resumen_a = self.convertir()
        segunda, resumen_b = self.convertir()
        self.assertEqual(primera, segunda)
        self.assertEqual(resumen_a, resumen_b)
        self.assertEqual(primera[0]["id"], "H-0001")
        self.assertEqual(primera[-1]["origen"], "historial")


if __name__ == "__main__":
    unittest.main()
