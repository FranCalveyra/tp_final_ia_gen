import tempfile
import unittest
from pathlib import Path

from admision_tp.conversaciones import validar_registro
from scripts.seleccionar_entrenamiento import convertir_historial, seleccionar_ids


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
        self.assertEqual(len(conversaciones), 1600)
        self.assertEqual(resumen["excluidas_error"], 240)
        self.assertEqual(resumen["excluidas_ruido"], 160)
        self.assertEqual(resumen["conversaciones_reales"], 1600)
        for conversacion in conversaciones:
            validar_registro(conversacion)
            self.assertEqual(conversacion["origen"], "historial")
            self.assertNotIn("practica", " ".join(conversacion["referencias"]))
            self.assertNotIn("test", " ".join(conversacion["referencias"]))

    def test_rechaza_correctas_que_contradicen_la_costumbre(self):
        costumbre = {"umbral_edad": 75, "condicion": "sin_acompanante"}
        fila = {"id": "H-1", "edad": "80", "acompanante": "su hija Paula", "aviso_al_voluntariado": "s\u00ed"}
        with self.assertRaises(ValueError):
            seleccionar_ids([fila], {"H-1": "correcta"}, costumbre)
        seleccionados, _ = seleccionar_ids([fila], {"H-1": "ruido"}, costumbre)
        self.assertEqual(seleccionados, [])

    def test_conversion_real_es_determinista(self):
        primera, resumen_a = self.convertir()
        segunda, resumen_b = self.convertir()
        self.assertEqual(primera, segunda)
        self.assertEqual(resumen_a, resumen_b)
        self.assertEqual(primera[0]["id"], "H-0001")
        self.assertEqual(primera[-1]["origen"], "historial")


if __name__ == "__main__":
    unittest.main()
