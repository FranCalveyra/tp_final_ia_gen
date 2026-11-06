import unittest
from pathlib import Path

from scripts.generar_sinteticos_entrenamiento import generar_sinteticos


RAIZ = Path(__file__).resolve().parents[1]


class SinteticosEntrenamientoTest(unittest.TestCase):
    def generar(self):
        return generar_sinteticos(
            RAIZ / "normas/original",
            RAIZ / "historial_costumbres.json",
        )

    def test_balance_y_fichas_normativamente_validas(self):
        conversaciones, auditoria, resumen = self.generar()
        self.assertEqual(len(conversaciones), 500)
        self.assertEqual(len(auditoria), 500)
        self.assertEqual(resumen["candidatos_generados"], 5000)
        self.assertEqual(resumen["por_circuito"], {
            "codigo_acv": 100,
            "ecg": 100,
            "ninguno": 100,
            "sepsis_posible": 100,
            "sepsis_probable": 100,
        })
        for bucket in ("art", "dos_credenciales", "acompanante", "sin_nadie"):
            self.assertGreaterEqual(resumen["por_dificultad"][bucket], 60)
        self.assertEqual(resumen["fichas_normativamente_validas"], 500)
        self.assertEqual(resumen["avisos_mayores_observados"], 175)
        self.assertEqual(resumen["muestra_mayores_observada"], 306)
        self.assertAlmostEqual(resumen["probabilidad_aviso_mayores"], 175 / 306, places=6)
        mayores = [item for item in auditoria if item["edad"] >= resumen["umbral_costumbre_edad"]]
        menores = [item for item in auditoria if item["edad"] < resumen["umbral_costumbre_edad"]]
        self.assertGreater(sum(item["ficha"]["aviso_al_voluntariado"] for item in mayores), 0)
        self.assertLess(sum(item["ficha"]["aviso_al_voluntariado"] for item in mayores), len(mayores))
        self.assertTrue(all(not item["ficha"]["aviso_al_voluntariado"] for item in menores))
        self.assertEqual(conversaciones[0]["origen"], "sintetico")
        self.assertTrue(all("practica" not in " ".join(c["referencias"]) for c in conversaciones))
        self.assertTrue(all("test" not in " ".join(c["referencias"]) for c in conversaciones))

    def test_generacion_es_determinista(self):
        primera = self.generar()
        segunda = self.generar()
        self.assertEqual(primera, segunda)
        self.assertEqual(primera[0][0]["id"], "S-12012-0001")
        self.assertEqual(primera[1][0]["id"], "S-12012-0001")


if __name__ == "__main__":
    unittest.main()
