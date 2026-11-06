import json
import tempfile
import unittest
from pathlib import Path

from admision_tp.conversaciones import validar_archivo
from scripts.preparar_datos_entrenamiento import preparar


RAIZ = Path(__file__).resolve().parents[1]


class PrepararDatosEntrenamientoTest(unittest.TestCase):
    def test_dataset_combinado_y_manifest_son_validos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            salida = Path(carpeta) / "entrenamiento"
            manifiesto = preparar(RAIZ, salida)
            self.assertEqual(manifiesto["reales"]["conversaciones_reales"], 1595)
            self.assertEqual(manifiesto["sinteticos"]["sinteticos_seleccionados"], 500)
            self.assertEqual(manifiesto["total_conversaciones"], 2095)
            self.assertEqual(manifiesto["por_origen"], {"historial": 1595, "sintetico": 500})
            informe = validar_archivo(salida / "conversaciones.jsonl")
            self.assertEqual(informe["validos"], 2095)
            self.assertFalse(informe["errores"])
            self.assertEqual(
                json.loads((salida / "manifiesto.json").read_text(encoding="utf-8")),
                manifiesto,
            )

    def test_no_se_puede_sobrescribir_la_salida(self):
        with tempfile.TemporaryDirectory() as carpeta:
            salida = Path(carpeta) / "entrenamiento"
            salida.mkdir()
            with self.assertRaises(FileExistsError):
                preparar(RAIZ, salida)


if __name__ == "__main__":
    unittest.main()
