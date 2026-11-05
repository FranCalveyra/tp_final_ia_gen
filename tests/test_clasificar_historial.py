import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clasificar_historial import ejecutar


RAIZ = Path(__file__).resolve().parents[1]


class ClasificacionHistorialTest(unittest.TestCase):
    def ejecutar_en_temporal(self):
        carpeta = Path(tempfile.mkdtemp())
        resultado = ejecutar(
            RAIZ / "historial/historial.csv",
            carpeta / "historial_clasificado.csv",
            RAIZ / "normas/original",
        )
        return carpeta, resultado

    def test_salida_tiene_dos_mil_ids_y_tipos_validos(self):
        carpeta, resultado = self.ejecutar_en_temporal()
        with (carpeta / "historial_clasificado.csv").open(encoding="utf-8", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
        self.assertEqual(len(filas), 2000)
        self.assertEqual(list(filas[0]), ["id", "tipo"])
        self.assertEqual({fila["id"] for fila in filas}, {f"H-{i:04d}" for i in range(1, 2001)})
        self.assertTrue({fila["tipo"] for fila in filas} <= {"correcta", "error", "ruido"})
        self.assertEqual(sum(resultado["conteo"].values()), 2000)

    def test_reglas_normativas_y_costumbre_tienen_casos_observables(self):
        carpeta, resultado = self.ejecutar_en_temporal()
        with (carpeta / "historial_clasificado.csv").open(encoding="utf-8", newline="") as archivo:
            tipos = {fila["id"]: fila["tipo"] for fila in csv.DictReader(archivo)}
        self.assertEqual(tipos["H-0014"], "error")  # registro incompatible con pasaporte y nivel urgente
        self.assertEqual(tipos["H-0040"], "error")  # box de paciente mayor antes del registro
        self.assertEqual(tipos["H-0010"], "error")  # verificación de la credencial equivocada
        self.assertEqual(tipos["H-0005"], "ruido")  # apartamiento etario de la costumbre observada
        self.assertEqual(resultado["costumbre"]["umbral_edad"], 75)
        self.assertEqual(resultado["costumbre"]["muestra_menores"], 1614)
        self.assertEqual(resultado["costumbre"]["no_menores"], 1494)

    def test_segunda_ejecucion_es_identica(self):
        carpeta_a, resultado_a = self.ejecutar_en_temporal()
        carpeta_b, resultado_b = self.ejecutar_en_temporal()
        self.assertEqual(
            (carpeta_a / "historial_clasificado.csv").read_bytes(),
            (carpeta_b / "historial_clasificado.csv").read_bytes(),
        )
        self.assertEqual(resultado_a, resultado_b)
        self.assertEqual(
            json.loads((carpeta_a / "historial_costumbres.json").read_text(encoding="utf-8")),
            json.loads((carpeta_b / "historial_costumbres.json").read_text(encoding="utf-8")),
        )


if __name__ == "__main__":
    unittest.main()
