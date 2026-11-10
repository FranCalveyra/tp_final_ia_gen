import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clasificar_historial import ejecutar, registro_esperado


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
        self.assertEqual(resultado["conteo"], {"correcta": 1631, "error": 279, "ruido": 90})

    def test_reglas_costumbres_y_preferencia_credenciales(self):
        carpeta, resultado = self.ejecutar_en_temporal()
        with (carpeta / "historial_clasificado.csv").open(encoding="utf-8", newline="") as archivo:
            tipos = {fila["id"]: fila["tipo"] for fila in csv.DictReader(archivo)}
        self.assertEqual(tipos["H-0014"], "error")
        self.assertEqual(tipos["H-0040"], "error")
        self.assertEqual(tipos["H-0010"], "error")
        self.assertEqual(tipos["H-0011"], "correcta")
        self.assertEqual(tipos["H-0005"], "ruido")

        costumbres = resultado["costumbres"]
        edad = next(c for c in costumbres if c["tipo"] == "aviso_al_voluntariado")
        credenciales = next(c for c in costumbres if c["tipo"] == "seleccion_credencial")
        self.assertEqual(edad["umbral_edad"], 75)
        self.assertEqual(edad["situacion_fields"], ["motivo", "cobertura"])
        self.assertEqual(len(edad["estratos_comparables"]), 8)
        self.assertTrue(edad["ids_respaldo"])
        self.assertNotIn("H-0140", edad["ids_ruido"])
        self.assertEqual(credenciales["preferido"], "prepaga")
        self.assertEqual(credenciales["muestra"], 171)
        self.assertEqual(credenciales["frecuencias"], {"obra_social": 9, "prepaga": 162})

    def test_trazas_validan_fichas_y_documentan_patron_de_operadores(self):
        carpeta = Path(tempfile.mkdtemp())
        resultado = ejecutar(
            RAIZ / "historial/historial.csv",
            carpeta / "historial_clasificado.csv",
            RAIZ / "normas/original",
            RAIZ / "historial/trazas.jsonl",
        )
        self.assertEqual(resultado["trazas"]["filas"], 2000)
        self.assertEqual(resultado["trazas"]["ids_coincidentes"], 2000)
        self.assertEqual(resultado["trazas"]["fichas_coincidentes"], 2000)
        patron = resultado["patrones_adicionales"][0]
        self.assertEqual(patron["grupos_destacados"], ["OP-04", "OP-11", "OP-17"])
        self.assertEqual(patron["resto"], {"muestra": 1190, "si": 0, "frecuencia_si": 0.0})
        self.assertFalse(patron["confirmada_como_costumbre_hospital"])
        texto = (carpeta / "costumbres.md").read_text(encoding="utf-8")
        self.assertIn("H-0005", texto)
        self.assertIn("OP-04", texto)

    def test_parametro_adm3_y_documento_se_aplican(self):
        row = {"documento_tipo": "pasaporte", "nivel": "3"}
        self.assertEqual(registro_esperado(row, {"P-ADM-3": 3}), "provisorio")
        self.assertEqual(registro_esperado(row, {"P-ADM-3": 2}), "definitivo")

    def test_salida_usa_fin_de_linea_lf(self):
        carpeta, _ = self.ejecutar_en_temporal()
        contenido = (carpeta / "historial_clasificado.csv").read_bytes()
        self.assertNotIn(b"\r", contenido)
        self.assertTrue(contenido.startswith(b"id,tipo\n"))

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
