import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clasificar_historial import (
    SI,
    aviso_segun_costumbre,
    cargar_parametros,
    cobertura_esperada,
    ejecutar,
    inferir_costumbre_aviso,
    razones_normativas,
    registro_esperado,
)


RAIZ = Path(__file__).resolve().parents[1]
PARAMETROS = cargar_parametros(RAIZ / "normas/original")


def cargar_historial() -> dict[str, dict[str, str]]:
    with (RAIZ / "historial/historial.csv").open(encoding="utf-8", newline="") as archivo:
        return {fila["id"]: fila for fila in csv.DictReader(archivo)}


def fila_aviso(identificador: str, edad: int, acompanante: str, aviso: bool, operador: str = "OP-01") -> dict:
    return {
        "id": identificador, "operador": operador, "edad": str(edad), "acompanante": acompanante,
        "puede_responder": SI, "llegada": "por sus propios medios",
        "aviso_al_voluntariado": SI if aviso else "no",
    }


class ClasificacionHistorialTest(unittest.TestCase):
    def ejecutar_en_temporal(self, trazas: bool = False):
        carpeta = Path(tempfile.mkdtemp())
        resultado = ejecutar(
            RAIZ / "historial/historial.csv",
            carpeta / "historial_clasificado.csv",
            RAIZ / "normas/original",
            RAIZ / "historial/trazas.jsonl" if trazas else None,
        )
        return carpeta, resultado

    def tipos(self, carpeta: Path) -> dict[str, str]:
        with (carpeta / "historial_clasificado.csv").open(encoding="utf-8", newline="") as archivo:
            return {fila["id"]: fila["tipo"] for fila in csv.DictReader(archivo)}

    def test_salida_tiene_dos_mil_ids_y_tipos_validos(self):
        carpeta, resultado = self.ejecutar_en_temporal()
        with (carpeta / "historial_clasificado.csv").open(encoding="utf-8", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
        self.assertEqual(len(filas), 2000)
        self.assertEqual(list(filas[0]), ["id", "tipo"])
        self.assertEqual({fila["id"] for fila in filas}, {f"H-{i:04d}" for i in range(1, 2001)})
        self.assertTrue({fila["tipo"] for fila in filas} <= {"correcta", "error", "ruido"})
        self.assertEqual(resultado["conteo"], {"correcta": 1600, "error": 240, "ruido": 160})

    def test_terminacion_reservada_notifica_este_activa_o_no(self):
        digito = str(PARAMETROS["P-COB-2"])
        otro = str((PARAMETROS["P-COB-2"] + 1) % 10)
        self.assertEqual(
            cobertura_esperada("obra_social", "OS X", f"10-1-{digito}", False, PARAMETROS),
            ("sin_cobertura", "OS X"),
        )
        self.assertEqual(
            cobertura_esperada("prepaga", "Pre X", f"1/0{digito}", True, PARAMETROS),
            ("sin_cobertura", "Pre X"),
        )
        self.assertEqual(
            cobertura_esperada("obra_social", "OS X", f"10-1-{otro}", False, PARAMETROS),
            ("sin_cobertura", ""),
        )
        self.assertEqual(
            cobertura_esperada("art", "ART X", f"20-1-{digito}", True, PARAMETROS),
            ("art", "ART X"),
        )

    def test_terminacion_reservada_inactiva_en_el_historial(self):
        historial = cargar_historial()
        # H-0003: número reservado, afiliación inactiva y notificación a la entidad (NOR-COB-04 §3).
        self.assertEqual(razones_normativas(historial["H-0003"], PARAMETROS), [])
        # H-0378: mismo caso sin notificar: viola NOR-COB-04 §1.
        self.assertEqual(razones_normativas(historial["H-0378"], PARAMETROS), ["entidad_a_notificar"])
        carpeta, _ = self.ejecutar_en_temporal()
        tipos = self.tipos(carpeta)
        self.assertEqual(tipos["H-0003"], "correcta")
        self.assertEqual(tipos["H-0378"], "error")
        # H-0140 cumple NOR-COB-04 pero avisa al voluntariado a un menor acompañado.
        self.assertEqual(tipos["H-0140"], "ruido")

    def test_prepaga_inactiva_queda_sin_cobertura_sin_recurrir_a_la_obra_social(self):
        historial = cargar_historial()
        fila = dict(historial["H-0021"])  # obra social + prepaga; se verificó la prepaga, inactiva
        self.assertEqual(razones_normativas(fila, PARAMETROS), [])
        fila["cobertura"], fila["entidad_a_notificar"] = "obra_social", "Obra Social Fluvial"
        self.assertIn("cobertura", razones_normativas(fila, PARAMETROS))

        _, resultado = self.ejecutar_en_temporal(trazas=True)
        credencial = next(c for c in resultado["costumbres"] if c["tipo"] == "seleccion_credencial")
        self.assertEqual(credencial["preferido"], "prepaga")
        self.assertEqual(credencial["frecuencias"], {"obra_social": 9, "prepaga": 163})
        self.assertEqual(credencial["preferido_inactiva"]["coberturas"], {"sin_cobertura": 39})
        self.assertIn("H-0021", credencial["preferido_inactiva"]["ids_respaldo"])
        self.assertEqual(credencial["todas_las_filas_preferido_inactiva"],
                         {"muestra": 45, "coberturas": {"sin_cobertura": 45}})
        self.assertEqual(credencial["verificaciones_por_traza"], {"1": 193})
        self.assertEqual(len(credencial["ids_ruido"]), 9)
        self.assertEqual(credencial["operadores_con_desvios"], ["OP-04", "OP-11", "OP-17"])

    def test_costumbre_del_voluntariado_considera_edad_y_acompanante(self):
        costumbre = {"umbral_edad": 75, "condicion": "sin_acompanante"}
        self.assertTrue(aviso_segun_costumbre(fila_aviso("a", 75, "", False), costumbre))
        self.assertFalse(aviso_segun_costumbre(fila_aviso("b", 80, "su hija Paula", False), costumbre))
        self.assertFalse(aviso_segun_costumbre(fila_aviso("c", 74, "", False), costumbre))

    def test_busqueda_de_la_costumbre_encuentra_umbral_y_condicion(self):
        filas = []
        for edad in range(60, 90):
            filas.append(fila_aviso(f"s-{edad}", edad, "", edad >= 81))
            filas.append(fila_aviso(f"c-{edad}", edad, "su hijo Juan", False))
        filas.append(fila_aviso("desvio", 65, "", True, operador="OP-99"))
        costumbre = inferir_costumbre_aviso(filas, filas)
        self.assertEqual((costumbre["umbral_edad"], costumbre["condicion"]), (81, "sin_acompanante"))
        self.assertEqual(costumbre["ids_ruido"], ["desvio"])
        self.assertEqual(costumbre["operadores_con_desvios"], ["OP-99"])
        self.assertLess(
            costumbre["comparacion_solo_edad"]["coinciden_limpias"], costumbre["coinciden_limpias"]
        )

    def test_costumbre_del_voluntariado_en_el_historial(self):
        carpeta, resultado = self.ejecutar_en_temporal()
        edad = next(c for c in resultado["costumbres"] if c["tipo"] == "aviso_al_voluntariado")
        self.assertEqual((edad["umbral_edad"], edad["condicion"]), (75, "sin_acompanante"))
        self.assertEqual(edad["operadores_con_desvios"], ["OP-04", "OP-11", "OP-17"])
        self.assertEqual(edad["todas_las_filas_operadores_sin_desvios"], {
            "avisar": {"muestra": 185, "siguen_costumbre": 185},
            "no_avisar_menores": {"muestra": 1348, "siguen_costumbre": 1348},
            "no_avisar_mayores_sin_condicion": {"muestra": 141, "siguen_costumbre": 141},
        })
        # Respaldo en ambos lados y ruido consistente: toda fila limpia que se aparta es ruido.
        self.assertTrue(edad["lados"]["avisar"]["ids_respaldo"])
        self.assertTrue(edad["lados"]["no_avisar_mayores_sin_condicion"]["ids_respaldo"])
        self.assertTrue(edad["lados"]["no_avisar_menores"]["ids_respaldo"])
        tipos = self.tipos(carpeta)
        historial = cargar_historial()
        for identificador, tipo in tipos.items():
            if tipo == "correcta":
                self.assertEqual(
                    aviso_segun_costumbre(historial[identificador], edad),
                    historial[identificador]["aviso_al_voluntariado"] == SI,
                    identificador,
                )
        self.assertEqual(tipos["H-0005"], "ruido")
        self.assertEqual(tipos["H-0014"], "error")
        self.assertEqual(tipos["H-0011"], "correcta")
        self.assertEqual({tipos[i] for i in edad["ids_ruido"]}, {"ruido"})

    def test_trazas_validan_fichas_y_documentan_desvios_por_operador(self):
        carpeta, resultado = self.ejecutar_en_temporal(trazas=True)
        self.assertEqual(resultado["trazas"]["filas"], 2000)
        self.assertEqual(resultado["trazas"]["ids_coincidentes"], 2000)
        self.assertEqual(resultado["trazas"]["fichas_coincidentes"], 2000)
        operadores = [item["operador"] for item in resultado["desvios_por_operador"]]
        self.assertEqual(operadores, ["OP-04", "OP-11", "OP-17"])
        self.assertEqual(sum(item["total"] for item in resultado["desvios_por_operador"]), 160)
        texto = (carpeta / "costumbres.md").read_text(encoding="utf-8")
        self.assertIn("H-0005", texto)
        self.assertIn("OP-04", texto)
        self.assertIn("sin acompañante", texto)

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
