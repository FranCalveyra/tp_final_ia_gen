import tempfile
import unittest
from pathlib import Path

from admision.normas import cargar_normas
from entrega import construir_agente, construir_buscador


RAIZ = Path(__file__).resolve().parents[1]


class BuscadorTests(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name)
        (self.ruta / "parametros.yaml").write_text("parametros: []\n")

    def norma(self, nombre, texto):
        (self.ruta / nombre).write_text(texto, encoding="utf-8")

    def test_orden_y_limite(self):
        self.norma("NOR-A-01.md", "registro paciente")
        self.norma("NOR-A-02.md", "cobertura paciente")
        self.norma("NOR-A-03.md", "registro paciente")
        buscador = construir_buscador(self.ruta)
        self.assertEqual([s.id for s in buscador.recuperar("registro", 5)], ["NOR-A-01", "NOR-A-03"])
        self.assertEqual(len(buscador.recuperar("registro", 1)), 1)
        self.assertEqual(buscador.k_entorno, 2)

    def test_normalizacion_preserva_texto(self):
        texto = "Identificación y AFILIACIÓN: paciente."
        self.norma("NOR-A-01.md", texto)
        buscador = construir_buscador(self.ruta)
        self.assertEqual(buscador.recuperar("IDENTIFICACION afiliacion", 2)[0].texto, texto)

    def test_consultas_vacias_y_sin_coincidencias(self):
        self.norma("NOR-A-01.md", "documento")
        buscador = construir_buscador(self.ruta)
        for consulta, k in [("", 2), ("???", 2), ("zzzz", 2), ("documento", 0), ("documento", -1)]:
            with self.subTest(consulta=consulta, k=k):
                self.assertEqual(buscador.recuperar(consulta, k), [])

    def test_corpus_vacio(self):
        self.assertEqual(construir_buscador(self.ruta).recuperar("documento", 2), [])

    def test_valores_y_textos_de_cada_version(self):
        for version in ("original", "b", "c"):
            with self.subTest(version=version):
                ruta = RAIZ / "normas" / version
                esperadas = cargar_normas(ruta)
                buscador = construir_buscador(ruta)
                for seccion in esperadas:
                    resultados = buscador.recuperar(seccion.id, 20)
                    self.assertIn(seccion, resultados)
                self.assertEqual(len(esperadas), 20)

    def test_solo_usa_carpeta_recibida_y_conserva_parametros(self):
        self.norma("NOR-A-01.md", "Regla exclusiva con parámetro P-ADM-1.")
        (self.ruta / "parametros.yaml").write_text("parametros:\n- codigo: P-ADM-1\n  valor: 99\n  unidad: min\n")
        buscador = construir_buscador(self.ruta)
        resultados = buscador.recuperar("exclusiva", 20)
        self.assertEqual(len(resultados), 1)
        self.assertIn("P-ADM-1 = 99 min", resultados[0].texto)
        self.assertEqual(buscador.recuperar("toracico", 2), [])

    def test_buscador_no_carga_modelo(self):
        from unittest.mock import patch
        with patch("admision_tp.modelo.ModeloLocal.cargar") as cargar:
            construir_buscador(self.ruta)
        cargar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
