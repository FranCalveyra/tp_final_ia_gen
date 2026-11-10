import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from admision.evaluar_rag import cargar_dataset
from admision_tp.buscador import BuscadorBM25, ConfigBM25
from entrega import construir_buscador
from scripts.comparar_buscadores import FusionLexica, ajustar_evidencia, comparar, ngramas

RAIZ = Path(__file__).resolve().parents[1]


class VariantesTests(unittest.TestCase):
    def test_seleccion_explicita_y_validacion(self):
        ruta = RAIZ / "normas/original"
        self.assertEqual(BuscadorBM25(ruta).variante, "baseline")
        for nombre, k in (("baseline", 2), ("sin_comunes", 2), ("prefijos", 3)):
            with patch.dict("os.environ", {"BUSCADOR_VARIANTE": nombre}):
                buscador = construir_buscador(ruta)
                self.assertEqual(buscador.variante, nombre)
                self.assertEqual(buscador.k_entorno, k)
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

    def carpeta(self, normas):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ruta = Path(tmp.name)
        (ruta / "parametros.yaml").write_text("parametros: []\n")
        for nombre, texto in normas.items():
            (ruta / nombre).write_text(texto, encoding="utf-8")
        return ruta

    def test_prefijos_unen_flexiones_sin_alterar_texto_ni_numeros(self):
        texto = "Se REGISTRA al paciente en 120 minutos."
        ruta = self.carpeta({"NOR-A-01.md": texto, "NOR-A-02.md": "Otra cosa 12."})
        b = BuscadorBM25(ruta, "prefijos")
        self.assertEqual([s.texto for s in b.recuperar("registrarlo pacientes", 3)], [texto])
        self.assertEqual(BuscadorBM25(ruta, "sin_comunes").recuperar("registrarlo", 3), [])
        self.assertEqual([s.id for s in b.recuperar("12", 3)], ["NOR-A-02"])

    def test_umbral_relativo_descarta_secciones_debiles(self):
        ruta = self.carpeta({"NOR-A-01.md": "sepsis sepsis codigo plazo",
                             "NOR-A-02.md": "plazo de otra area con mucho texto adicional distinto",
                             "NOR-A-03.md": "nada que ver"})
        sin_umbral = BuscadorBM25(ruta, ConfigBM25(k_entorno=3))
        con_umbral = BuscadorBM25(ruta, ConfigBM25(umbral_relativo=0.6, k_entorno=3))
        self.assertEqual([s.id for s in sin_umbral.recuperar("sepsis plazo", 3)], ["NOR-A-01", "NOR-A-02"])
        self.assertEqual([s.id for s in con_umbral.recuperar("sepsis plazo", 3)], ["NOR-A-01"])
        self.assertEqual(con_umbral.variante, "personalizada")

    def test_evidencia_ajustada_usa_valores_de_la_carpeta(self):
        dataset = cargar_dataset(RAIZ / "rag/consultas_dev.jsonl")
        self.assertEqual(ajustar_evidencia(dataset, RAIZ / "normas/original"), dataset)
        ajustado = {x["id"]: x for x in ajustar_evidencia(dataset, RAIZ / "normas/b")}
        self.assertNotEqual(ajustado["d034"]["evidencia"], {x["id"]: x for x in dataset}["d034"]["evidencia"])
        self.assertEqual(ajustado["d001"], dataset[0])

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
        prefijos = r["prefijos"]
        self.assertEqual(prefijos["metricas"]["consultas_con_toda_la_evidencia"], 55)
        self.assertEqual(prefijos["metricas"]["tokens_por_consulta"], 549)
        self.assertEqual(prefijos["ganadas_vs_baseline"], ["d021", "d022", "d024", "d026", "d028",
                                                           "d031", "d032", "d034", "d049"])
        self.assertEqual(prefijos["perdidas_vs_baseline"], ["d027", "d052"])
        self.assertEqual(prefijos["perdidas_vs_sin_comunes"], ["d027", "d052"])


if __name__ == "__main__":
    unittest.main()
