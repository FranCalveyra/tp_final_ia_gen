"""Conjunto de desarrollo para seleccionar prompts (issue #13), probado con los datos reales del repo."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from admision.casos import cargar_casos, cargar_esperadas
from admision.comparar import comparar
from admision.correr import revisar_que_no_haya_esperadas
from admision.entorno import CAMPOS, Entorno, nota_de_triage
from admision_tp.buscador import BuscadorBM25
from scripts.generar_prompt_dev import (
    SEMILLA_ENTRENAMIENTO,
    cuotas,
    generar_conjunto,
    huellas_excluidas,
    solapamiento,
)

RAIZ = Path(__file__).resolve().parents[1]
DEV = RAIZ / "datos/prompt-dev/v1"


def carpetas():
    return sorted(p.name for p in (DEV / "entradas").iterdir())


def leer(nombre):
    return (cargar_casos(DEV / "entradas" / nombre / "casos.jsonl"),
            cargar_esperadas(DEV / "esperadas" / f"esperadas-{nombre}.jsonl"))


def valores(ruta_normas):
    datos = yaml.safe_load((ruta_normas / "parametros.yaml").read_text(encoding="utf-8"))
    return {p["codigo"]: p["valor"] for p in datos["parametros"]}


class ConjuntoVersionado(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifiesto = json.loads((DEV / "manifiesto.json").read_text(encoding="utf-8"))

    def test_el_script_reproduce_los_archivos_versionados(self):
        with tempfile.TemporaryDirectory() as tmp:
            nuevo = generar_conjunto(Path(tmp) / "dev")
            for relativa, sha in nuevo["archivos_sha256"].items():
                self.assertEqual((Path(tmp) / "dev" / relativa).read_bytes(), (DEV / relativa).read_bytes(), relativa)
            sin_git = lambda m: {k: v for k, v in m.items() if k != "git"}  # noqa: E731
            self.assertEqual(sin_git(nuevo), sin_git(self.manifiesto))
        archivos = {p.relative_to(DEV).as_posix() for p in DEV.rglob("*") if p.is_file()}
        self.assertEqual(archivos - {"manifiesto.json"}, set(self.manifiesto["archivos_sha256"]))

    def test_rechaza_sobrescribir_semilla_de_entrenamiento_y_practica(self):
        with self.assertRaises(FileExistsError):
            generar_conjunto(DEV)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generar_conjunto(Path(tmp) / "x", semilla=SEMILLA_ENTRENAMIENTO)
            self.assertFalse((Path(tmp) / "x").exists())
        with self.assertRaises(ValueError):
            generar_conjunto(RAIZ / "practica/dev-no-permitido")
        self.assertFalse((RAIZ / "practica/dev-no-permitido").exists())

    def test_formato_del_kit_y_ids(self):
        ids, pacientes = set(), set()
        for nombre in carpetas():
            casos, esperadas = leer(nombre)
            self.assertEqual(len(casos), self.manifiesto["casos_por_carpeta"])
            self.assertEqual([c.id for c in casos], list(esperadas))
            for caso in casos:
                esperada = esperadas[caso.id]
                self.assertEqual(set(esperada), {"caso", "ficha", "verificacion"})
                self.assertEqual(list(esperada["ficha"]), CAMPOS)
                self.assertEqual(set(esperada["verificacion"]), {"requerida", "prohibidas"})
                ids.add(caso.id)
                pacientes.add(caso.paciente_id)
        self.assertEqual(len(ids), len(carpetas()) * self.manifiesto["casos_por_carpeta"])
        self.assertEqual(len(pacientes), len(ids))

    def test_esperadas_resolubles_en_el_entorno_del_kit(self):
        """Un agente que verifica lo requerido y registra la esperada pasa los nueve controles."""
        for nombre in carpetas():
            casos, esperadas = leer(nombre)
            buscador = BuscadorBM25(DEV / "entradas" / nombre / "normas")
            for caso in casos:
                esperada = esperadas[caso.id]
                entorno = Entorno(caso, buscador)
                entorno.paso("leer_triage", {"paciente_id": caso.paciente_id})
                if esperada["verificacion"]["requerida"]:
                    entidad, numero = esperada["verificacion"]["requerida"]
                    observacion, _ = entorno.paso("verificar_afiliacion",
                                                  {"entidad": entidad, "numero_afiliado": numero})
                    self.assertNotIn("Error", observacion)
                entorno.paso("registrar_decision", esperada["ficha"])
                self.assertTrue(entorno.terminado, caso.id)
                self.assertTrue(comparar(esperada, entorno.decision, entorno.log)["resuelto"], caso.id)

    def test_esperadas_fuera_del_alcance_del_agente(self):
        for nombre in carpetas():
            revisar_que_no_haya_esperadas(DEV / "entradas" / nombre / "normas",
                                          DEV / "entradas" / nombre / "casos.jsonl")

    def test_sin_pacientes_de_practica_ni_de_entrenamiento(self):
        huellas = huellas_excluidas(RAIZ)
        practica = {"\n".join(nota_de_triage(c).splitlines()[1:]) for c in cargar_casos(RAIZ / "practica/casos.jsonl")}
        for nombre in carpetas():
            for caso in leer(nombre)[0]:
                self.assertIsNone(solapamiento(caso, huellas), caso.id)
                self.assertNotIn("\n".join(nota_de_triage(caso).splitlines()[1:]), practica)
        self.assertNotEqual(self.manifiesto["semilla"], SEMILLA_ENTRENAMIENTO)
        self.assertEqual(self.manifiesto["exclusion"]["normas_b_c"], "no se leen")

    def test_normas_con_textos_originales_y_valores_de_la_referencia(self):
        originales = valores(RAIZ / "normas/original")
        referencia = self.manifiesto["parametros"]["valores_de_la_referencia"]
        conjuntos = []
        for nombre in carpetas():
            normas = DEV / "entradas" / nombre / "normas"
            for seccion in sorted((RAIZ / "normas/original").glob("NOR-*.md")):
                self.assertEqual((normas / seccion.name).read_bytes(), seccion.read_bytes())
            actuales = valores(normas)
            self.assertEqual(set(actuales), set(originales))
            self.assertNotEqual(actuales, originales)
            for codigo, valor in actuales.items():
                self.assertIn(valor, referencia.get(codigo, [originales[codigo]]), codigo)
            conjuntos.append(actuales)
        self.assertEqual(len(conjuntos), self.manifiesto["carpetas"])
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in conjuntos}), len(conjuntos))

    def test_balance_cubre_los_controles_debiles(self):
        umbral = self.manifiesto["costumbres"]["aviso_umbral_edad"]
        q = cuotas(self.manifiesto["casos_por_carpeta"])
        for nombre in carpetas():
            casos, esperadas = leer(nombre)
            mayores_solos = [c for c in casos if c.edad >= umbral and not c.acompanante]
            self.assertEqual(len(mayores_solos), q["aviso"]["mayor_sin_acompanante"])
            # La costumbre del aviso se refleja tal cual en las esperadas.
            for caso in casos:
                self.assertEqual(esperadas[caso.id]["ficha"]["aviso_al_voluntariado"], caso in mayores_solos)
            fichas = [e["ficha"] for e in esperadas.values()]
            self.assertGreater(sum(f["box_antes_del_registro"] for f in fichas), 0)
            self.assertGreater(sum(not f["box_antes_del_registro"] for f in fichas), 0)
            self.assertEqual({f["registro"] for f in fichas}, {"definitivo", "provisorio"})
            self.assertEqual({f["circuito"] for f in fichas}, {"ecg", "codigo_acv", "codigo_sepsis", "ninguno"})
            self.assertGreater(sum(bool(e["verificacion"]["prohibidas"]) for e in esperadas.values()), 0)
            self.assertEqual(self.manifiesto["por_carpeta"][nombre]["balance"]["dificultad"],
                             dict(sorted(q["bucket"].items())))


if __name__ == "__main__":
    unittest.main()
