"""Candidatos de prompt, configuración de evaluación y ranking (issue #13, fase 1), sin GPU."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import tempfile
import unittest

from admision.comparar import CONTROLES
from admision.entorno import MAX_LLAMADAS
from admision_tp.agente_prompt import max_turnos_para
from admision_tp.experimentos_prompt import ConfiguracionEvaluacionPrompt, ejecutar_evaluacion_prompt
from admision_tp.prompts import PromptSpec, cargar_prompt_baseline
from admision_tp.seleccion_prompts import cargar_corrida, corridas_en, rankear
from scripts.rankear_prompts import main as rankear_main

RAIZ = Path(__file__).resolve().parents[1]
PROMPTS = RAIZ / "configuraciones/prompts"
CANDIDATOS = ("corto-v1", "procedimiento-v1", "procedimiento-costumbres-v1", "procedimiento-costumbres-corto-v1")
CONFIG = RAIZ / "configuraciones/evaluacion-prompt-kaggle-t4-l4.json"
CONFIG_PREFIJOS = RAIZ / "configuraciones/evaluacion-prompt-kaggle-t4-l4-prefijos.json"
DEV = RAIZ / "datos/prompt-dev/v1"


def candidato(nombre):
    return PromptSpec.cargar(PROMPTS / f"{nombre}.json")


def digitos_libres(texto):
    """Dígitos fuera de códigos y rangos de secciones, el id de ejemplo y las «2 secciones» del buscador
    que describe el baseline (es el k del buscador, no un parámetro de las normas)."""
    texto = re.sub(r"NOR-[A-Z]{3}-\d{2}( a (NOR-[A-Z]{3}-)?\d{2})?", "", texto)
    texto = texto.replace("P-1234", "").replace("las 2 secciones", "")
    texto = re.sub(r"(?m)^\d\. |\b\d\) ", "", texto)  # números de paso del procedimiento
    return re.findall(r"\d+", texto)


class Candidatos(unittest.TestCase):
    def test_cargan_con_hash_valido_y_mantienen_el_formato(self):
        base = cargar_prompt_baseline()
        for nombre in CANDIDATOS:
            with self.subTest(nombre=nombre):
                spec = candidato(nombre)
                self.assertEqual(spec.nombre, nombre)
                self.assertNotEqual(spec.system_prompt, base.system_prompt)
                for campo in ("mensaje_inicial", "formato_observacion", "mensaje_json_invalido"):
                    self.assertEqual(getattr(spec, campo), getattr(base, campo))

    def test_sin_valores_de_parametros_y_umbral_de_costumbre_tomado_de_los_datos(self):
        umbral = str(json.loads((RAIZ / "historial_costumbres.json").read_text(encoding="utf-8"))
                     ["costumbres"][0]["umbral_edad"])
        self.assertEqual(digitos_libres(cargar_prompt_baseline().system_prompt), [])
        for nombre in CANDIDATOS:
            with self.subTest(nombre=nombre):
                esperado = [umbral] if "costumbres" in nombre else []
                self.assertEqual(digitos_libres(candidato(nombre).system_prompt), esperado)

    def test_ablacion_y_contenido(self):
        base = cargar_prompt_baseline().system_prompt
        corto, b, c, d = (candidato(n).system_prompt for n in CANDIDATOS)
        self.assertLess(len(corto), len(base))
        self.assertTrue(b.startswith(base))
        self.assertTrue(c.startswith(b))
        self.assertLess(len(d), len(c))
        for texto in (b, c, d):
            for palabra in ("Procedimiento", "box", "circuito", "cobertura", "Verificá", "ART"):
                self.assertIn(palabra, texto)
        for texto in (c, d):
            self.assertIn("acompañante", texto)
            self.assertIn("solo la prepaga", texto)
            self.assertIn("sin_cobertura", texto)
        for texto in (corto, b, c, d):
            for herramienta in ("leer_triage", "preguntar", "verificar_afiliacion", "buscar_normativa",
                                "registrar_decision", "aviso_al_voluntariado", "entidad_a_notificar"):
                self.assertIn(herramienta, texto)


class ConfiguracionCandidatos(unittest.TestCase):
    def test_como_la_entrega_en_t4_con_lote_4(self):
        config = ConfiguracionEvaluacionPrompt.cargar(CONFIG)
        self.assertEqual(config.max_turnos, max_turnos_para(MAX_LLAMADAS, config.max_sin_json))
        self.assertEqual((config.lote, config.dtype, config.device_map, config.buscador_variante),
                         (4, "float16", "auto", "baseline"))
        prefijos = ConfiguracionEvaluacionPrompt.cargar(CONFIG_PREFIJOS)
        self.assertEqual({**prefijos.como_dict(), "buscador_variante": "baseline"}, config.como_dict())
        self.assertEqual(prefijos.buscador_variante, "prefijos")


class TokenizadorBytes:
    chat_template = "plantilla de prueba"

    def encode(self, texto, add_special_tokens):
        return list(texto.encode())

    def apply_chat_template(self, mensajes, **opciones):
        return {"input_ids": list("".join(m["content"] for m in mensajes).encode())}


class ModeloConstante:
    """Lee el triage y registra siempre la misma ficha: no mide calidad, solo la plomería."""

    def __call__(self, conversaciones):
        respuestas = []
        for mensajes in conversaciones:
            paciente = re.search(r"P-\d+", mensajes[1]["content"]).group(0)
            if len(mensajes) == 2:
                respuestas.append(json.dumps({"herramienta": "leer_triage", "argumentos": {"paciente_id": paciente}}))
            else:
                respuestas.append(json.dumps({"herramienta": "registrar_decision", "argumentos": {
                    "registro": "provisorio", "numero_documento": None, "box_antes_del_registro": True,
                    "circuito": "codigo_sepsis", "plazo_circuito_min": 30, "cobertura": "sin_cobertura",
                    "entidad_a_notificar": None, "aviso_al_voluntariado": False}}))
        return respuestas


def correr(salida, prompt, carpeta, proposito="evaluacion"):
    return ejecutar_evaluacion_prompt(
        prompt=PROMPTS / f"{prompt}.json", configuracion=CONFIG,
        casos=DEV / "entradas" / carpeta / "casos.jsonl",
        esperadas=DEV / "esperadas" / f"esperadas-{carpeta}.jsonl",
        normas=DEV / "entradas" / carpeta / "normas", salida=salida,
        modelo_lote=ModeloConstante(), tokenizer=TokenizadorBytes(), proposito=proposito)


class RankingConCorridasDelRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.tmp.name) / "sesion"
        cls.raiz.mkdir()
        for prompt in ("baseline", "corto-v1"):
            for carpeta in ("n1", "n2"):
                correr(cls.raiz / f"{prompt}-{carpeta}", prompt, carpeta)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mismo_puntaje_desempata_por_tokens_del_sistema(self):
        filas = rankear([cargar_corrida(r) for r in corridas_en([self.raiz])])
        self.assertEqual([f["nombre"] for f in filas], ["corto-v1", "baseline"])
        self.assertEqual(filas[0]["resueltos"], filas[1]["resueltos"])
        self.assertEqual(filas[0]["casos"], 40)
        self.assertEqual(set(filas[0]["controles"]), set(CONTROLES))
        self.assertLess(filas[0]["tokens_system"], filas[1]["tokens_system"])

    def test_cli_rechaza_corridas_no_comparables_y_conjuntos_incompletos(self):
        with tempfile.TemporaryDirectory() as tmp:
            plomeria = Path(tmp) / "plomeria"
            correr(plomeria, "baseline", "n1", proposito="plomeria")
            for rutas in ([self.raiz, plomeria], [self.raiz / "baseline-n1", self.raiz / "corto-v1-n1",
                                                  self.raiz / "corto-v1-n2"]):
                with self.subTest(rutas=rutas), redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(rankear_main([str(r) for r in rutas]), 1)
                self.assertIn("rechazado", error.getvalue())
        with redirect_stdout(io.StringIO()) as salida:
            self.assertEqual(rankear_main([str(self.raiz)]), 0)
        self.assertIn("| 1 | corto-v1 |", salida.getvalue())

    def test_rechaza_resultados_alterados(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp) / "alterada"
            correr(corrida, "baseline", "n1")
            with open(corrida / "resultados.jsonl", "a", encoding="utf-8") as archivo:
                archivo.write("\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                cargar_corrida(corrida)


def fila(prompt, conjunto, resueltos, tokens, config=None):
    return {"ruta": f"{prompt}-{conjunto}", "prompt": {"nombre": prompt, "sha256": prompt * 4},
            "configuracion": config or {"lote": 4}, "commit": "c", "conjunto": conjunto, "casos": 20,
            "resueltos": resueltos, "controles": {c: resueltos for c in CONTROLES},
            "tokens_system": tokens, "chat_template": "t"}


class ReglasDeRanking(unittest.TestCase):
    def test_suma_resueltos_sobre_carpetas_antes_que_tokens(self):
        filas = rankear([fila("largo", "n1", 6, 900), fila("largo", "n2", 5, 900),
                         fila("corto", "n1", 9, 400), fila("corto", "n2", 1, 400),
                         fila("medio", "n1", 5, 600), fila("medio", "n2", 6, 600)])
        self.assertEqual([(f["nombre"], f["resueltos"]) for f in filas],
                         [("medio", 11), ("largo", 11), ("corto", 10)])
        self.assertEqual([f["posicion"] for f in filas], [1, 2, 3])

    def test_exige_misma_configuracion_y_una_corrida_por_conjunto(self):
        casos = {
            "configuracion": [fila("a", "n1", 1, 1), fila("a", "n2", 1, 1, {"lote": 1})],
            "faltante": [fila("a", "n1", 1, 1), fila("a", "n2", 1, 1), fila("b", "n1", 1, 1)],
            "repetida": [fila("a", "n1", 1, 1), fila("a", "n1", 2, 1)],
        }
        for motivo, corridas in casos.items():
            with self.subTest(motivo=motivo), self.assertRaises(ValueError):
                rankear(corridas)


if __name__ == "__main__":
    unittest.main()


class CandidatoFormato(unittest.TestCase):
    """#15: el formato de los mensajes, aislado del mensaje de sistema (que queda igual a procedimiento-v1)."""

    def test_solo_cambia_el_formato_y_sigue_sin_numeros_de_normas(self):
        base, spec = candidato("procedimiento-v1"), candidato("procedimiento-formato-v1")
        self.assertEqual(spec.nombre, "procedimiento-formato-v1")
        self.assertEqual((spec.system_prompt, spec.mensaje_inicial), (base.system_prompt, base.mensaje_inicial))
        self.assertNotEqual(spec.formato_observacion, base.formato_observacion)
        self.assertNotEqual(spec.mensaje_json_invalido, base.mensaje_json_invalido)
        self.assertEqual(digitos_libres(spec.mensaje_json_invalido), [])
        self.assertEqual(spec.observacion_para("leer_triage", "Nota."), "[leer_triage]\nNota.")
        self.assertLess(len(spec.observacion_para("x", "")), len(base.observacion_para("x", "")))
