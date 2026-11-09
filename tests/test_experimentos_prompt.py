"""Runner de prompt con datos y generación artificiales; no mide calidad real."""

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from admision_tp.experimentos_prompt import (
    ConfiguracionEvaluacionPrompt,
    ejecutar_evaluacion_prompt,
    main,
)


RAIZ = Path(__file__).resolve().parents[1]


class TokenizadorArtificial:
    chat_template = "chat artificial de evaluación"

    def encode(self, texto, add_special_tokens):
        return list(texto.encode())

    def apply_chat_template(self, mensajes, **opciones):
        texto = "".join(m["content"] for m in mensajes) + "<assistant>"
        return {"input_ids": list(texto.encode())}


def caso():
    return {
        "id": "externo-1", "paciente_id": "P-EXT-1", "nombre": "Paciente Externo",
        "edad": 30, "nivel": 5, "motivo": "dolor", "nota": "Nota externa.",
        "sepsis": "no", "llegada": "por sus propios medios", "documento_tipo": "DNI",
        "documento_numero": "30111222", "credenciales": [], "puede_responder": True,
        "no_responde_porque": None, "acompanante": None, "trabajo": None,
        "declarada": ["obra_social", "Obra Social Bermejo", "A-1"], "art": None,
        "afiliaciones": {"Obra Social Bermejo|A-1": True},
    }


def esperada():
    return {"caso": "externo-1", "ficha": {
        "registro": "definitivo", "numero_documento": "30111222",
        "box_antes_del_registro": False, "circuito": "ninguno",
        "plazo_circuito_min": None, "cobertura": "obra_social",
        "entidad_a_notificar": "Obra Social Bermejo", "aviso_al_voluntariado": False,
    }, "verificacion": {"requerida": ["Obra Social Bermejo", "A-1"], "prohibidas": []}}


class ModeloGuionado:
    def __init__(self, snapshot_esperadas):
        self.snapshot_esperadas = snapshot_esperadas

    def __call__(self, conversaciones):
        # La respuesta esperada no se copia hasta que el agente termina de jugar.
        assert not self.snapshot_esperadas.exists()
        respuestas = []
        for mensajes in conversaciones:
            turno = sum(m["role"] == "assistant" for m in mensajes)
            acciones = [
                {"herramienta": "leer_triage", "argumentos": {"paciente_id": "P-EXT-1"}},
                {"herramienta": "verificar_afiliacion",
                 "argumentos": {"entidad": "Obra Social Bermejo", "numero_afiliado": "A-1"}},
                {"herramienta": "registrar_decision", "argumentos": {
                    **esperada()["ficha"], "fundamentos": ["NOR-ADM-01"]}},
            ]
            respuestas.append(json.dumps(acciones[turno]))
        return respuestas


class ExperimentosPromptTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.prompt = self.raiz / "prompt.json"
        self.prompt.write_bytes((RAIZ / "configuraciones/prompts/baseline.json").read_bytes())
        self.config = self.raiz / "config.json"
        self.config.write_bytes((RAIZ / "configuraciones/evaluacion-prompt-ejemplo.json").read_bytes())
        self.casos, self.esperadas = self.raiz / "casos-externos.jsonl", self.raiz / "respuestas.jsonl"
        self.casos.write_text(json.dumps(caso()) + "\n", encoding="utf-8")
        self.esperadas.write_text(json.dumps(esperada()) + "\n", encoding="utf-8")

    def ejecutar(self, nombre="corrida"):
        salida = self.raiz / nombre
        return ejecutar_evaluacion_prompt(
            prompt=self.prompt, configuracion=self.config, casos=self.casos,
            esperadas=self.esperadas, normas=RAIZ / "normas/original", salida=salida,
            modelo_lote=ModeloGuionado(salida / "entradas/esperadas.jsonl"),
            tokenizer=TokenizadorArtificial())

    def test_runner_separa_esperadas_y_registra_snapshots_hashes_y_metricas(self):
        manifiesto = self.ejecutar()
        salida = self.raiz / "corrida"
        self.assertEqual(manifiesto["estado"], "completada")
        self.assertEqual(manifiesto["metricas"]["resueltos_9_de_9"], 1)
        self.assertEqual(manifiesto["metricas"]["controles_correctos"], 9)
        self.assertEqual(manifiesto["prompt"]["nombre"], "baseline")
        self.assertEqual(json.loads((salida / "manifiesto.json").read_text()), manifiesto)
        for nombre, origen in (("prompt.json", self.prompt), ("configuracion.json", self.config),
                               ("casos.jsonl", self.casos), ("esperadas.jsonl", self.esperadas)):
            snapshot = salida / "entradas" / nombre
            self.assertEqual(snapshot.read_bytes(), origen.read_bytes())
            self.assertEqual(manifiesto["entradas"][nombre], hashlib.sha256(snapshot.read_bytes()).hexdigest())
        for nombre, sha in manifiesto["artefactos"].items():
            self.assertEqual(hashlib.sha256((salida / nombre).read_bytes()).hexdigest(), sha)
        self.assertIn("parametros.yaml", manifiesto["entradas"]["normas"])
        self.assertIs(type(manifiesto["codigo"]["dirty"]), bool)

    def test_salida_no_se_reutiliza(self):
        primero = self.ejecutar()
        antes = (self.raiz / "corrida/manifiesto.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "ya existe"):
            self.ejecutar()
        self.assertEqual((self.raiz / "corrida/manifiesto.json").read_bytes(), antes)
        self.assertEqual(json.loads(antes), primero)

    def test_fallo_de_configuracion_queda_registrado_sin_exponer_contenido(self):
        self.config.write_text('{"secreto": "no registrar"}', encoding="utf-8")
        salida = self.raiz / "fallida"
        with self.assertRaises(ValueError):
            ejecutar_evaluacion_prompt(
                prompt=self.prompt, configuracion=self.config, casos=self.casos,
                esperadas=self.esperadas, normas=RAIZ / "normas/original", salida=salida,
                modelo_lote=object(), tokenizer=TokenizadorArtificial())
        manifiesto = json.loads((salida / "manifiesto.json").read_text())
        self.assertEqual(manifiesto["estado"], "fallida")
        self.assertNotIn("secreto", manifiesto["error"]["mensaje"])
        self.assertIn("normas", manifiesto["entradas"])
        self.assertEqual((salida / "entradas/esperadas.jsonl").read_bytes(), self.esperadas.read_bytes())

    def test_configuracion_cerrada_y_sin_adapter(self):
        dato = json.loads(self.config.read_text())
        config = ConfiguracionEvaluacionPrompt.desde_dict(dato)
        self.assertIsNone(config.configuracion_modelo().adapter_repo)
        for cambio in ({"semilla": True}, {"revision": "main"}, {"lote": 0},
                       {"buscador_variante": "otra"}, {"adapter_repo": "x/y"}):
            with self.subTest(cambio=cambio), self.assertRaises(ValueError):
                ConfiguracionEvaluacionPrompt.desde_dict({**dato, **cambio})

    def test_cli_help_no_carga_modelo(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as salida:
            main(["--help"])
        self.assertEqual(salida.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
