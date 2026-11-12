"""Runner de prompt con datos y generación artificiales; no mide calidad real."""

import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from admision.correr import EsperadasAlAlcance
from admision_tp import experimentos_prompt
from admision_tp.evaluacion_prompts import clave_comparacion, clave_de_manifiesto
from admision_tp.experimentos_prompt import (
    ConfiguracionEvaluacionPrompt,
    _asegurar_token_hf,
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


class ModeloNoDebeCargarse:
    def __call__(self, conversaciones):
        raise AssertionError("el modelo no debía usarse")


class ModeloContextoExcedido:
    """Falla como ``ModeloLocal`` cuando el prompt no entra en ``max_contexto``."""

    def __call__(self, conversaciones):
        raise ValueError("Contexto excedido: no se recorta la conversación silenciosamente")


def cargador_falso(modelo):
    def cargar(config):
        modelo.procesador = TokenizadorArtificial()
        return modelo
    return cargar


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

    def ejecutar(self, nombre="corrida", *, casos=None, esperadas=None, modelo=None, **extra):
        salida = self.raiz / nombre
        return ejecutar_evaluacion_prompt(
            prompt=self.prompt, configuracion=self.config, casos=casos or self.casos,
            esperadas=esperadas or self.esperadas, normas=RAIZ / "normas/original", salida=salida,
            modelo_lote=modelo or ModeloGuionado(salida / "entradas/esperadas.jsonl"),
            tokenizer=TokenizadorArtificial(), **extra)

    def manifiesto(self, nombre):
        return json.loads((self.raiz / nombre / "manifiesto.json").read_text())

    def args_cli(self, nombre, *extra):
        return ["--prompt", str(self.prompt), "--configuracion", str(self.config),
                "--casos", str(self.casos), "--esperadas", str(self.esperadas),
                "--normas", str(RAIZ / "normas/original"),
                "--salida", str(self.raiz / nombre), *extra]

    def practica_simulada(self):
        """Carpeta con la forma de practica/: casos y esperadas-*.jsonl juntos."""
        carpeta = self.raiz / "practica"
        carpeta.mkdir()
        (carpeta / "casos.jsonl").write_bytes(self.casos.read_bytes())
        (carpeta / "esperadas-original.jsonl").write_bytes(self.esperadas.read_bytes())
        return carpeta

    def test_runner_separa_esperadas_y_registra_snapshots_hashes_y_metricas(self):
        manifiesto = self.ejecutar()
        salida = self.raiz / "corrida"
        self.assertEqual(manifiesto["estado"], "completada")
        self.assertEqual(manifiesto["proposito"], "evaluacion")
        self.assertIs(manifiesto["comparable"], True)
        self.assertEqual(manifiesto["guardas"]["esperadas_al_alcance"], "verificada_en_origen")
        self.assertEqual(clave_de_manifiesto(manifiesto), clave_comparacion(manifiesto["metricas"]))
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

    def test_adapter_opcional_llega_al_modelo_y_queda_en_la_configuracion(self):
        dato = json.loads(self.config.read_text())
        self.assertNotIn("adapter", ConfiguracionEvaluacionPrompt.desde_dict(dato).como_dict())
        adapter = {"repo": "FranCalveyra/tp-admision", "revision": "a" * 40, "subcarpeta": "final"}
        config = ConfiguracionEvaluacionPrompt.desde_dict({**dato, "adapter": adapter})
        modelo = config.configuracion_modelo()
        self.assertEqual((modelo.adapter_repo, modelo.adapter_revision, modelo.adapter_subcarpeta),
                         ("FranCalveyra/tp-admision", "a" * 40, "final"))
        self.assertEqual(config.como_dict()["adapter"], adapter)
        self.assertEqual(ConfiguracionEvaluacionPrompt.desde_dict(config.como_dict()), config)
        for malo in ({**adapter, "revision": "main"}, {"repo": "x/y"}, {**adapter, "extra": 1}, "x/y"):
            with self.subTest(adapter=malo), self.assertRaises(ValueError):
                ConfiguracionEvaluacionPrompt.desde_dict({**dato, "adapter": malo})

    def test_cli_help_no_carga_modelo(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as salida:
            main(["--help"])
        self.assertEqual(salida.exception.code, 0)

    def test_rechaza_practica_del_repo_antes_de_copiar(self):
        practica = RAIZ / "practica"
        combinaciones = {
            "ambos": (practica / "casos.jsonl", practica / "esperadas-original.jsonl"),
            "solo_esperadas": (self.casos, practica / "esperadas-original.jsonl"),
        }
        for nombre, (casos, esperadas) in combinaciones.items():
            with self.subTest(nombre):
                with self.assertRaisesRegex(ValueError, "practica/"):
                    self.ejecutar(nombre, casos=casos, esperadas=esperadas,
                                  modelo=ModeloNoDebeCargarse())
                manifiesto = self.manifiesto(nombre)
                self.assertEqual((manifiesto["estado"], manifiesto["proposito"]),
                                 ("fallida", "evaluacion"))
                self.assertIs(manifiesto["comparable"], False)
                self.assertEqual(manifiesto["entradas"], {})
                self.assertEqual(list((self.raiz / nombre / "entradas").iterdir()), [])

    def test_rechaza_practica_a_traves_de_symlink(self):
        enlace = self.raiz / "enlace"
        enlace.symlink_to(RAIZ / "practica", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "practica/"):
            self.ejecutar(casos=enlace / "casos.jsonl", modelo=ModeloNoDebeCargarse())

    def test_chequeo_del_kit_corre_sobre_la_carpeta_original(self):
        # Una esperada con nombre del kit junto a los casos originales: la copia no la tendría.
        (self.raiz / "esperadas-original.jsonl").write_bytes(self.esperadas.read_bytes())
        with self.assertRaises(EsperadasAlAlcance):
            self.ejecutar(modelo=ModeloNoDebeCargarse())
        manifiesto = self.manifiesto("corrida")
        self.assertEqual(manifiesto["estado"], "fallida")
        self.assertEqual(manifiesto["error"]["tipo"], "EsperadasAlAlcance")
        self.assertFalse((self.raiz / "corrida/entradas/esperadas.jsonl").exists())

    def test_plomeria_explicita_acepta_practica_y_no_es_comparable(self):
        practica = self.practica_simulada()
        casos, esperadas = practica / "casos.jsonl", practica / "esperadas-original.jsonl"
        with mock.patch.object(experimentos_prompt, "_PRACTICA", practica):
            with self.assertRaisesRegex(ValueError, "--solo-plomeria"):
                self.ejecutar("evaluacion", casos=casos, esperadas=esperadas,
                              modelo=ModeloNoDebeCargarse())
            manifiesto = self.ejecutar("plomeria", casos=casos, esperadas=esperadas,
                                       proposito="plomeria")
        self.assertEqual((manifiesto["estado"], manifiesto["proposito"]), ("completada", "plomeria"))
        self.assertIs(manifiesto["comparable"], False)
        self.assertEqual(manifiesto["guardas"], {
            "datos_practica": ["casos", "esperadas"], "esperadas_al_alcance": "omitida_plomeria"})
        self.assertIsNone(manifiesto["metricas"]["clave_comparacion"])
        metricas = json.loads((self.raiz / "plomeria/metricas.json").read_text())
        self.assertIs(metricas["comparable"], False)
        with self.assertRaisesRegex(ValueError, "no comparable"):
            clave_de_manifiesto(manifiesto)
        with self.assertRaisesRegex(ValueError, "no comparable"):
            clave_comparacion(metricas)

    def test_contexto_excedido_queda_con_error_y_sin_clave(self):
        with redirect_stderr(io.StringIO()):
            manifiesto = self.ejecutar(modelo=ModeloContextoExcedido())
        self.assertEqual(manifiesto["estado"], "completada_con_error")
        self.assertIs(manifiesto["comparable"], False)
        self.assertEqual(manifiesto["metricas"]["motivo_no_comparable"], "estado completada_con_error")
        self.assertIsNone(manifiesto["metricas"]["clave_comparacion"])
        self.assertEqual(manifiesto["metricas"]["resueltos_9_de_9"], 0)
        tiempos = json.loads((self.raiz / "corrida/tiempos.json").read_text())
        self.assertIn("Contexto excedido", tiempos["error"])
        with self.assertRaisesRegex(ValueError, "no comparable"):
            clave_de_manifiesto(manifiesto)
        with self.assertRaisesRegex(ValueError, "no comparable"):
            clave_comparacion(manifiesto["metricas"])

    def test_cli_sale_con_error_si_el_agente_fallo(self):
        stdout = io.StringIO()
        with mock.patch.object(experimentos_prompt, "_cargar_modelo_real",
                               cargador_falso(ModeloContextoExcedido())), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            codigo = main(self.args_cli("cli"))
        self.assertEqual(codigo, 1)
        salida = json.loads(stdout.getvalue())
        self.assertEqual((salida["estado"], salida["comparable"]), ("completada_con_error", False))

    def test_cli_solo_plomeria_marca_el_proposito(self):
        practica = self.practica_simulada()
        self.casos, self.esperadas = practica / "casos.jsonl", practica / "esperadas-original.jsonl"
        modelo = ModeloGuionado(self.raiz / "cli/entradas/esperadas.jsonl")
        with mock.patch.object(experimentos_prompt, "_PRACTICA", practica), \
                mock.patch.object(experimentos_prompt, "_cargar_modelo_real", cargador_falso(modelo)), \
                redirect_stdout(io.StringIO()):
            codigo = main(self.args_cli("cli", "--solo-plomeria"))
        self.assertEqual(codigo, 0)
        manifiesto = self.manifiesto("cli")
        self.assertEqual((manifiesto["proposito"], manifiesto["comparable"]), ("plomeria", False))

    def test_cli_casos_malformados_falla_limpio_y_registra(self):
        malformados = {
            "campo_faltante": json.dumps({"id": "x"}) + "\n",
            "json_roto": "{no es json\n",
            "campo_desconocido": json.dumps({**caso(), "extra": 1}) + "\n",
        }
        for nombre, contenido in malformados.items():
            with self.subTest(nombre):
                self.casos.write_text(contenido, encoding="utf-8")
                stderr = io.StringIO()
                with mock.patch.object(experimentos_prompt, "_cargar_modelo_real",
                                       side_effect=AssertionError("no debía cargar")), \
                        redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    codigo = main(self.args_cli(nombre))
                self.assertEqual(codigo, 1)
                self.assertIn("casos con formato inválido", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                manifiesto = self.manifiesto(nombre)
                self.assertEqual(manifiesto["estado"], "fallida")
                self.assertEqual(manifiesto["error"]["tipo"], "ValueError")

    def test_esperadas_malformadas_fallan_con_valueerror(self):
        self.esperadas.write_text(json.dumps({"sin_caso": True}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "esperadas con formato inválido"):
            self.ejecutar()
        self.assertEqual(self.manifiesto("corrida")["estado"], "fallida")

    def test_configuracion_kaggle_t4_solo_cambia_dtype_y_device_map(self):
        a100 = json.loads((RAIZ / "configuraciones/evaluacion-prompt-ejemplo.json").read_text())
        t4 = json.loads((RAIZ / "configuraciones/evaluacion-prompt-kaggle-t4.json").read_text())
        self.assertEqual((t4["dtype"], t4["device_map"]), ("float16", "auto"))
        self.assertEqual((a100["dtype"], a100["device_map"]), ("bfloat16", "cuda:0"))
        self.assertEqual({k: v for k, v in t4.items() if k not in ("dtype", "device_map")},
                         {k: v for k, v in a100.items() if k not in ("dtype", "device_map")})
        modelo = ConfiguracionEvaluacionPrompt.cargar(
            RAIZ / "configuraciones/evaluacion-prompt-kaggle-t4.json").configuracion_modelo()
        self.assertEqual((modelo.dtype, modelo.device_map), ("float16", "auto"))


class TokenHFTest(unittest.TestCase):
    def test_usa_variable_de_entorno_sin_consultar_secretos(self):
        entorno = {"HF_TOKEN": "hf_entorno"}
        _asegurar_token_hf(entorno, obtener_secreto=mock.Mock(side_effect=AssertionError))
        self.assertEqual(entorno["HF_TOKEN"], "hf_entorno")

    def test_toma_el_kaggle_secret_si_falta_la_variable(self):
        entorno, secreto = {}, mock.Mock(return_value="hf_kaggle")
        _asegurar_token_hf(entorno, obtener_secreto=secreto)
        secreto.assert_called_once_with("HF_TOKEN")
        self.assertEqual(entorno["HF_TOKEN"], "hf_kaggle")

    def test_falta_de_token_falla_claro_sin_mostrar_detalles(self):
        for efecto in (ImportError("sin kaggle_secrets"), RuntimeError("secreto sin adjuntar"), None):
            with self.subTest(efecto=efecto):
                obtener = mock.Mock(side_effect=efecto) if efecto else mock.Mock(return_value="")
                with self.assertRaisesRegex(RuntimeError, "Falta HF_TOKEN") as error:
                    _asegurar_token_hf({"HF_TOKEN": ""}, obtener_secreto=obtener)
                self.assertNotIn("kaggle_secrets", str(error.exception))


if __name__ == "__main__":
    unittest.main()
