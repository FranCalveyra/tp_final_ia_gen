import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from admision_tp.prompts import RUTA_BASELINE, PromptSpec
from infra import kaggle_baseline as kb
from infra import kaggle_evaluacion as ke

RAIZ = Path(__file__).resolve().parents[1]
SHA_BASELINE = PromptSpec.cargar(RUTA_BASELINE).sha256
REVISION = "a" * 40


def plan(*configuraciones, **extra) -> dict:
    return {"version": 1, "nombre": "prueba", "lote": 4, "buscador": "baseline",
            "configuraciones": list(configuraciones), **extra}


def config(etiqueta="sin_entrenar", adapter=None, **extra) -> dict:
    return {"etiqueta": etiqueta, "prompt": "configuraciones/prompts/baseline.json",
            "prompt_sha256": SHA_BASELINE, "adapter": adapter, **extra}


class ValidacionPlanTest(unittest.TestCase):
    def test_plan_actual_es_valido_y_corre_las_tres_versiones(self):
        dato = json.loads((RAIZ / "configuraciones/evaluacion/actual.json").read_text(encoding="utf-8"))
        corridas = ke.validar_plan(dato)
        self.assertEqual([c.carpeta for c in corridas], ["sin_entrenar-original", "sin_entrenar-b", "sin_entrenar-c"])
        self.assertTrue(all(c.cantidad == 50 and c.adapter is None for c in corridas))
        self.assertEqual(corridas[0].prompt["sha256"], SHA_BASELINE)

    def test_plantilla_se_rechaza(self):
        dato = json.loads((RAIZ / "configuraciones/evaluacion/plantilla-cuatro-filas.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "plantilla"):
            ke.validar_plan(dato)

    def test_prompt_alterado_o_hash_fijado_distinto_falla(self):
        datos = json.loads(RUTA_BASELINE.read_text(encoding="utf-8"))
        datos["system_prompt"] += " (alterado sin recalcular el hash)"
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            (raiz / "alterado.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                ke.validar_plan(plan({**config(), "prompt": "alterado.json"}), raiz=raiz)
        with self.assertRaisesRegex(ValueError, "prompt_sha256 no coincide"):
            ke.validar_plan(plan(config(prompt_sha256="0" * 64)))

    def test_adapter_con_revision_mutable_falla(self):
        for adapter in ({"repo": "org/repo", "revision": "main"}, {"repo": "org/repo"}, {"revision": REVISION}):
            with self.subTest(adapter=adapter), self.assertRaisesRegex(ValueError, "adapter|revisión"):
                ke.validar_plan(plan(config("solo_entrenamiento", adapter)))

    def test_etiquetas_repetidas_y_versiones_invalidas_fallan(self):
        with self.assertRaisesRegex(ValueError, "etiqueta repetida"):
            ke.validar_plan(plan(config(), config()))
        with self.assertRaisesRegex(ValueError, "versiones"):
            ke.validar_plan(plan(config(versiones=["original", "original"])))
        with self.assertRaisesRegex(ValueError, "cantidad"):
            ke.validar_plan(plan(config(cantidad=51)))

    def test_plan_invalido_no_crea_salida_ni_corre(self):
        with tempfile.TemporaryDirectory() as carpeta, patch.object(kb, "correr") as correr, \
                patch.object(kb, "describir_maquina", return_value={}):
            ruta = Path(carpeta) / "plan.json"
            ruta.write_text(json.dumps(plan(config(), config())), encoding="utf-8")
            destino = Path(carpeta) / "salida"
            with patch.dict(os.environ, {"HF_TOKEN": "hf_x"}), self.assertRaises(ValueError):
                ke.main(["--plan", str(ruta), "--salida", str(destino)])
            self.assertFalse(destino.exists())
        correr.assert_not_called()


class EjecucionPlanTest(unittest.TestCase):
    def _corridas(self):
        return ke.validar_plan(plan(config(), config("solo_entrenamiento", {"repo": "org/repo", "revision": REVISION},
                                                     versiones=["b"])))

    def test_entorno_por_corrida_no_hereda_de_otra(self):
        vistos = []

        def correr(version, cantidad, destino, dtype, device_map, token):
            vistos.append((destino.name, {k: os.environ.get(k) for k in ke.CONTROLADAS}))
            return 0
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.dict(os.environ, {"ADAPTER_REPO": "otro/repo", "PROMPT_SPEC": "x.json", "MODELO_ID": "otro"}):
            antes = dict(os.environ)
            codigo = ke.ejecutar_plan(self._corridas(), Path(carpeta) / "eval", "float16", "auto", "hf_x", {},
                                      correr=correr)
            self.assertEqual(dict(os.environ), antes)
        self.assertEqual(codigo, 0)
        entornos = dict(vistos)
        base = {"PROMPT_SPEC": "configuraciones/prompts/baseline.json", "MODELO_LOTE": "4",
                "BUSCADOR_VARIANTE": "baseline", "MODELO_ID": None, "MODELO_REVISION": None}
        self.assertEqual(entornos["sin_entrenar-original"],
                         {**base, "ADAPTER_REPO": None, "ADAPTER_REVISION": None, "ADAPTER_SUBCARPETA": None})
        self.assertEqual(entornos["solo_entrenamiento-b"],
                         {**base, "ADAPTER_REPO": "org/repo", "ADAPTER_REVISION": REVISION,
                          "ADAPTER_SUBCARPETA": "final"})
        self.assertEqual(list(entornos), ["sin_entrenar-original", "sin_entrenar-b", "sin_entrenar-c",
                                          "solo_entrenamiento-b"])

    def test_una_corrida_fallida_no_frena_las_demas(self):
        def correr(version, cantidad, destino, dtype, device_map, token):
            if version == "b":
                raise RuntimeError(f"CUDA rota con {token}")
            return 1 if version == "c" else 0
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "eval"
            codigo = ke.ejecutar_plan(ke.validar_plan(plan(config())), destino, "float16", "auto", "hf_x",
                                      {"plan_sha256": "abc"}, correr=correr)
            registro = json.loads((destino / "evaluacion.json").read_text(encoding="utf-8"))
        self.assertEqual(codigo, 1)
        filas = {f["version"]: f for f in registro["corridas"]}
        self.assertEqual({v: (f["estado"], f["returncode"]) for v, f in filas.items()},
                         {"original": ("terminada", 0), "b": ("fallida", None), "c": ("fallida", 1)})
        self.assertEqual(filas["b"]["error"], "RuntimeError: CUDA rota con ***")
        self.assertEqual(registro["plan_sha256"], "abc")
        self.assertIn("fin", registro)

    def test_no_pisa_una_salida_existente(self):
        with tempfile.TemporaryDirectory() as carpeta, patch.object(kb, "correr") as correr:
            with self.assertRaises(FileExistsError):
                ke.ejecutar_plan(self._corridas(), Path(carpeta), "float16", "auto", "hf_x", {})
        correr.assert_not_called()

    def test_con_el_runner_real_cada_runner_json_registra_prompt_y_adapter(self):
        with tempfile.TemporaryDirectory() as carpeta, \
                patch.object(kb, "ejecutar_con_log", return_value=(0, "")), \
                patch.object(kb, "describir_maquina", return_value={}):
            destino = Path(carpeta) / "eval"
            ke.ejecutar_plan(self._corridas(), destino, "float16", "auto", "hf_x", {})
            runners = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
                       for p in destino.glob("*/runner.json")}
        self.assertEqual(len(runners), 4)
        entrenado = runners["solo_entrenamiento-b"]
        self.assertEqual((entrenado["variante"], entrenado["cantidad"]), ("b", 50))
        self.assertEqual((entrenado["configuracion"]["adapter_repo"], entrenado["configuracion"]["adapter_revision"],
                          entrenado["configuracion"]["lote"]), ("org/repo", REVISION, 4))
        self.assertIsNone(runners["sin_entrenar-c"]["configuracion"]["adapter_repo"])
        self.assertEqual(runners["sin_entrenar-c"]["prompt"]["sha256"], SHA_BASELINE)


if __name__ == "__main__":
    unittest.main()
