"""Contrato reproducible del entorno; no instala dependencias pesadas."""

from pathlib import Path
import tomllib
import unittest

RAIZ = Path(__file__).resolve().parents[1]


class EntornoTests(unittest.TestCase):
    def test_cpu_no_requiere_modelo_y_no_construye_proyecto(self):
        proyecto = tomllib.loads((RAIZ / "pyproject.toml").read_text())
        self.assertFalse(proyecto["tool"]["uv"]["package"])
        self.assertEqual(proyecto["project"]["dependencies"], ["pyyaml==6.0.3"])
        self.assertTrue(all("sys_platform == 'linux'" in d
                            for d in proyecto["dependency-groups"]["inferencia"]))

    def test_pines_directos_coinciden_con_export_y_lock(self):
        proyecto = tomllib.loads((RAIZ / "pyproject.toml").read_text())
        lock = tomllib.loads((RAIZ / "uv.lock").read_text())
        versiones = {(p["name"], p["version"]) for p in lock["package"]}
        exportado = (RAIZ / "requirements-inferencia.txt").read_text().splitlines()
        for dependencia in (proyecto["project"]["dependencies"] +
                            proyecto["dependency-groups"]["inferencia"]):
            pin = dependencia.split(";")[0].strip()
            self.assertIn(tuple(pin.split("==")), versiones)
            self.assertTrue(any(l.split(";")[0].strip() == pin for l in exportado))

    def test_plantilla_secretos_sin_valores(self):
        variables = dict(linea.split("=", 1) for linea in
                         (RAIZ / ".env.example").read_text().splitlines()
                         if linea and not linea.startswith("#"))
        self.assertEqual(variables["HF_TOKEN"], "")
        self.assertEqual(variables["WANDB_API_KEY"], "")
        self.assertEqual(variables["BUSCADOR_VARIANTE"], "baseline")


if __name__ == "__main__":
    unittest.main()
