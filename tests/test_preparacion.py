"""Preparación auditada con demostraciones artificiales, nunca entrenamiento."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from admision_tp.preparacion import preparar_corrida, validar_configuracion
from test_conversaciones import registro

RAIZ = Path(__file__).resolve().parents[1]


class PreparacionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((RAIZ / "configuraciones/sft-ejemplo.json").read_text())
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.datos = self.raiz / "datos.jsonl"
        self.datos.write_text(json.dumps(registro()) + "\n", encoding="utf-8")
        self.ruta_config = self.raiz / "config.json"
        self.ruta_config.write_text(json.dumps(self.config), encoding="utf-8")

    def preparar(self, nombre="salida"):
        return preparar_corrida(self.datos, self.ruta_config, self.raiz / nombre)

    def test_snapshots_hashes_y_determinismo(self):
        with patch("admision_tp.preparacion._codigo", return_value={"commit": "a" * 40, "dirty": True}):
            primero, segundo = self.preparar(), self.preparar("otra")
        self.assertEqual(primero, segundo)
        salida = self.raiz / "salida"
        self.assertEqual((salida / "conversaciones.jsonl").read_bytes(), self.datos.read_bytes())
        self.assertEqual(primero["dataset"]["validos"], 1)
        self.assertTrue(primero["codigo"]["dirty"])
        for nombre, sha in primero["archivos"].items():
            self.assertEqual(hashlib.sha256((salida / nombre).read_bytes()).hexdigest(), sha)
        self.assertEqual(json.loads((salida / "manifiesto.json").read_text()), primero)
        self.assertEqual(primero["estado"], "preparada_no_entrenada")

    def test_configuracion_estricta(self):
        for campo, valor in [("semilla", True), ("batch_size", 0), ("epochs", float("nan")),
                             ("learning_rate", float("inf")), ("revision", "main"),
                             ("hf_repo", "../privado"), ("wandb_entity", []),
                             ("max_seq_length", False)]:
            with self.subTest(campo=campo), self.assertRaises(ValueError):
                config = copy.deepcopy(self.config)
                config[campo] = valor
                validar_configuracion(config)
        with self.assertRaises(ValueError):
            validar_configuracion({**self.config, "WANDB_API_KEY": "no-admitida"})

    def test_adapter_inicial_opcional_con_revision_inmutable(self):
        inicial = {"repo": "FranCalveyra/tp-admision", "revision": "d" * 40, "subcarpeta": "final"}
        validar_configuracion({**self.config, "adapter_inicial": inicial})
        for malo in ({**inicial, "revision": "main"}, {"repo": "x/y", "revision": "d" * 40},
                     {**inicial, "repo": "../privado"}, {**inicial, "subcarpeta": ""}, None):
            with self.subTest(adapter_inicial=malo), self.assertRaises(ValueError):
                validar_configuracion({**self.config, "adapter_inicial": malo})

    def test_invalido_no_deja_salida_ni_temporales(self):
        for dato in ("", "{malformado", json.dumps({**registro(), "origen": "practica"})):
            self.datos.write_text(dato, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.preparar()
            self.assertFalse((self.raiz / "salida").exists())
            self.assertEqual(list(self.raiz.glob(".preparacion-*")), [])

    def test_no_pisa_salida_archivo_directorio_o_symlink(self):
        salida = self.raiz / "salida"
        salida.mkdir()
        with self.assertRaises(ValueError):
            self.preparar()
        salida.rmdir()
        salida.write_text("conservar")
        with self.assertRaises(ValueError):
            self.preparar()
        self.assertEqual(salida.read_text(), "conservar")
        salida.unlink()
        salida.symlink_to(self.raiz / "inexistente")
        with self.assertRaises(ValueError):
            self.preparar()

    def test_cli_y_estado_git(self):
        comando = [sys.executable, "-m", "scripts.preparar_entrenamiento", "--datos", str(self.datos),
                   "--configuracion", str(self.ruta_config), "--salida", str(self.raiz / "cli")]
        resultado = subprocess.run(comando, capture_output=True, text=True, cwd=RAIZ)
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        manifiesto = json.loads((self.raiz / "cli/manifiesto.json").read_text())
        self.assertRegex(manifiesto["codigo"]["commit"], r"^[0-9a-f]{40}$")
        self.assertIs(type(manifiesto["codigo"]["dirty"]), bool)
        self.assertEqual(subprocess.run(comando, capture_output=True).returncode, 1)

    def test_config_invalida_no_expone_contenido(self):
        self.ruta_config.write_text('{"secreto":INVALIDO}')
        with self.assertRaises(ValueError) as error:
            self.preparar()
        self.assertNotIn("secreto", str(error.exception))
        self.assertFalse((self.raiz / "salida").exists())


if __name__ == "__main__":
    unittest.main()
