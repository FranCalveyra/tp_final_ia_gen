"""`scripts.ver_caso`: lo que pide el README para un caso (conversación, búsquedas, ficha y 9 controles)."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from scripts.ver_caso import main

TRAZA = {
    "caso": "pev-001", "segundos": 12.0,
    "mensajes": [
        {"role": "system", "content": "Sos el agente de admisión.\nSegunda línea del sistema."},
        {"role": "user", "content": "Ingresó el paciente P-1."},
        {"role": "assistant", "content": '{"herramienta": "buscar_normativa", "argumentos": {"consulta": "ecg"}}'},
        {"role": "user", "content": "Resultado de buscar_normativa:\n## NOR-TDE-01 · Dolor torácico\n§1. Texto."},
    ],
    "log": [
        {"herramienta": "leer_triage", "argumentos": {"paciente_id": "P-1"}, "observacion": "Nota.", "valida": True},
        {"herramienta": "buscar_normativa", "argumentos": {"consulta": "ecg"},
         "observacion": "## NOR-TDE-01 · Dolor torácico\n§1. Texto.\n\n## NOR-ADM-01 · Registro\n§1. Otro.",
         "valida": True},
    ],
    "decision": {"registro": "provisorio", "circuito": "ecg", "aviso_al_voluntariado": False},
}
RESULTADO = {"caso": "pev-001", "resuelto": False, "parcial": 0.889,
             "controles": {"registro": True, "circuito": True, "cobertura": False}}


class VerCasoTest(unittest.TestCase):
    def correr(self, carpeta, *extra):
        salida = io.StringIO()
        with redirect_stdout(salida):
            codigo = main([str(carpeta), *extra])
        return codigo, salida.getvalue()

    def carpeta(self, raiz, con_resultados=True):
        (raiz / "trazas.jsonl").write_text(json.dumps(TRAZA, ensure_ascii=False) + "\n", encoding="utf-8")
        if con_resultados:
            (raiz / "resultados.jsonl").write_text(json.dumps(RESULTADO) + "\n", encoding="utf-8")
        return raiz

    def test_muestra_conversacion_busquedas_ficha_y_controles(self):
        with tempfile.TemporaryDirectory() as temporal:
            codigo, texto = self.correr(self.carpeta(Path(temporal)))
        self.assertEqual(codigo, 0)
        self.assertIn("[user] Ingresó el paciente P-1.", texto)
        # El sistema se resume: es largo y se repite igual en cada caso.
        self.assertIn("[system] Sos el agente de admisión. (", texto)
        self.assertNotIn("Segunda línea del sistema", texto)
        self.assertIn("«ecg» → NOR-TDE-01 · Dolor torácico; NOR-ADM-01 · Registro", texto)
        self.assertIn('"circuito": "ecg"', texto)
        self.assertIn("✗ cobertura", texto)
        self.assertIn("✓ registro", texto)
        self.assertIn("No resuelto (2/3 controles)", texto)

    def test_sistema_completo_a_pedido_y_sin_resultados_avisa(self):
        with tempfile.TemporaryDirectory() as temporal:
            codigo, texto = self.correr(self.carpeta(Path(temporal), con_resultados=False), "--sistema-completo")
        self.assertEqual(codigo, 0)
        self.assertIn("Segunda línea del sistema", texto)
        self.assertIn("sin resultados.jsonl", texto)

    def test_caso_inexistente_falla(self):
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = self.carpeta(Path(temporal))
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                main([str(carpeta), "--caso", "pev-999"])


if __name__ == "__main__":
    unittest.main()
