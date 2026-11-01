"""Ejemplos artificiales para validar formato, no datos para entrenar."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from admision_tp.conversaciones import validar_archivo, validar_registro


def registro():
    ficha = dict(registro="provisorio", numero_documento=None, box_antes_del_registro=False,
                 circuito="ninguno", plazo_circuito_min=None, cobertura="sin_cobertura",
                 entidad_a_notificar=None, aviso_al_voluntariado=False)
    return {"version": 1, "id": "s-001", "origen": "sintetico", "referencias": ["generador:semilla-42"],
            "messages": [{"role": "system", "content": "Instrucción de ejemplo"},
                         {"role": "user", "content": "Paciente sintético"},
                         {"role": "assistant", "content": json.dumps({"herramienta": "leer_triage",
                                                                            "argumentos": {"paciente_id": "P-1"}})},
                         {"role": "user", "content": "Resultado de leer_triage:\nNota sintética"},
                         {"role": "assistant", "content": json.dumps({"herramienta": "registrar_decision",
                                                                            "argumentos": ficha})}]}


class ConversacionesTest(unittest.TestCase):
    def test_valido_preserva_contenido(self):
        dato = registro()
        antes = copy.deepcopy(dato)
        validar_registro(dato)
        self.assertEqual(dato, antes)

    def test_rechaza_formato_y_procedencia(self):
        cambios = [("version", True), ("origen", "practica"), ("referencias", []),
                   ("referencias", ["practica/casos.jsonl"]), ("id", "test-001"), ("messages", [])]
        for clave, valor in cambios:
            with self.subTest(clave=clave, valor=valor), self.assertRaises(ValueError):
                dato = registro()
                dato[clave] = valor
                validar_registro(dato)

    def test_acciones_malformadas(self):
        acciones = ['texto {"herramienta":"leer_triage"}', '{"herramienta": "otra", "argumentos": {}}',
                    '{"herramienta":"leer_triage","argumentos":{"paciente_id":1}}',
                    '{"herramienta":"leer_triage","herramienta":"buscar_normativa","argumentos":{}}',
                    '{"herramienta":"preguntar","argumentos":{"paciente_id":"P-1","tema":"otro","a_quien":"paciente"}}']
        for accion in acciones:
            with self.subTest(accion=accion), self.assertRaises(ValueError):
                dato = registro()
                dato["messages"][2]["content"] = accion
                validar_registro(dato)

    def test_tipos_ficha(self):
        for clave, valor in [("aviso_al_voluntariado", "false"), ("plazo_circuito_min", True),
                             ("numero_documento", 123), ("circuito", []), ("fundamentos", "NOR-ADM-01")]:
            with self.subTest(clave=clave), self.assertRaises(ValueError):
                dato = registro()
                accion = json.loads(dato["messages"][-1]["content"])
                accion["argumentos"][clave] = valor
                dato["messages"][-1]["content"] = json.dumps(accion)
                validar_registro(dato)

    def test_observacion_terminal_y_limite(self):
        for modo in ("observacion", "rol", "terminal", "limite"):
            dato = registro()
            if modo == "observacion":
                dato["messages"][3]["content"] = "Resultado de preguntar:\nNota"
            elif modo == "rol":
                dato["messages"][3]["role"] = "assistant"
            elif modo == "terminal":
                dato["messages"][2] = copy.deepcopy(dato["messages"][-1])
            else:
                dato["messages"] = dato["messages"][:2] + dato["messages"][2:4] * 40 + dato["messages"][-1:]
            with self.subTest(modo=modo), self.assertRaises(ValueError):
                validar_registro(dato)

    def test_archivo_hash_duplicados_json_y_cli(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "conversaciones.jsonl"
            dato = json.dumps(registro(), ensure_ascii=False) + "\n"
            ruta.write_text(dato, encoding="utf-8")
            informe = validar_archivo(ruta)
            self.assertEqual(informe["validos"], 1)
            self.assertEqual(informe["sha256"], hashlib.sha256(ruta.read_bytes()).hexdigest())
            self.assertFalse(informe["errores"])
            resultado = subprocess.run([sys.executable, "-m", "scripts.validar_conversaciones", str(ruta)],
                                       capture_output=True, text=True)
            self.assertEqual(resultado.returncode, 0, resultado.stderr)
            self.assertEqual(json.loads(resultado.stdout), informe)
            ruta.write_text(dato + dato + "{\"sensible\":INVALIDO}\n", encoding="utf-8")
            informe = validar_archivo(ruta)
            self.assertEqual([e["linea"] for e in informe["errores"]], [2, 3])
            self.assertNotIn("sensible", json.dumps(informe))
            resultado = subprocess.run([sys.executable, "-m", "scripts.validar_conversaciones", str(ruta)], capture_output=True)
            self.assertEqual(resultado.returncode, 1)

    def test_vacio_utf8_y_lineas_vacias(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "datos.jsonl"
            for contenido in (b"", b"\xff", b"\n"):
                ruta.write_bytes(contenido)
                self.assertTrue(validar_archivo(ruta)["errores"])


if __name__ == "__main__":
    unittest.main()
