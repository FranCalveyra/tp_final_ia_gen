"""Fixtures artificiales: verifican conversión, no calidad de entrenamiento."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from admision_tp.conversion_trazas import convertir_traza


def traza():
    ficha = {"registro": "provisorio", "numero_documento": None,
             "box_antes_del_registro": False, "circuito": "ninguno",
             "plazo_circuito_min": None, "cobertura": "sin_cobertura",
             "entidad_a_notificar": None, "aviso_al_voluntariado": False}
    return {"id": "E-1", "paciente_id": "P-1", "ficha": ficha,
            "pasos": [
                {"orden": 1, "herramienta": "leer_triage", "argumentos": {"paciente_id": "P-1"},
                 "resultado": "Nota sintética\nCon espacios  "},
                {"orden": 2, "herramienta": "buscar_normativa", "argumentos": {"consulta": "registro"},
                 "resultado": "## Norma\nValores: P-ADM-1 = nivel 3.\n"},
                {"orden": 3, "herramienta": "registrar_decision",
                 "argumentos": {**ficha, "fundamentos": ["NOR-ADM-01"]},
                 "resultado": "Ficha registrada."}]}


class ConversionTrazaTest(unittest.TestCase):
    def convertir(self, dato):
        return convertir_traza(dato, origen="sintetico", sistema="Sistema explícito",
                               referencia="entrenamiento/trazas.jsonl")

    def test_conserva_acciones_y_observaciones_omite_solo_resultado_terminal(self):
        dato = traza()
        antes = copy.deepcopy(dato)
        salida = self.convertir(dato)
        self.assertEqual({k: v for k, v in salida.items() if k != "messages"},
                         {"version": 1, "id": "E-1", "origen": "sintetico",
                          "referencias": ["entrenamiento/trazas.jsonl#E-1"]})
        mensajes = salida["messages"]
        self.assertEqual([m["role"] for m in mensajes],
                         ["system", "user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertEqual(mensajes[0]["content"], "Sistema explícito")
        self.assertEqual(mensajes[1]["content"],
                         "Ingresó a la guardia el paciente P-1. Completá su ficha de admisión.")
        self.assertEqual(json.loads(mensajes[2]["content"]),
                         {"herramienta": "leer_triage", "argumentos": {"paciente_id": "P-1"}})
        self.assertEqual(mensajes[3]["content"], "Resultado de leer_triage:\nNota sintética\nCon espacios  ")
        self.assertEqual(json.loads(mensajes[4]["content"]),
                         {"herramienta": "buscar_normativa", "argumentos": {"consulta": "registro"}})
        self.assertEqual(mensajes[5]["content"],
                         "Resultado de buscar_normativa:\n## Norma\nValores: P-ADM-1 = nivel 3.\n")
        self.assertEqual(json.loads(mensajes[-1]["content"]), {
            "herramienta": "registrar_decision", "argumentos": {
                "registro": "provisorio", "numero_documento": None, "box_antes_del_registro": False,
                "circuito": "ninguno", "plazo_circuito_min": None, "cobertura": "sin_cobertura",
                "entidad_a_notificar": None, "aviso_al_voluntariado": False, "fundamentos": ["NOR-ADM-01"]}})
        self.assertNotIn("Ficha registrada.", json.dumps(salida))
        self.assertEqual(dato, antes)

    def test_rechaza_traza_inconsistente_y_contexto_implicito(self):
        cambios = [lambda d: d.update(extra="no admitido"),
                   lambda d: d.update(paciente_id=12),
                   lambda d: d["pasos"][0].update(orden=True),
                   lambda d: d["pasos"][1].update(orden=4),
                   lambda d: d["pasos"][0]["argumentos"].update(paciente_id="P-otro"),
                   lambda d: d["pasos"][1].update(resultado=""),
                   lambda d: d["pasos"][-1].update(resultado=None),
                   lambda d: d["ficha"].update(cobertura="art"),
                   lambda d: d["pasos"].insert(0, copy.deepcopy(d["pasos"][-1])),
                   lambda d: d.update(pasos=[])]
        for cambio in cambios:
            dato = traza()
            cambio(dato)
            with self.subTest(cambio=cambio), self.assertRaises(ValueError):
                self.convertir(dato)
        for campo, valor in [("origen", "referencia"), ("sistema", " "), ("referencia", ""),
                             ("referencia", "practica/casos.jsonl")]:
            opciones = dict(origen="sintetico", sistema="Sistema", referencia="fuente.jsonl")
            opciones[campo] = valor
            with self.subTest(campo=campo, valor=valor), self.assertRaises(ValueError):
                convertir_traza(traza(), **opciones)

    def test_archivo_determinista_y_validado_con_referencias_a_fuente(self):
        from admision_tp.conversion_trazas import convertir_archivo
        from admision_tp.conversaciones import validar_archivo
        with tempfile.TemporaryDirectory() as carpeta:
            fuente = Path(carpeta) / "entrada.jsonl"
            fuente.write_text(json.dumps(traza()) + "\n", encoding="utf-8")
            uno, dos = Path(carpeta) / "uno.jsonl", Path(carpeta) / "dos.jsonl"
            for destino in (uno, dos):
                informe = convertir_archivo(fuente, destino, origen="historial", sistema="Prompt elegido")
                self.assertEqual(informe["registros"], 1)
                self.assertEqual(validar_archivo(destino)["validos"], 1)
            self.assertEqual(uno.read_bytes(), dos.read_bytes())
            dato = json.loads(uno.read_text())
            self.assertEqual(dato["referencias"], [f"{fuente.as_posix()}#E-1"])
            self.assertEqual(dato["origen"], "historial")

    def test_archivo_rechaza_invalidos_completos_sin_salida_y_no_sobrescribe(self):
        from admision_tp.conversion_trazas import convertir_archivo
        with tempfile.TemporaryDirectory() as carpeta:
            fuente, salida = Path(carpeta) / "entrada.jsonl", Path(carpeta) / "salida.jsonl"
            valido = json.dumps(traza()) + "\n"
            for contenido in (b"", b"\xff", b"\n", (valido * 2).encode(),
                              (valido + '{"secreto":INVALIDO}\n').encode()):
                fuente.write_bytes(contenido)
                with self.subTest(contenido=contenido):
                    with self.assertRaises(ValueError) as error:
                        convertir_archivo(fuente, salida, origen="sintetico", sistema="Sistema")
                    self.assertNotIn("secreto", str(error.exception))
                    self.assertFalse(salida.exists())
            fuente.write_text(valido, encoding="utf-8")
            salida.write_bytes(b"no tocar")
            with self.assertRaises(FileExistsError):
                convertir_archivo(fuente, salida, origen="sintetico", sistema="Sistema")
            self.assertEqual(salida.read_bytes(), b"no tocar")
            salida.unlink()
            salida.symlink_to(Path(carpeta) / "inexistente")
            with self.assertRaises(FileExistsError):
                convertir_archivo(fuente, salida, origen="sintetico", sistema="Sistema")
            self.assertTrue(salida.is_symlink())

    def test_ficha_con_tipo_distinto_no_es_igual_a_la_accion(self):
        dato = traza()
        dato["ficha"]["box_antes_del_registro"] = 0
        with self.assertRaises(ValueError):
            self.convertir(dato)

    def test_historial_con_pregunta_y_verificacion_preserva_las_cinco_herramientas(self):
        dato = traza()
        dato.update(fecha="2026-03-01 00:00", operador="OP-01")
        dato["pasos"][2:2] = [
            {"orden": 3, "herramienta": "preguntar", "argumentos":
             {"paciente_id": "P-1", "a_quien": "paciente", "tema": "cobertura"},
             "resultado": "Tiene una prepaga."},
            {"orden": 4, "herramienta": "verificar_afiliacion", "argumentos":
             {"entidad": "Entidad sintética", "numero_afiliado": "42"},
             "resultado": "Afiliación inactiva."}]
        dato["pasos"][-1]["orden"] = 5
        mensajes = self.convertir(dato)["messages"]
        self.assertEqual(json.loads(mensajes[6]["content"]), {"herramienta": "preguntar", "argumentos":
                         {"paciente_id": "P-1", "a_quien": "paciente", "tema": "cobertura"}})
        self.assertEqual(mensajes[7]["content"], "Resultado de preguntar:\nTiene una prepaga.")
        self.assertEqual(json.loads(mensajes[8]["content"]), {"herramienta": "verificar_afiliacion",
                         "argumentos": {"entidad": "Entidad sintética", "numero_afiliado": "42"}})
        self.assertEqual(mensajes[9]["content"], "Resultado de verificar_afiliacion:\nAfiliación inactiva.")

    def test_cli_exige_contexto_y_rechaza_salida_existente(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as carpeta:
            fuente, salida, sistema = (Path(carpeta) / n for n in ("entrada.jsonl", "salida.jsonl", "prompt.txt"))
            fuente.write_text(json.dumps(traza()) + "\n", encoding="utf-8")
            sistema.write_text("Sistema explícito\n", encoding="utf-8")
            comando = [sys.executable, "-m", "scripts.convertir_trazas", "--entrada", str(fuente),
                       "--salida", str(salida)]
            incompleto = subprocess.run(comando, capture_output=True, text=True)
            self.assertEqual(incompleto.returncode, 2)
            self.assertFalse(salida.exists())
            comando += ["--origen", "sintetico", "--sistema", str(sistema)]
            exito = subprocess.run(comando, capture_output=True, text=True)
            self.assertEqual(exito.returncode, 0, exito.stderr)
            self.assertEqual(json.loads(exito.stdout)["registros"], 1)
            self.assertEqual(json.loads(salida.read_text())["messages"][0]["content"], "Sistema explícito\n")
            repetido = subprocess.run(comando, capture_output=True, text=True)
            self.assertEqual(repetido.returncode, 1)
