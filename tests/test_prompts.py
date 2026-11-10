import copy
import json
from pathlib import Path
import tempfile
import unittest

from admision.ejemplo import mensaje_de_sistema, primer_mensaje
from admision_tp.prompts import PromptSpec, cargar_prompt_baseline


class PromptSpecTest(unittest.TestCase):
    def setUp(self):
        self.baseline = cargar_prompt_baseline()

    def test_baseline_preserva_textos_del_agente_original(self):
        self.assertEqual(self.baseline.system_prompt, mensaje_de_sistema())
        self.assertEqual(self.baseline.mensaje_inicial_para("P-1234"), primer_mensaje("P-1234"))
        self.assertEqual(self.baseline.observacion_para("leer_triage", "nota"),
                         "Resultado de leer_triage:\nnota")
        self.assertEqual(self.baseline.mensaje_json_invalido,
            'No entendí tu respuesta. Respondé solo con un objeto JSON: '
            '{"herramienta": "<nombre>", "argumentos": {...}}')

    def test_hash_estable_y_sensible_al_contenido(self):
        copia = PromptSpec(**{k: getattr(self.baseline, k) for k in (
            "version", "nombre", "system_prompt", "mensaje_inicial",
            "formato_observacion", "mensaje_json_invalido")})
        self.assertEqual(copia.sha256, self.baseline.sha256)
        variante = PromptSpec(**{**copia.__dict__, "system_prompt": copia.system_prompt + "\nOtra instrucción."})
        self.assertNotEqual(variante.sha256, self.baseline.sha256)

    def test_nombre_no_cambia_identidad_del_contenido(self):
        variante = PromptSpec(**{**self.baseline.__dict__, "nombre": "otra-etiqueta"})
        self.assertEqual(variante.sha256, self.baseline.sha256)

    def test_carga_rechaza_hash_alterado_y_campos_desconocidos(self):
        for cambio in ("contenido", "campo"):
            dato = self.baseline.como_dict()
            if cambio == "contenido":
                dato["system_prompt"] += " alterado"
            else:
                dato["extra"] = True
            with self.subTest(cambio=cambio), tempfile.TemporaryDirectory() as carpeta:
                ruta = Path(carpeta) / "prompt.json"
                ruta.write_text(json.dumps(dato, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    PromptSpec.cargar(ruta)

    def test_valida_placeholders_y_no_muta_entrada(self):
        dato = self.baseline.como_dict()
        antes = copy.deepcopy(dato)
        self.assertEqual(PromptSpec.desde_dict(dato), self.baseline)
        self.assertEqual(dato, antes)
        for campo, valor in (("mensaje_inicial", "Sin identificador"),
                             ("formato_observacion", "{herramienta}")):
            kwargs = dict(self.baseline.__dict__)
            kwargs[campo] = valor
            with self.subTest(campo=campo), self.assertRaises(ValueError):
                PromptSpec(**kwargs)


if __name__ == "__main__":
    unittest.main()
