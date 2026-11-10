import json
from dataclasses import replace
import unittest

from admision.correr import caracteres_de_contexto
from admision.entorno import Entorno
from admision_tp.agente_prompt import AgentePrompt
from admision_tp.contexto import desglose, rejugar_traza
from admision_tp.prompts import cargar_prompt_baseline
from scripts.medir_contexto import medir
from test_agente_prompt import ModeloPorPaciente, accion, caso_entorno


def trazas_jugadas(prompt):
    """Trazas como las de `admision.correr`, jugadas con AgentePrompt y el entorno real del kit."""
    ficha = {"registro": "definitivo", "numero_documento": "30111222", "box_antes_del_registro": False,
             "circuito": "ninguno", "plazo_circuito_min": None, "cobertura": "sin_cobertura",
             "entidad_a_notificar": None, "aviso_al_voluntariado": False}
    guiones = {
        # Corrección por JSON inválido, error de herramienta y ficha registrada.
        "P-1": ["hola", accion("leer_triage", paciente_id="P-1"), accion("preguntar", paciente_id="P-1"),
                accion("registrar_decision", **ficha)],
        # Tres respuestas sin JSON seguidas: termina sin ficha y con la última observación entregada.
        "P-2": [accion("leer_triage", paciente_id="P-2"), "a", "b", "c"],
    }
    entornos = [Entorno(caso_entorno(p), buscador=None) for p in guiones]
    conversaciones = AgentePrompt(ModeloPorPaciente(guiones), prompt).jugar_lote(entornos)
    return [{"mensajes": m, "log": e.log, "decision": e.decision} for m, e in zip(conversaciones, entornos)]


class ContextoTest(unittest.TestCase):
    def setUp(self):
        self.baseline = cargar_prompt_baseline()
        self.trazas = trazas_jugadas(self.baseline)
        self.compacto = replace(self.baseline, nombre="compacto", formato_observacion="{herramienta}:\n{observacion}")

    def test_rejugar_con_el_prompt_de_origen_reproduce_la_traza(self):
        self.assertIsNotNone(self.trazas[0]["decision"])
        self.assertIsNone(self.trazas[1]["decision"])
        for traza in self.trazas:
            self.assertEqual(rejugar_traza(traza, self.baseline, self.baseline), traza["mensajes"])

    def test_rejugar_con_otro_prompt_es_jugar_con_ese_prompt(self):
        for traza, directa in zip(self.trazas, trazas_jugadas(self.compacto)):
            self.assertEqual(rejugar_traza(traza, self.compacto, self.baseline), directa["mensajes"])

    def test_desglose_suma_el_total_del_kit(self):
        mensajes = self.trazas[0]["mensajes"]
        partes = desglose(mensajes, self.baseline)
        self.assertEqual(sum(partes.values()), caracteres_de_contexto(mensajes))
        self.assertGreater(partes["correcciones"], 0)
        self.assertEqual(partes["sistema"], len(self.baseline.system_prompt) * 4)

    def test_medir_informa_ahorro_y_rechaza_trazas_que_no_reproduce(self):
        informe = medir(self.trazas, [self.baseline, self.compacto], self.baseline)
        self.assertEqual(informe["prompts"]["baseline"]["ahorro_total"], 0)
        self.assertGreater(informe["prompts"]["compacto"]["ahorro_total"], 0)
        alterada = json.loads(json.dumps(self.trazas))
        alterada[0]["log"][0]["observacion"] += " cambiada"
        with self.assertRaisesRegex(ValueError, "no reproduce"):
            medir(alterada, [self.baseline], self.baseline)


if __name__ == "__main__":
    unittest.main()
