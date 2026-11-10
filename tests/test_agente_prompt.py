import json
import unittest

from admision.ejemplo import AgenteJSON
from admision_tp.agente_prompt import AgentePrompt
from admision_tp.prompts import PromptSpec, cargar_prompt_baseline


def accion(herramienta, **argumentos):
    return json.dumps({"herramienta": herramienta, "argumentos": argumentos}, ensure_ascii=False)


ACCIONES = [
    "Voy a comenzar.",
    accion("leer_triage", paciente_id="P-1"),
    accion("preguntar", paciente_id="P-1", a_quien="paciente", tema="cobertura"),
    accion("verificar_afiliacion", entidad="Obra Social Bermejo", numero_afiliado="12-345678-9"),
    accion("buscar_normativa", consulta="registro y cobertura"),
    accion("registrar_decision", registro="definitivo", numero_documento="30111222",
           box_antes_del_registro=False, circuito="ninguno", plazo_circuito_min=None,
           cobertura="obra_social", entidad_a_notificar="Obra Social Bermejo",
           aviso_al_voluntariado=False),
]


class EntornoSimulado:
    def __init__(self, paciente_id="P-1"):
        self.paciente_id = paciente_id
        self.llamadas = []

    def paso(self, herramienta, argumentos):
        self.llamadas.append((herramienta, argumentos))
        return f"observación de {herramienta}", herramienta == "registrar_decision"


class ModeloSimulado:
    def __init__(self, acciones=None):
        self.acciones = acciones or ACCIONES
        self.turnos = {}
        self.entradas = []

    def __call__(self, conversaciones):
        self.entradas.extend(json.loads(json.dumps(conversaciones, ensure_ascii=False)))
        respuestas = []
        for mensajes in conversaciones:
            paciente = mensajes[1]["content"]
            turno = self.turnos.get(paciente, 0)
            self.turnos[paciente] = turno + 1
            respuesta = self.acciones[turno]
            if "P-1" not in paciente:
                respuesta = respuesta.replace("P-1", "P-2")
            respuestas.append(respuesta)
        return respuestas


def con_cambios(prompt, **cambios):
    campos = ("version", "nombre", "system_prompt", "mensaje_inicial",
              "formato_observacion", "mensaje_json_invalido")
    return PromptSpec(**{**{campo: getattr(prompt, campo) for campo in campos}, **cambios})


class AgentePromptTest(unittest.TestCase):
    def setUp(self):
        self.baseline = cargar_prompt_baseline()

    def _jugar(self, clase, prompt=None):
        modelo = ModeloSimulado()
        entorno = EntornoSimulado()
        agente = clase(modelo) if prompt is None else clase(modelo, prompt)
        mensajes = agente.jugar_lote([entorno])
        return mensajes, entorno.llamadas, modelo.entradas

    def test_baseline_coincide_con_agente_original(self):
        original = self._jugar(AgenteJSON)
        configurable = self._jugar(AgentePrompt, self.baseline)
        self.assertEqual(configurable, original)
        mensajes, llamadas, entradas = configurable
        self.assertEqual([nombre for nombre, _ in llamadas], [
            "leer_triage", "preguntar", "verificar_afiliacion", "buscar_normativa", "registrar_decision"])
        self.assertEqual(llamadas[1][1], {"paciente_id": "P-1", "a_quien": "paciente", "tema": "cobertura"})
        self.assertEqual(mensajes[0][3]["content"], self.baseline.mensaje_json_invalido)
        self.assertIn("Resultado de leer_triage:\nobservación de leer_triage", mensajes[0][5]["content"])
        self.assertTrue(all(entrada == mensajes[0][:len(entrada)] for entrada in entradas))

    def test_cambiar_system_modifica_solo_ese_mensaje(self):
        mensajes_base, _, _ = self._jugar(AgentePrompt, self.baseline)
        variante = con_cambios(self.baseline, nombre="system-variante", system_prompt="Sistema alternativo")
        mensajes_variante, _, _ = self._jugar(AgentePrompt, variante)
        self.assertEqual(mensajes_variante[0][0], {"role": "system", "content": "Sistema alternativo"})
        self.assertEqual(mensajes_variante[0][1:], mensajes_base[0][1:])

    def test_configura_inicio_observacion_y_correccion(self):
        variante = con_cambios(
            self.baseline,
            nombre="formatos-variante",
            mensaje_inicial="Caso {paciente_id}",
            formato_observacion="{herramienta} => {observacion}",
            mensaje_json_invalido="JSON inválido",
        )
        mensajes, _, _ = self._jugar(AgentePrompt, variante)
        self.assertEqual(mensajes[0][1]["content"], "Caso P-1")
        self.assertEqual(mensajes[0][3]["content"], "JSON inválido")
        self.assertEqual(mensajes[0][5]["content"], "leer_triage => observación de leer_triage")

    def test_jugar_lote_preserva_orden_y_devuelve_conversaciones_completas(self):
        modelo = ModeloSimulado()
        entornos = [EntornoSimulado("P-1"), EntornoSimulado("P-2")]
        agente = AgentePrompt(modelo, self.baseline)
        conversaciones = agente.jugar_lote(entornos)
        self.assertEqual(len(conversaciones), 2)
        self.assertIn("P-1", conversaciones[0][1]["content"])
        self.assertIn("P-2", conversaciones[1][1]["content"])
        self.assertEqual([nombre for nombre, _ in entornos[0].llamadas],
                         [nombre for nombre, _ in entornos[1].llamadas])
        self.assertEqual(len(conversaciones[0]), len(conversaciones[1]))

    def test_expone_version_y_hash_sin_agregarlos_a_la_conversacion(self):
        agente = AgentePrompt(ModeloSimulado(), self.baseline)
        self.assertEqual(agente.identidad_prompt, {
            "nombre": "baseline", "version": 1, "sha256": self.baseline.sha256})
        mensajes = agente.jugar_lote([EntornoSimulado()])[0]
        self.assertNotIn(self.baseline.sha256, json.dumps(mensajes))


if __name__ == "__main__":
    unittest.main()
