import json
import unittest

from admision.casos import Caso
from admision.ejemplo import AgenteJSON
from admision.entorno import MAX_LLAMADAS, Entorno
from admision_tp.agente_prompt import MENSAJE_CONTEXTO_EXCEDIDO, AgentePrompt
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


class ModeloPorPaciente:
    """Guion por paciente; registra qué pacientes entran en cada llamada al modelo."""

    def __init__(self, guiones):
        self.guiones = guiones
        self.turnos = {}
        self.llamadas = []

    def __call__(self, conversaciones):
        pacientes = [next(p for p in self.guiones if p in m[1]["content"]) for m in conversaciones]
        self.llamadas.append(pacientes)
        respuestas = []
        for paciente in pacientes:
            turno = self.turnos.get(paciente, 0)
            self.turnos[paciente] = turno + 1
            guion = self.guiones[paciente]
            respuestas.append(guion(turno) if callable(guion) else guion[turno])
        return respuestas


def caso_entorno(paciente_id="P-1"):
    return Caso.desde_dict({
        "id": "c-1", "paciente_id": paciente_id, "nombre": "Paciente Simulado",
        "edad": 30, "nivel": 5, "motivo": "dolor", "nota": "Nota simulada.",
        "sepsis": "no", "llegada": "por sus propios medios", "documento_tipo": "DNI",
        "documento_numero": "30111222", "credenciales": [], "puede_responder": True,
        "no_responde_porque": None, "acompanante": None, "trabajo": None,
        "declarada": None, "art": None, "afiliaciones": {},
    })


class EquivalenciaLimitesTest(unittest.TestCase):
    """Bordes del coordinador: AgentePrompt(baseline) debe igualar a AgenteJSON."""

    def setUp(self):
        self.baseline = cargar_prompt_baseline()

    def _ambos(self, guiones, entornos_factory, **limites):
        resultados = []
        for crear in (lambda m: AgenteJSON(m, **limites),
                      lambda m: AgentePrompt(m, self.baseline, **limites)):
            modelo = ModeloPorPaciente(guiones)
            entornos = entornos_factory()
            mensajes = crear(modelo).jugar_lote(entornos)
            resultados.append((mensajes, modelo.llamadas, entornos))
        (original, llamadas_o, _), (configurable, llamadas_c, entornos) = resultados
        self.assertEqual(configurable, original)
        self.assertEqual(llamadas_c, llamadas_o)
        return configurable, llamadas_c, entornos

    def test_tres_respuestas_sin_json_seguidas_cortan_el_caso(self):
        mensajes, llamadas, entornos = self._ambos(
            {"P-1": lambda turno: "no es JSON"}, lambda: [EntornoSimulado()])
        asistente = [m for m in mensajes[0] if m["role"] == "assistant"]
        self.assertEqual(len(asistente), 3)
        self.assertEqual(len(llamadas), 3)
        self.assertEqual(entornos[0].llamadas, [])
        # La tercera respuesta inválida corta sin agregar otra corrección.
        self.assertEqual(mensajes[0][-1], {"role": "assistant", "content": "no es JSON"})

    def test_json_valido_reinicia_el_contador_sin_json(self):
        guion = ["x", "y", accion("buscar_normativa", consulta="a"), "z", "w",
                 accion("registrar_decision")]
        mensajes, _, entornos = self._ambos({"P-1": guion}, lambda: [EntornoSimulado()])
        self.assertEqual([n for n, _ in entornos[0].llamadas],
                         ["buscar_normativa", "registrar_decision"])
        self.assertEqual(sum(m["role"] == "assistant" for m in mensajes[0]), 6)

    def test_max_turnos_corta_el_caso(self):
        busqueda = accion("buscar_normativa", consulta="cobertura")
        mensajes, _, entornos = self._ambos(
            {"P-1": lambda turno: busqueda}, lambda: [EntornoSimulado()], max_turnos=5)
        self.assertEqual(sum(m["role"] == "assistant" for m in mensajes[0]), 5)
        self.assertEqual(len(entornos[0].llamadas), 5)
        self.assertEqual(mensajes[0][-1]["role"], "user")

    def test_limite_de_llamadas_del_entorno_termina_el_caso(self):
        lectura = accion("leer_triage", paciente_id="P-1")

        def entornos():
            return [Entorno(caso_entorno(), buscador=None)]
        mensajes, _, (entorno,) = self._ambos(
            {"P-1": lambda turno: lectura}, entornos, max_turnos=MAX_LLAMADAS + 20)
        self.assertEqual(MAX_LLAMADAS, 40)
        self.assertEqual(len(entorno.log), MAX_LLAMADAS)
        self.assertTrue(entorno.terminado)
        self.assertIsNone(entorno.decision)
        self.assertEqual(sum(m["role"] == "assistant" for m in mensajes[0]), MAX_LLAMADAS)
        self.assertEqual(mensajes[0][-1]["role"], "assistant")

    def test_lote_con_casos_de_distinto_largo(self):
        fin = accion("registrar_decision")
        guiones = {
            "P-1": [accion("leer_triage", paciente_id="P-1"), fin],
            "P-2": ["sin json", accion("leer_triage", paciente_id="P-2"),
                    accion("buscar_normativa", consulta="x"), fin],
            "P-3": [fin],
        }
        mensajes, llamadas, entornos = self._ambos(
            guiones, lambda: [EntornoSimulado("P-1"), EntornoSimulado("P-2"), EntornoSimulado("P-3")])
        self.assertEqual(llamadas, [["P-1", "P-2", "P-3"], ["P-1", "P-2"], ["P-2"], ["P-2"]])
        self.assertEqual([sum(m["role"] == "assistant" for m in c) for c in mensajes], [2, 4, 1])
        for paciente, conversacion in zip(("P-1", "P-2", "P-3"), mensajes):
            self.assertIn(paciente, conversacion[1]["content"])
        self.assertEqual([e.llamadas[-1][0] for e in entornos], ["registrar_decision"] * 3)


class EntornoSinFin(EntornoSimulado):
    """Nunca termina el caso por su cuenta: el tope tiene que ponerlo el agente."""

    def paso(self, herramienta, argumentos):
        self.llamadas.append((herramienta, argumentos))
        return f"observación de {herramienta}", False


class TopeDeLlamadasTest(unittest.TestCase):
    """#7: el corte sale del tope de 40 llamadas del kit, no de 16 turnos fijos."""

    def setUp(self):
        self.baseline = cargar_prompt_baseline()

    def test_turnos_derivados_del_tope_dejan_lugar_a_reintentos(self):
        agente = AgentePrompt(ModeloPorPaciente({}), self.baseline)
        self.assertEqual(agente.max_llamadas, MAX_LLAMADAS)
        self.assertEqual(agente.max_turnos, MAX_LLAMADAS * agente.max_sin_json)

    def test_entorno_real_termina_en_40_llamadas_aunque_haya_reintentos(self):
        lectura = accion("leer_triage", paciente_id="P-1")
        # Dos respuestas sin JSON antes de cada llamada: el peor caso que no corta por formato.
        modelo = ModeloPorPaciente({"P-1": lambda turno: lectura if turno % 3 == 2 else "sin json"})
        entorno = Entorno(caso_entorno(), buscador=None)
        mensajes = AgentePrompt(modelo, self.baseline).jugar_lote([entorno])[0]
        self.assertEqual(len(entorno.log), MAX_LLAMADAS)
        self.assertTrue(entorno.terminado)
        self.assertIsNone(entorno.decision)
        self.assertEqual(sum(m["role"] == "assistant" for m in mensajes), 3 * MAX_LLAMADAS)
        self.assertEqual(mensajes[-1], {"role": "assistant", "content": lectura})
        # El caso terminado no vuelve a pedir respuestas al modelo.
        self.assertEqual(len(modelo.llamadas), 3 * MAX_LLAMADAS)

    def test_el_agente_no_pasa_de_40_llamadas_si_el_entorno_no_corta(self):
        busqueda = accion("buscar_normativa", consulta="x")
        entorno = EntornoSinFin()
        mensajes = AgentePrompt(ModeloPorPaciente({"P-1": lambda turno: busqueda}),
                                self.baseline).jugar_lote([entorno])[0]
        self.assertEqual(len(entorno.llamadas), MAX_LLAMADAS)
        self.assertEqual(sum(m["role"] == "assistant" for m in mensajes), MAX_LLAMADAS)

    def test_tope_explicito_de_turnos_se_respeta(self):
        busqueda = accion("buscar_normativa", consulta="x")
        entorno = EntornoSinFin()
        AgentePrompt(ModeloPorPaciente({"P-1": lambda turno: busqueda}), self.baseline,
                     max_turnos=16).jugar_lote([entorno])
        self.assertEqual(len(entorno.llamadas), 16)


class ModeloConContexto(ModeloPorPaciente):
    """Doble con la capacidad opcional: mide en caracteres y admite `limite`."""

    def __init__(self, guiones, limite):
        super().__init__(guiones)
        self.limite = limite
        self.mediciones = 0

    def medir_contexto(self, conversacion):
        self.mediciones += 1
        return sum(len(m["content"]) for m in conversacion), self.limite


class EntornoConObservacion(EntornoSimulado):
    def __init__(self, paciente_id, observacion):
        super().__init__(paciente_id)
        self.observacion = observacion

    def paso(self, herramienta, argumentos):
        self.llamadas.append((herramienta, argumentos))
        return self.observacion, herramienta == "registrar_decision"


class GuardaDeContextoTest(unittest.TestCase):
    def setUp(self):
        self.baseline = cargar_prompt_baseline()
        fin = accion("registrar_decision")
        self.guiones = {p: [accion("leer_triage", paciente_id=p), fin] for p in ("P-1", "P-2", "P-3", "P-4")}
        self.entornos = [EntornoConObservacion(p, "x" * (5000 if p == "P-3" else 10))
                         for p in ("P-1", "P-2", "P-3", "P-4")]

    def test_caso_que_no_entra_termina_sin_decision_y_los_demas_siguen(self):
        modelo = ModeloConContexto(self.guiones, limite=len(self.baseline.system_prompt) + 1000)
        conversaciones = AgentePrompt(modelo, self.baseline).jugar_lote(self.entornos)
        self.assertEqual(modelo.llamadas, [["P-1", "P-2", "P-3", "P-4"], ["P-1", "P-2", "P-4"]])
        for entorno in (self.entornos[0], self.entornos[1], self.entornos[3]):
            self.assertEqual([n for n, _ in entorno.llamadas], ["leer_triage", "registrar_decision"])
        self.assertEqual([n for n, _ in self.entornos[2].llamadas], ["leer_triage"])
        cortada = conversaciones[2]
        # Nada se recorta: la observación grande queda entera y el motivo, al final.
        self.assertEqual(cortada[3]["content"], self.baseline.observacion_para("leer_triage", "x" * 5000))
        tokens = sum(len(m["content"]) for m in cortada[:4])
        self.assertEqual(cortada[4], {"role": "user", "content": MENSAJE_CONTEXTO_EXCEDIDO.format(
            tokens=tokens, limite=modelo.limite)})
        self.assertEqual(len(cortada), 5)
        for conversacion in (conversaciones[0], conversaciones[1], conversaciones[3]):
            self.assertEqual(json.loads(conversacion[-1]["content"])["herramienta"], "registrar_decision")

    def test_contexto_que_entra_no_cambia_la_conversacion(self):
        sin_capacidad = AgentePrompt(ModeloPorPaciente(self.guiones), self.baseline).jugar_lote(
            [EntornoConObservacion(p, "x" * 10) for p in self.guiones])
        modelo = ModeloConContexto(self.guiones, limite=10 ** 6)
        con_capacidad = AgentePrompt(modelo, self.baseline).jugar_lote(
            [EntornoConObservacion(p, "x" * 10) for p in self.guiones])
        self.assertEqual(con_capacidad, sin_capacidad)
        self.assertEqual(modelo.mediciones, 8)

    def test_sistema_que_no_entra_corta_antes_de_la_primera_llamada(self):
        modelo = ModeloConContexto(self.guiones, limite=10)
        conversaciones = AgentePrompt(modelo, self.baseline).jugar_lote(self.entornos)
        self.assertEqual(modelo.llamadas, [])
        self.assertTrue(all(len(c) == 3 and c[2]["content"].startswith("[agente]") for c in conversaciones))


if __name__ == "__main__":
    unittest.main()
