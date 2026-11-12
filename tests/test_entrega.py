"""`entrega.construir_agente` con AgentePrompt: mismo comportamiento que la entrega anterior
(AgenteJSON) salvo el tope de turnos, comprobado con respuestas aleatorias sobre el entorno real."""

import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import entrega
from admision.casos import Caso
from admision.ejemplo import AgenteJSON
from admision.entorno import MAX_LLAMADAS, Entorno
from admision_tp.agente_prompt import AgentePrompt
from admision_tp.modelo import ModeloLocal
from admision_tp.prompts import RAIZ, RUTA_ENTREGA, PromptSpec, cargar_prompt_baseline

FICHA = {"registro": "definitivo", "numero_documento": "30111222", "box_antes_del_registro": False,
         "circuito": "ninguno", "plazo_circuito_min": None, "cobertura": "obra_social",
         "entidad_a_notificar": "Obra Social Bermejo", "aviso_al_voluntariado": False,
         "fundamentos": ["NOR-ADM-01"]}


def caso(numero):
    return Caso(
        id=f"dif-{numero}", paciente_id=f"P-90{numero}", nombre=f"Paciente {numero}", edad=40, nivel=4,
        motivo="dolor torácico" if numero % 2 else "dolor de tobillo", nota="Nota.", sepsis="no",
        llegada="en ambulancia" if numero % 3 == 0 else "por sus propios medios", documento_tipo="DNI",
        documento_numero="30111222", credenciales=(("obra_social", "Obra Social Bermejo", "123"),),
        puede_responder=numero % 4 != 0, no_responde_porque=None if numero % 4 else "está confuso",
        acompanante="su hija" if numero % 2 else None, trabajo=None,
        declarada=("obra_social", "Obra Social Bermejo", "123"), art=None,
        afiliaciones={"Obra Social Bermejo|123": True})


def respuesta_aleatoria(rng, paciente, turno):
    """Mezcla acciones válidas, errores de herramienta, texto sin JSON y fichas incompletas."""
    if turno >= 14:
        return json.dumps({"herramienta": "registrar_decision", "argumentos": FICHA}, ensure_ascii=False)
    opciones = [
        lambda: json.dumps({"herramienta": "leer_triage", "argumentos": {"paciente_id": paciente}}),
        lambda: json.dumps({"herramienta": "preguntar", "argumentos": {
            "paciente_id": paciente, "a_quien": rng.choice(["paciente", "acompanante", "x"]),
            "tema": rng.choice(["cobertura", "art"])}}),
        lambda: json.dumps({"herramienta": "verificar_afiliacion", "argumentos": {
            "entidad": "Obra Social Bermejo", "numero_afiliado": rng.choice(["123", "999"])}}),
        lambda: "```json\n" + json.dumps({"herramienta": "buscar_normativa", "argumentos": {
            "consulta": rng.choice(["registro documento", "cobertura obra social", "dolor torácico ECG"])}}) + "\n```",
        lambda: json.dumps({"herramienta": "inexistente", "argumentos": {}}),
        lambda: rng.choice(["Voy a pensar.", "{no es json", ""]),
        lambda: json.dumps({"herramienta": "registrar_decision", "argumentos": {"registro": "definitivo"}}),
        lambda: json.dumps({"herramienta": "registrar_decision", "argumentos": FICHA}, ensure_ascii=False),
    ]
    return rng.choice(opciones)()


class ModeloAleatorio:
    """Respuesta determinista por (semilla, paciente, turno): igual para los dos agentes."""

    def __init__(self, semilla, maximo=None):
        self.semilla, self.maximo = semilla, maximo
        self.turnos, self.lotes = {}, []

    def __call__(self, conversaciones):
        self.lotes.append(len(conversaciones))
        respuestas = []
        for mensajes in conversaciones:
            paciente = mensajes[1]["content"].split("paciente ")[1].split(".")[0]
            turno = self.turnos.get(paciente, 0)
            self.turnos[paciente] = turno + 1
            rng = random.Random(f"{self.semilla}-{paciente}-{turno}")
            if self.maximo is None:
                respuestas.append(respuesta_aleatoria(rng, paciente, turno))
            else:  # nunca registra: sirve para ver dónde corta cada agente
                respuestas.append(json.dumps({"herramienta": "leer_triage", "argumentos": {"paciente_id": paciente}}))
        return respuestas


def jugar(agente, casos):
    buscador = entrega.construir_buscador(RAIZ / "normas/original")
    entornos = [Entorno(c, buscador) for c in casos]
    conversaciones = agente.jugar_lote(entornos)
    return conversaciones, [e.log for e in entornos], [e.decision for e in entornos]


class FabricaTest(unittest.TestCase):
    def test_construye_agente_prompt_con_el_prompt_elegido_y_carga_el_modelo_una_vez(self):
        modelo = ModeloAleatorio(0)
        with patch.dict("os.environ", {}, clear=True), patch.object(ModeloLocal, "cargar", return_value=modelo) as cargar:
            agente = entrega.construir_agente()
        cargar.assert_called_once()
        self.assertIsInstance(agente, AgentePrompt)
        self.assertIs(agente.modelo_lote, modelo)
        self.assertEqual(agente.prompt, PromptSpec.cargar(RUTA_ENTREGA))
        self.assertEqual(agente.prompt.nombre, "procedimiento-v1")
        self.assertEqual(agente.max_turnos, MAX_LLAMADAS * 3)

    def test_prompt_configurable_por_entorno_relativo_o_absoluto(self):
        from dataclasses import replace
        otro = replace(cargar_prompt_baseline(), nombre="otro", mensaje_inicial="Paciente {paciente_id}.")
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta).resolve() / "otro.json"
            ruta.write_text(json.dumps(otro.como_dict(), ensure_ascii=False), encoding="utf-8")
            for valor in (str(ruta), os.path.relpath(ruta, RAIZ)):
                with patch.dict("os.environ", {"PROMPT_SPEC": valor}), patch.object(ModeloLocal, "cargar"):
                    self.assertEqual(entrega.construir_agente().prompt, otro)

    def test_prompt_alterado_falla_antes_de_cargar_el_modelo(self):
        dato = cargar_prompt_baseline().como_dict()
        dato["system_prompt"] += " cambio sin actualizar el hash"
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "alterado.json"
            ruta.write_text(json.dumps(dato, ensure_ascii=False), encoding="utf-8")
            with patch.dict("os.environ", {"PROMPT_SPEC": str(ruta)}), \
                    patch.object(ModeloLocal, "cargar") as cargar, self.assertRaisesRegex(ValueError, "sha256"):
                entrega.construir_agente()
            cargar.assert_not_called()


class DiferencialTest(unittest.TestCase):
    """Entrega nueva vs. la anterior (`AgenteJSON(ModeloLocal)`), con el entorno y buscador reales.

    Con el prompt baseline (el de `AgenteJSON`): compara la plomería, no el prompt de la entrega."""

    def _ambas(self, crear_modelo, casos):
        anterior = jugar(AgenteJSON(crear_modelo()), casos)
        with patch.dict("os.environ", {"PROMPT_SPEC": "configuraciones/prompts/baseline.json"}, clear=True), \
                patch.object(ModeloLocal, "cargar", return_value=crear_modelo()):
            nueva = jugar(entrega.construir_agente(), casos)
        return anterior, nueva

    def test_misma_conversacion_log_y_ficha_con_respuestas_aleatorias(self):
        casos = [caso(n) for n in range(1, 5)]
        decisiones = cortes = 0
        for semilla in range(40):
            anterior, nueva = self._ambas(lambda: ModeloAleatorio(semilla), casos)
            self.assertEqual(nueva, anterior, f"semilla {semilla}")
            decisiones += sum(d is not None for d in nueva[2])
            cortes += sum(d is None for d in nueva[2])
        # El guion ejercita ambos finales: con ficha y sin ficha (tres respuestas sin JSON).
        self.assertGreater(decisiones, 0)
        self.assertGreater(cortes, 0)

    def test_unica_diferencia_es_el_tope_de_16_turnos(self):
        anterior, nueva = self._ambas(lambda: ModeloAleatorio(0, maximo=True), [caso(1)])
        self.assertEqual(len(anterior[1][0]), 16)
        self.assertEqual(len(nueva[1][0]), MAX_LLAMADAS)
        self.assertEqual(nueva[0][0][:len(anterior[0][0])], anterior[0][0])
        self.assertEqual((anterior[2], nueva[2]), ([None], [None]))


if __name__ == "__main__":
    unittest.main()
