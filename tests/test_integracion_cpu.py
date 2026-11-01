"""Integración real del kit; únicamente la generación del modelo está simulada."""

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import entrega
from admision.casos import Caso, leer_jsonl
from admision.correr import jugar, puntuar
from admision_tp.modelo import ModeloLocal

RAIZ = Path(__file__).resolve().parents[1]


def caso(numero):
    return Caso(
        id=f"integracion-{numero}", paciente_id=f"P-CPU-{numero}",
        nombre=f"Paciente Sintético {numero}", edad=30, nivel=5,
        motivo="dolor de tobillo", nota="Dolor tras actividad deportiva.",
        sepsis="no", llegada="por sus propios medios", documento_tipo="DNI",
        documento_numero=f"3000000{numero}", credenciales=(), puede_responder=True,
        no_responde_porque=None, acompanante=None, trabajo=None,
        declarada=("obra_social", "Obra Social Bermejo", f"1234{numero}"),
        art=None, afiliaciones={f"Obra Social Bermejo|1234{numero}": True})


def accion(nombre, **argumentos):
    return json.dumps({"herramienta": nombre, "argumentos": argumentos})


class ModeloGuionado:
    """Solo para tests: acciones predefinidas, sin fichas esperadas como entrada.

    El paciente 2 responde inicialmente sin JSON para ejercitar el reintento.
    Esto no razona sobre normas ni mide calidad de un modelo real.
    """

    def __init__(self):
        self.turnos = {}
        self.lotes = []
        self.entradas = []

    def __call__(self, conversaciones):
        self.lotes.append(len(conversaciones))
        self.entradas.extend(json.loads(json.dumps(conversaciones)))
        respuestas = []
        for mensajes in conversaciones:
            paciente = mensajes[1]["content"].split("paciente ")[1].split(".")[0]
            numero = paciente.rsplit("-", 1)[1]
            turno = self.turnos.get(paciente, 0)
            self.turnos[paciente] = turno + 1
            acciones = [
                accion("leer_triage", paciente_id=paciente),
                accion("buscar_normativa", consulta="NOR-ADM-01 identificación registro documento"),
                accion("buscar_normativa", consulta="NOR-COB-01 consulta cobertura credencial"),
                accion("preguntar", paciente_id=paciente, a_quien="paciente", tema="cobertura"),
                accion("verificar_afiliacion", entidad="Obra Social Bermejo", numero_afiliado=f"1234{numero}"),
                accion("registrar_decision", registro="definitivo", numero_documento=f"3000000{numero}",
                       box_antes_del_registro=False, circuito="ninguno", plazo_circuito_min=None,
                       cobertura="obra_social", entidad_a_notificar="Obra Social Bermejo",
                       aviso_al_voluntariado=False, fundamentos=["NOR-ADM-01", "NOR-COB-02"]),
            ]
            if numero == "2":
                acciones.insert(0, "Voy a leer el triage.")
            respuestas.append(acciones[turno])
        return respuestas


class IntegracionCPUTest(unittest.TestCase):
    def test_lote_real_reintento_trazas_y_nueve_controles(self):
        modelo = ModeloGuionado()
        with tempfile.TemporaryDirectory() as temporal, patch.dict(
                "os.environ", {"BUSCADOR_VARIANTE": "baseline"}), patch.object(
                ModeloLocal, "cargar", return_value=modelo) as cargar:
            salida = Path(temporal)
            trazas = jugar(entrega, [caso(1), caso(2)], RAIZ / "normas/original", salida)
            cargar.assert_called_once()
            self.assertEqual(trazas, leer_jsonl(salida / "trazas.jsonl"))
            tiempos = json.loads((salida / "tiempos.json").read_text())
            self.assertIsNone(tiempos["error"])
            self.assertEqual([t["caso"] for t in trazas], ["integracion-1", "integracion-2"])
            self.assertEqual(modelo.lotes, [2, 2, 2, 2, 2, 2, 1])
            for numero, traza in enumerate(trazas, 1):
                self.assertEqual([e["herramienta"] for e in traza["log"]], [
                    "leer_triage", "buscar_normativa", "buscar_normativa", "preguntar",
                    "verificar_afiliacion", "registrar_decision"])
                self.assertTrue(all(e["valida"] for e in traza["log"]))
                mensajes = traza["mensajes"]
                self.assertEqual(mensajes[0]["role"], "system")
                self.assertEqual(len(mensajes), 15 if numero == 2 else 13)
                self.assertEqual(json.loads(mensajes[-1]["content"])["herramienta"], "registrar_decision")
                for evento in traza["log"][:-1]:
                    self.assertIn({"role": "user", "content":
                        f"Resultado de {evento['herramienta']}:\n{evento['observacion']}"}, mensajes)
                self.assertIn("Valores:", traza["log"][1]["observacion"])
                self.assertIn("NOR-ADM-01", traza["log"][1]["observacion"])
                self.assertIn("activa", traza["log"][4]["observacion"])
                # Todas las entradas reales al modelo son prefijos de la traza final.
                entradas = [m for m in modelo.entradas if f"P-CPU-{numero}" in m[1]["content"]]
                self.assertEqual(len(entradas), 7 if numero == 2 else 6)
                self.assertTrue(all(m == mensajes[:len(m)] for m in entradas))
            self.assertIn("No entendí tu respuesta", trazas[1]["mensajes"][3]["content"])

            # La puntuación se prepara DESPUÉS de jugar y nunca llega al modelo.
            esperadas = {f"integracion-{n}": {
                "caso": f"integracion-{n}", "ficha": {
                    "registro": "definitivo", "numero_documento": f"3000000{n}",
                    "box_antes_del_registro": False, "circuito": "ninguno",
                    "plazo_circuito_min": None, "cobertura": "obra_social",
                    "entidad_a_notificar": "Obra Social Bermejo", "aviso_al_voluntariado": False},
                "verificacion": {"requerida": ["Obra Social Bermejo", f"1234{n}"], "prohibidas": []},
            } for n in (1, 2)}
            resultados, resumen = puntuar(trazas, esperadas, salida, tiempos)
            self.assertEqual(resumen["resueltos"], 2)
            self.assertEqual(len(resumen["por_control"]), 9)
            self.assertTrue(all(v == 1 for v in resumen["por_control"].values()))
            self.assertEqual(resultados, leer_jsonl(salida / "resultados.jsonl"))
            self.assertEqual(resumen, json.loads((salida / "resumen.json").read_text()))
            self.assertTrue(all(r["caracteres_de_contexto"] > 0 for r in resultados))
            self.assertTrue(all(r["busquedas"] == 2 for r in resultados))

    def test_error_del_modelo_preserva_log_y_diagnostico_del_runner(self):
        modelo = ModeloGuionado()

        def fallar_despues_del_triage(conversaciones):
            if modelo.lotes:
                raise RuntimeError("fallo simulado de generación")
            return modelo(conversaciones)

        with tempfile.TemporaryDirectory() as temporal, patch.object(
                ModeloLocal, "cargar", return_value=fallar_despues_del_triage), redirect_stderr(io.StringIO()):
            salida = Path(temporal)
            trazas = jugar(entrega, [caso(1)], RAIZ / "normas/original", salida)
            tiempos = json.loads((salida / "tiempos.json").read_text())
            self.assertEqual(tiempos["error"], "RuntimeError: fallo simulado de generación")
            self.assertIn("RuntimeError", tiempos["error_detalle"])
            self.assertEqual(trazas, leer_jsonl(salida / "trazas.jsonl"))
            self.assertEqual(trazas[0]["log"][0]["herramienta"], "leer_triage")
            self.assertIsNone(trazas[0]["decision"])
            # El kit conserva el log, pero descarta conversaciones si jugar_lote lanza.
            self.assertEqual(trazas[0]["mensajes"], [])


if __name__ == "__main__":
    unittest.main()
