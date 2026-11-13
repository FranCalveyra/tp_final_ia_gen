"""Subconjunto balanceado para reforzar el aviso al voluntariado sobre un adapter ya entrenado."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.subconjunto_aviso import armar_subconjunto, main


def conversacion(numero, edad, acompanante, aviso):
    triage = (f"Resultado de leer_triage:\nNota de triage del paciente P-{numero}\n"
              f"Paciente: Persona {numero}, {edad} años.\nNivel de triage: 3.\nAcompañante: {acompanante}.\n")
    decision = {"herramienta": "registrar_decision", "argumentos": {"aviso_al_voluntariado": aviso}}
    return {"version": 1, "id": f"S-{numero:03d}", "origen": "sintetico", "referencias": ["x"], "messages": [
        {"role": "system", "content": "Sistema."},
        {"role": "user", "content": f"Ingresó el paciente P-{numero}."},
        {"role": "assistant", "content": json.dumps({"herramienta": "leer_triage", "argumentos": {}})},
        {"role": "user", "content": triage},
        {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
    ]}


def datos():
    registros = [conversacion(1, 80, "ninguno", True), conversacion(2, 90, "ninguno", True),
                 conversacion(3, 85, "su hija", False), conversacion(4, 82, "su esposo", False),
                 conversacion(5, 70, "ninguno", False)]  # 70 < menor positivo: no es negativo difícil
    registros += [conversacion(10 + i, 30 + i, "ninguno" if i % 2 else "su madre", False) for i in range(20)]
    return registros


class SubconjuntoAvisoTest(unittest.TestCase):
    def test_todos_los_positivos_y_negativos_dificiles_y_muestra_del_resto(self):
        elegidos, resumen = armar_subconjunto(datos(), otros=4, semilla=7)
        ids = [r["id"] for r in elegidos]
        self.assertTrue({"S-001", "S-002", "S-003", "S-004"} <= set(ids))
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(resumen["edad_minima_positivos"], 80)
        self.assertEqual(resumen["grupos"], {"positivos": 2, "negativos_mismo_rango_de_edad": 2, "otros_negativos": 4})
        # Determinista con la semilla, y conserva el orden original.
        self.assertEqual(armar_subconjunto(datos(), otros=4, semilla=7)[0], elegidos)
        orden = {r["id"]: i for i, r in enumerate(datos())}
        self.assertEqual(ids, sorted(ids, key=orden.get))

    def test_lee_el_aviso_aunque_la_decision_lleve_prefijo_de_hechos(self):
        registro = conversacion(1, 80, "ninguno", True)
        registro["messages"][-1]["content"] = "Edad: 80 años. Acompañante: ninguno.\n" + registro["messages"][-1]["content"]
        elegidos, resumen = armar_subconjunto([registro], otros=0, semilla=1)
        self.assertEqual(resumen["grupos"]["positivos"], 1)

    def test_falla_si_no_hay_positivos_o_falta_el_triage(self):
        with self.assertRaisesRegex(ValueError, "positivos"):
            armar_subconjunto([conversacion(1, 40, "ninguno", False)], otros=0, semilla=1)
        roto = conversacion(1, 80, "ninguno", True)
        roto["messages"][3]["content"] = "sin triage"
        with self.assertRaisesRegex(ValueError, "triage"):
            armar_subconjunto([roto], otros=0, semilla=1)

    def test_cli_copia_lineas_tal_cual_y_registra_manifiesto(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            origen = raiz / "conversaciones.jsonl"
            origen.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in datos()), encoding="utf-8")
            salida = raiz / "v4"
            with patch("scripts.subconjunto_aviso._codigo", return_value={"commit": "a" * 40, "dirty": False}):
                self.assertEqual(main(["--datos", str(origen), "--salida", str(salida),
                                       "--otros", "4", "--semilla", "7"]), 0)
            lineas = (salida / "conversaciones.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lineas), 8)
            self.assertTrue(set(lineas) <= set(origen.read_text(encoding="utf-8").splitlines()))
            manifiesto = json.loads((salida / "manifiesto.json").read_text())
            self.assertEqual(manifiesto["semilla"], 7)
            self.assertEqual(manifiesto["registros"], 8)
            self.assertEqual(manifiesto["git"]["commit"], "a" * 40)
            self.assertIn("conversaciones.jsonl", manifiesto["archivos_sha256"])
            with self.assertRaisesRegex(ValueError, "existe"):
                main(["--datos", str(origen), "--salida", str(salida), "--otros", "4", "--semilla", "7"])


if __name__ == "__main__":
    unittest.main()
