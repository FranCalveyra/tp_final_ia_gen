"""Prefijo de hechos (edad y acompañante) antes de la decisión final, copiado de la nota de triage (#15)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admision_tp.conversaciones import validar_registro
from scripts.anteponer_datos_aviso import anteponer, main
from test_conversaciones import registro as registro_base


def conversacion(numero, edad, acompanante, aviso):
    dato = registro_base()
    dato["id"] = f"S-{numero:03d}"
    dato["messages"][3]["content"] = (
        f"Resultado de leer_triage:\nNota de triage del paciente P-{numero}\n"
        f"Paciente: Persona {numero}, {edad} años.\nNivel de triage: 3.\nAcompañante: {acompanante}.\n")
    accion = json.loads(dato["messages"][-1]["content"])
    accion["argumentos"]["aviso_al_voluntariado"] = aviso
    dato["messages"][-1]["content"] = json.dumps(accion, ensure_ascii=False)
    return dato


class AnteponerDatosAvisoTest(unittest.TestCase):
    def test_antepone_los_hechos_solo_al_ultimo_mensaje(self):
        original = conversacion(1, 88, "ninguno", True)
        nuevo = anteponer(original)
        self.assertEqual(nuevo["messages"][-1]["content"],
                         "Edad: 88 años. Acompañante: ninguno.\n" + original["messages"][-1]["content"])
        self.assertEqual(nuevo["messages"][:-1], original["messages"][:-1])
        validar_registro(nuevo)

    def test_acompanante_con_puntos_y_parentesis_conserva_el_valor(self):
        nuevo = anteponer(conversacion(2, 40, "su hija (Dra. Ana)", False))
        self.assertTrue(nuevo["messages"][-1]["content"].startswith("Edad: 40 años. Acompañante: su hija (Dra. Ana).\n"))

    def test_falla_si_falta_un_dato_o_la_decision_o_hay_llaves(self):
        sin_edad = conversacion(1, 80, "ninguno", True)
        sin_edad["messages"][3]["content"] = sin_edad["messages"][3]["content"].replace("80 años", "")
        sin_acompanante = conversacion(1, 80, "ninguno", True)
        sin_acompanante["messages"][3]["content"] = "Resultado de leer_triage:\nNota de triage.\nPaciente: X, 80 años.\n"
        sin_decision = conversacion(1, 80, "ninguno", True)
        sin_decision["messages"][-1]["content"] = json.dumps({"herramienta": "leer_triage", "argumentos": {}})
        con_llave = conversacion(1, 80, "{ninguno}", True)
        for caso in (sin_edad, sin_acompanante, sin_decision, con_llave):
            with self.subTest(), self.assertRaises(ValueError):
                anteponer(caso)

    def test_cli_escribe_carpeta_nueva_con_manifiesto_y_no_sobrescribe(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            origen = raiz / "conversaciones.jsonl"
            registros = [conversacion(1, 88, "ninguno", True), conversacion(2, 30, "su madre", False)]
            origen.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
            salida = raiz / "v6"
            with patch("scripts.anteponer_datos_aviso._codigo", return_value={"commit": "a" * 40, "dirty": False}):
                self.assertEqual(main(["--datos", str(origen), "--salida", str(salida)]), 0)
                with self.assertRaisesRegex(ValueError, "existe"):
                    main(["--datos", str(origen), "--salida", str(salida)])
            nuevos = [json.loads(x) for x in (salida / "conversaciones.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["messages"][:-1] for r in nuevos], [r["messages"][:-1] for r in registros])
            self.assertTrue(nuevos[1]["messages"][-1]["content"].startswith("Edad: 30 años. Acompañante: su madre.\n"))
            manifiesto = json.loads((salida / "manifiesto.json").read_text())
            self.assertEqual((manifiesto["registros"], manifiesto["origen"]["registros"]), (2, 2))
            self.assertEqual(manifiesto["git"]["dirty"], False)
            self.assertEqual(set(manifiesto), {"version", "proposito", "comando", "git", "origen", "registros",
                                               "archivos_sha256"})


if __name__ == "__main__":
    unittest.main()
