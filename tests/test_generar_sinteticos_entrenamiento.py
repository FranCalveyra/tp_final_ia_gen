import json
import unittest
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import entrega
from admision.generador import generar
from scripts.clasificar_historial import SI, aviso_segun_costumbre, cargar_parametros
from scripts.generar_sinteticos_entrenamiento import (
    costumbres_por_tipo,
    deducir_ficha,
    generar_sinteticos,
)


RAIZ = Path(__file__).resolve().parents[1]
SECCION_DEL_CIRCUITO = {
    "ecg": "NOR-TDE-01",
    "codigo_acv": "NOR-TDE-02",
    "sepsis_probable": "NOR-TDE-03",
    "sepsis_posible": "NOR-TDE-03",
}


@lru_cache(maxsize=1)
def generar_una_vez():
    return generar_sinteticos(RAIZ / "normas/original", RAIZ / "historial_costumbres.json")


def costumbres():
    return costumbres_por_tipo(json.loads((RAIZ / "historial_costumbres.json").read_text(encoding="utf-8")))


def observaciones_de_busqueda(conversacion: dict) -> list[str]:
    mensajes = conversacion["messages"]
    return [
        mensajes[i + 1]["content"].split("\n", 1)[1]
        for i, mensaje in enumerate(mensajes[:-1])
        if mensaje["role"] == "assistant"
        and json.loads(mensaje["content"])["herramienta"] == "buscar_normativa"
    ]


class SinteticosEntrenamientoTest(unittest.TestCase):
    def test_balance_y_fichas_normativamente_validas(self):
        conversaciones, auditoria, resumen = generar_una_vez()
        self.assertEqual(len(conversaciones), 500)
        self.assertEqual(len(auditoria), 500)
        self.assertEqual(resumen["candidatos_generados"], 5000)
        self.assertEqual(resumen["por_circuito"], {
            "codigo_acv": 100,
            "ecg": 100,
            "ninguno": 100,
            "sepsis_posible": 100,
            "sepsis_probable": 100,
        })
        for bucket in ("art", "dos_credenciales", "acompanante", "sin_nadie"):
            self.assertGreaterEqual(resumen["por_dificultad"][bucket], 60)
        self.assertEqual(resumen["fichas_normativamente_validas"], 500)
        self.assertEqual(conversaciones[0]["origen"], "sintetico")
        self.assertTrue(all("practica" not in " ".join(c["referencias"]) for c in conversaciones))
        self.assertTrue(all("test" not in " ".join(c["referencias"]) for c in conversaciones))

    def test_aviso_sigue_la_costumbre_completa_sin_muestreo(self):
        _, auditoria, resumen = generar_una_vez()
        self.assertEqual(resumen["costumbre_aviso"], {"umbral_edad": 75, "condicion": "sin_acompanante"})
        aviso = costumbres()["aviso_al_voluntariado"]
        for item in auditoria:
            fila = {"edad": str(item["edad"]), "acompanante": "x" if item["acompanante"] else ""}
            self.assertEqual(item["ficha"]["aviso_al_voluntariado"], aviso_segun_costumbre(fila, aviso))
        mayores_solos = [i for i in auditoria if i["edad"] >= 75 and not i["acompanante"]]
        mayores_acompanados = [i for i in auditoria if i["edad"] >= 75 and i["acompanante"]]
        self.assertTrue(mayores_solos and mayores_acompanados)
        self.assertTrue(all(i["ficha"]["aviso_al_voluntariado"] for i in mayores_solos))
        self.assertFalse(any(i["ficha"]["aviso_al_voluntariado"] for i in mayores_acompanados))

    def test_terminacion_reservada_inactiva_notifica_a_la_entidad(self):
        parametros = cargar_parametros(RAIZ / "normas/original")
        digito = str(parametros["P-COB-2"])
        caso = next(
            c for c in generar(semilla=12012, n=5000, prefijo="t")
            if len(c.credenciales) == 1 and not c.trabajo
        )
        tipo, entidad, _ = caso.credenciales[0]
        numero = f"10-123456-{digito}"
        caso = replace(
            caso,
            credenciales=((tipo, entidad, numero),),
            afiliaciones={f"{entidad}|{numero}": False},
        )
        _, ficha = deducir_ficha(caso, parametros, costumbres())
        self.assertEqual(ficha["cobertura"], "sin_cobertura")
        self.assertEqual(ficha["entidad_a_notificar"], entidad)
        self.assertIn("NOR-COB-04", ficha["fundamentos"])

    def test_busquedas_salen_del_buscador_real_de_la_entrega(self):
        conversaciones, auditoria, resumen = generar_una_vez()
        buscador = entrega.construir_buscador(RAIZ / "normas/original")
        self.assertEqual(resumen["buscador"]["clase"], type(buscador).__name__)
        self.assertEqual(resumen["buscador"]["k_entorno"], buscador.k_entorno)
        for conversacion, item in zip(conversaciones, auditoria):
            mensajes = conversacion["messages"]
            for i, mensaje in enumerate(mensajes[:-1]):
                if mensaje["role"] != "assistant":
                    continue
                accion = json.loads(mensaje["content"])
                if accion["herramienta"] != "buscar_normativa":
                    continue
                esperadas = buscador.recuperar(accion["argumentos"]["consulta"], buscador.k_entorno)
                self.assertEqual(len(esperadas), buscador.k_entorno)
                self.assertEqual(
                    mensajes[i + 1]["content"],
                    "Resultado de buscar_normativa:\n" + "\n\n".join(s.texto for s in esperadas),
                )
            seccion = SECCION_DEL_CIRCUITO.get(item["circuito"])
            if seccion:
                self.assertTrue(
                    any(f"## {seccion}" in texto for texto in observaciones_de_busqueda(conversacion)),
                    f"{conversacion['id']} no recupera {seccion}",
                )

    def test_generacion_es_determinista(self):
        primera = generar_una_vez()
        segunda = generar_sinteticos(RAIZ / "normas/original", RAIZ / "historial_costumbres.json")
        self.assertEqual(primera, segunda)
        self.assertEqual(primera[0][0]["id"], "S-12012-0001")
        self.assertEqual(primera[1][0]["id"], "S-12012-0001")


if __name__ == "__main__":
    unittest.main()
