"""Métricas de prompt con tokenizador y resultados artificiales."""

import hashlib
import json
import unittest

from admision.comparar import CONTROLES
from admision_tp.evaluacion_prompts import (
    calcular_metricas_candidato,
    clave_comparacion,
    medir_longitud_prompt,
)
from admision_tp.prompts import cargar_prompt_baseline


class TokenizadorArtificial:
    chat_template = "plantilla de chat artificial"

    def encode(self, texto, add_special_tokens):
        assert add_special_tokens is False
        return list(texto.encode("utf-8"))

    def apply_chat_template(self, mensajes, **opciones):
        assert opciones == {
            "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
            "padding": False, "truncation": False, "return_dict": True,
        }
        texto = "".join(f"<{m['role']}>{m['content']}" for m in mensajes) + "<assistant>"
        return {"input_ids": list(texto.encode("utf-8"))}


def resultado(caso, correctos, *, sin_decision=False):
    controles = {c: i < correctos for i, c in enumerate(CONTROLES)}
    return {
        "caso": caso, "controles": controles, "resuelto": all(controles.values()),
        "parcial": round(correctos / 9, 3), "obtenida": None if sin_decision else {},
        "invalidas": 1, "llamadas": 4, "busquedas": 2,
        "caracteres_de_contexto": 100, "segundos": 0.5,
    }


def traza(caso, con_json=True):
    respuesta = json.dumps({"herramienta": "leer_triage", "argumentos": {}}) if con_json else "sin json"
    return {"caso": caso, "mensajes": [{"role": "assistant", "content": respuesta}],
            "log": [{"herramienta": "buscar_normativa"}]}


class EvaluacionPromptsTest(unittest.TestCase):
    def test_mide_textos_y_contexto_con_tokenizador_inyectado(self):
        prompt, tokenizer = cargar_prompt_baseline(), TokenizadorArtificial()
        medicion = medir_longitud_prompt(prompt, tokenizer)
        self.assertEqual(medicion["por_texto"]["system_prompt"],
                         len(prompt.system_prompt.encode("utf-8")))
        self.assertEqual(medicion["total_textos_configurables"], sum(
            len(x.encode("utf-8")) for x in (
                prompt.system_prompt, prompt.mensaje_inicial,
                prompt.formato_observacion, prompt.mensaje_json_invalido)))
        self.assertGreater(medicion["contexto_inicial"], medicion["por_texto"]["system_prompt"])
        self.assertEqual(medicion["chat_template_sha256"],
                         hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())

    def test_rechaza_tokenizador_sin_plantilla_o_ids_anidados(self):
        prompt = cargar_prompt_baseline()
        with self.assertRaises(ValueError):
            medir_longitud_prompt(prompt, object())

        class Anidado(TokenizadorArtificial):
            def encode(self, texto, add_special_tokens):
                return [[1, 2]]
        with self.assertRaisesRegex(ValueError, "anidados"):
            medir_longitud_prompt(prompt, Anidado())

    def test_resume_nueve_controles_json_y_eficiencia(self):
        medicion = medir_longitud_prompt(cargar_prompt_baseline(), TokenizadorArtificial())
        metricas = calcular_metricas_candidato(
            [resultado("c1", 9), resultado("c2", 4, sin_decision=True)],
            [traza("c1"), traza("c2", con_json=False)], medicion)
        self.assertEqual((metricas["resueltos_9_de_9"], metricas["controles_correctos"]), (1, 13))
        self.assertEqual(metricas["controles_totales"], 18)
        self.assertEqual(metricas["respuestas_sin_json"], 1)
        self.assertEqual(metricas["llamadas_invalidas"], 2)
        self.assertEqual(metricas["busquedas"], {"total": 4, "medio": 2.0})
        self.assertEqual(metricas["segundos"]["total"], 1.0)
        self.assertEqual(metricas["puntaje_calidad"], 32)

    def test_comparacion_prioriza_casos_completos_y_luego_parciales(self):
        medicion = medir_longitud_prompt(cargar_prompt_baseline(), TokenizadorArtificial())
        completo = calcular_metricas_candidato(
            [resultado("c1", 9), resultado("c2", 0)], [traza("c1"), traza("c2")], medicion)
        parcial = calcular_metricas_candidato(
            [resultado("c1", 8), resultado("c2", 8)], [traza("c1"), traza("c2")], medicion)
        self.assertGreater(clave_comparacion(completo), clave_comparacion(parcial))
        mejor_parcial = calcular_metricas_candidato(
            [resultado("c1", 8), resultado("c2", 7)], [traza("c1"), traza("c2")], medicion)
        self.assertGreater(clave_comparacion(parcial), clave_comparacion(mejor_parcial))

    def test_rechaza_casos_desalineados_o_controles_incompletos(self):
        medicion = medir_longitud_prompt(cargar_prompt_baseline(), TokenizadorArtificial())
        with self.assertRaises(ValueError):
            calcular_metricas_candidato([resultado("c1", 9)], [traza("otro")], medicion)
        incompleto = resultado("c1", 9)
        incompleto["controles"].pop(CONTROLES[0])
        with self.assertRaises(ValueError):
            calcular_metricas_candidato([incompleto], [traza("c1")], medicion)


if __name__ == "__main__":
    unittest.main()
