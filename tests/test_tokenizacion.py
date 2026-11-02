"""Máscaras y artefactos con tokenizador artificial; no prueba Gemma real."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admision_tp.preparacion import preparar_corrida
from admision_tp.tokenizacion import tokenizar_conversacion, tokenizar_preparacion
from test_conversaciones import registro


class TokenizadorArtificial:
    chat_template = "plantilla artificial para test"

    def apply_chat_template(self, mensajes, **opciones):
        assert opciones["enable_thinking"] is False
        assert opciones["padding"] is False and opciones["truncation"] is False
        texto = "<BOS>" + "".join(
            f"<{m['role']}>" + m["content"] + "<FIN>" for m in mensajes)
        if opciones["add_generation_prompt"]:
            texto += "<assistant>"
        return {"input_ids": list(map(ord, texto)), "attention_mask": [1] * len(texto)}


class TokenizacionTest(unittest.TestCase):
    def test_solo_respuestas_y_cierre_nunca_contexto(self):
        dato, tokenizer = registro(), TokenizadorArtificial()
        antes = copy.deepcopy(dato)
        resultado = tokenizar_conversacion(dato, tokenizer, 8192)
        esperado = "".join(m["content"] + "<FIN>" for m in dato["messages"] if m["role"] == "assistant")
        supervisado = "".join(chr(x) for x in resultado["labels"] if x != -100)
        self.assertEqual(supervisado, esperado)
        self.assertNotIn("<assistant>", supervisado)
        self.assertEqual(dato, antes)
        self.assertEqual(resultado["attention_mask"], [1] * len(resultado["input_ids"]))
        for entrada, etiqueta in zip(resultado["input_ids"], resultado["labels"]):
            self.assertIn(etiqueta, (-100, entrada))

    def test_limite_exacto_y_rechazo_sin_recortar(self):
        tokenizer, dato = TokenizadorArtificial(), registro()
        largo = len(tokenizar_conversacion(dato, tokenizer, 8192)["input_ids"])
        self.assertEqual(len(tokenizar_conversacion(dato, tokenizer, largo)["labels"]), largo)
        for limite in (largo - 1, 0, True):
            with self.subTest(limite=limite), self.assertRaises(ValueError):
                tokenizar_conversacion(dato, tokenizer, limite)

    def test_rechaza_prefijos_cambiantes_y_padding(self):
        class Inestable(TokenizadorArtificial):
            def apply_chat_template(self, mensajes, **opciones):
                salida = super().apply_chat_template(mensajes, **opciones)
                if opciones["add_generation_prompt"]:
                    salida["input_ids"][0] += 1
                return salida
        class Padding(TokenizadorArtificial):
            def apply_chat_template(self, mensajes, **opciones):
                salida = super().apply_chat_template(mensajes, **opciones)
                salida["attention_mask"][0] = 0
                return salida
        for tokenizer in (Inestable(), Padding()):
            with self.subTest(tokenizer=type(tokenizer).__name__), self.assertRaises(ValueError):
                tokenizar_conversacion(registro(), tokenizer, 8192)

    def test_artefacto_hashes_y_snapshots_intactos(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            datos, config = raiz / "datos.jsonl", raiz / "config.json"
            datos.write_text(json.dumps(registro()) + "\n")
            config.write_bytes(Path("configuraciones/sft-ejemplo.json").read_bytes())
            preparar_corrida(datos, config, raiz / "preparada")
            origen = raiz / "preparada"
            antes = {p.name: p.read_bytes() for p in origen.iterdir()}
            with patch("admision_tp.tokenizacion._codigo", return_value={"commit": "a" * 40, "dirty": True}):
                informe = tokenizar_preparacion(origen, raiz / "tokens", TokenizadorArtificial())
                otro = tokenizar_preparacion(origen, raiz / "otros", TokenizadorArtificial())
            self.assertEqual(informe, otro)
            self.assertEqual(informe["registros"], 1)
            self.assertGreater(informe["supervisados"], 0)
            contenido = (raiz / "tokens/tokens.jsonl").read_bytes()
            self.assertEqual(informe["archivos"]["tokens.jsonl"], hashlib.sha256(contenido).hexdigest())
            self.assertEqual(antes, {p.name: p.read_bytes() for p in origen.iterdir()})
            with self.assertRaises(ValueError):
                tokenizar_preparacion(origen, raiz / "tokens", TokenizadorArtificial())
            (origen / "conversaciones.jsonl").write_text("alterado")
            with self.assertRaises(ValueError):
                tokenizar_preparacion(origen, raiz / "no-creada", TokenizadorArtificial())
            self.assertFalse((raiz / "no-creada").exists())


@unittest.skipUnless(os.getenv("TP_TOKENIZADOR_LOCAL"), "requiere snapshot real local explícito")
class TokenizadorRealTest(unittest.TestCase):
    def test_gemma_fijado_solo_supervisa_respuestas(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["TP_TOKENIZADOR_LOCAL"], local_files_only=True)
        self.assertEqual(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                         "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5")
        dato = registro()
        resultado = tokenizar_conversacion(dato, tokenizer, 8192)
        supervisado = tokenizer.decode([x for x in resultado["labels"] if x != -100],
                                        skip_special_tokens=False)
        esperado = "".join(m["content"] + "<turn|>\n" for m in dato["messages"]
                            if m["role"] == "assistant")
        self.assertEqual(supervisado, esperado)
        self.assertEqual(len(resultado["input_ids"]), 167)
        self.assertEqual(sum(x != -100 for x in resultado["labels"]), 126)
        with self.assertRaises(ValueError):
            tokenizar_conversacion(dato, tokenizer, 166)


if __name__ == "__main__":
    unittest.main()
