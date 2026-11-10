from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

from admision.entorno import CAMPOS
from scripts import tabla_evaluacion as te

FICHA = {campo: None for campo in CAMPOS} | {"registro": "provisorio", "circuito": "ecg"}
PROMPT = {"nombre": "baseline", "version": 1, "sha256": "f" * 64, "archivo": "configuraciones/prompts/baseline.json"}


def escribir_practica(carpeta: Path) -> Path:
    practica = carpeta / "practica"
    practica.mkdir()
    for version in te.VERSIONES:
        filas = [{"caso": f"c{i}", "ficha": FICHA, "verificacion": {"requerida": None, "prohibidas": []}}
                 for i in range(2)]
        (practica / f"esperadas-{version}.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
    return practica


def escribir_corrida(carpeta: Path, version: str, resueltos: int, estado: str = "terminada",
                     adapter: str | None = None, prompt: dict | None = PROMPT) -> Path:
    (carpeta / "salida").mkdir(parents=True)
    trazas = [{"caso": f"c{i}", "mensajes": [{"role": "system", "content": "x" * 10},
                                             {"role": "assistant", "content": "y"}],
               "log": [{"herramienta": "buscar_normativa", "argumentos": {}, "valida": True}],
               "decision": FICHA if i < resueltos else {**FICHA, "circuito": "otro"}, "segundos": 3.0}
              for i in range(2)]
    (carpeta / "salida/trazas.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trazas), encoding="utf-8")
    (carpeta / "salida/tiempos.json").write_text(json.dumps(
        {"casos": 2, "construir_agente": 40.0, "jugar_lote": 30.0, "jugar_lote_por_caso": 15.0, "error": None}))
    runner = {"variante": version, "cantidad": 2, "estado": estado, "inicio": "2026-11-10T10:00:00+0000",
              "fin": "2026-11-10T10:10:00+0000", "error_agente": "CUDA" if estado == "fallida" else None,
              "configuracion": {"modelo": "google/gemma-4-E4B-it", "revision": "e" * 40, "lote": 4,
                                "adapter_repo": adapter, "adapter_revision": "a" * 40 if adapter else None,
                                "adapter_subcarpeta": "final", "metricas": f"/kaggle/working/x/{carpeta.name}/m.jsonl",
                                "dtype": "float16", "device_map": "auto"},
              "buscador": {"variante": "baseline"},
              "maquina": {"commit": "c" * 40, "sucio": False, "gpus": ["Tesla T4, 15360 MiB, 580"] * 2,
                          "torch": {"torch": "2.15.0+cu132"}, "python": "3.13"}}
    if prompt:
        runner["prompt"] = prompt
    (carpeta / "runner.json").write_text(json.dumps(runner), encoding="utf-8")
    return carpeta


class TablaEvaluacionTest(unittest.TestCase):
    def setUp(self):
        salida = redirect_stdout(io.StringIO())  # main imprime la tabla.
        salida.__enter__()
        self.addCleanup(salida.__exit__, None, None, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.practica = escribir_practica(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def _evaluacion(self) -> Path:
        """Una carpeta de plan con original terminada, b fallida y c sin runner (murió antes de arrancar)."""
        evaluacion = self.dir / "eval"
        escribir_corrida(evaluacion / "sin_entrenar-original", "original", 2)
        escribir_corrida(evaluacion / "sin_entrenar-b", "b", 0, estado="fallida")
        filas = [{"etiqueta": "sin_entrenar", "version": v, "carpeta": f"sin_entrenar-{v}", "estado": e,
                  "error": err} for v, e, err in (("original", "terminada", None), ("b", "fallida", None),
                                                  ("c", "fallida", "ValueError: prompt"))]
        (evaluacion / "evaluacion.json").write_text(json.dumps(
            {"comando": ["python", "infra/kaggle_evaluacion.py"], "plan_sha256": "9" * 64, "corridas": filas}))
        return evaluacion

    def test_tabla_con_faltantes_referencia_y_procedencia(self):
        entrenado = escribir_corrida(self.dir / "suelta", "c", 1, adapter="org/repo")
        salida = self.dir / "tabla"
        te.main(["--evaluacion", str(self._evaluacion()), "--corrida", f"ambos={entrenado}",
                 "--practica", str(self.practica), "--salida", str(salida)])
        tabla = json.loads((salida / "tabla.json").read_text(encoding="utf-8"))
        md = (salida / "tabla.md").read_text(encoding="utf-8")
        celdas = {(c["etiqueta"], c["version"]): c for c in tabla["celdas"]}
        self.assertEqual(set(celdas), {("sin_entrenar", "original"), ("ambos", "c")})
        self.assertEqual((celdas[("sin_entrenar", "original")]["resueltos"], celdas[("ambos", "c")]["resueltos"]),
                         (2, 1))
        # El kit redondea cada caso (0,889) y después el promedio.
        self.assertEqual(celdas[("ambos", "c")]["parcial_medio"], 0.945)
        self.assertEqual(celdas[("ambos", "c")]["caracteres_de_contexto_por_caso"], 10)
        self.assertEqual(celdas[("ambos", "c")]["segundos_por_caso"], 15.0)
        faltantes = {(f["etiqueta"], f["version"]): f["faltante"] for f in tabla["faltantes"]}
        self.assertIn("CUDA", faltantes[("sin_entrenar", "b")])
        self.assertIn("ValueError: prompt", faltantes[("sin_entrenar", "c")])
        self.assertEqual(faltantes[("solo_prompt", "original")], "sin corrida")
        self.assertEqual(len(faltantes), 10)
        self.assertIn("| sin_entrenar | 2/2 | — | — | sin entrenar | baseline v1 (ffffffffffff) |", md)
        self.assertIn("| ambos | — | — | 1/2 | org/repo@" + "a" * 40 + "/final |", md)
        self.assertIn("| *Referencia de la cátedra: sin entrenar* | 12/50 | 15/50 | 12/50 |", md)
        self.assertIn("Referencia de la cátedra: fine-tuning supervisado simple", md)
        self.assertIn("no se usó para seleccionar", md)
        self.assertTrue((salida / "corridas/ambos-c/resultados.jsonl").is_file())

    def test_procedencia_completa_por_numero(self):
        corrida = escribir_corrida(self.dir / "r", "original", 1, prompt=None)
        salida = self.dir / "tabla"
        te.main(["--corrida", f"sin_entrenar={corrida}", "--practica", str(self.practica), "--salida", str(salida)])
        celda = json.loads((salida / "tabla.json").read_text(encoding="utf-8"))["celdas"][0]
        p = celda["procedencia"]
        for clave in ("commit", "sucio", "modelo", "checkpoint", "prompt", "buscador", "hardware", "trazas_sha256",
                      "esperadas_sha256", "comando_jugar", "comando_puntuar", "inicio", "fin"):
            self.assertIn(clave, p)
        self.assertEqual((p["commit"], p["sucio"], p["checkpoint"]), ("c" * 40, False, "sin entrenar"))
        self.assertEqual(len(p["trazas_sha256"]), 64)
        self.assertIn("MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline python infra/kaggle_baseline.py --variante original",
                      p["comando_jugar"])
        self.assertTrue(any("no registra el prompt" in a for a in celda["avisos"]))
        self.assertTrue(any("A100" in a for a in celda["avisos"]))

    def test_no_pisa_una_tabla_ni_acepta_corridas_sin_procedencia(self):
        corrida = escribir_corrida(self.dir / "r", "original", 1)
        salida = self.dir / "tabla"
        salida.mkdir()
        with self.assertRaises(FileExistsError):
            te.main(["--corrida", f"x={corrida}", "--practica", str(self.practica), "--salida", str(salida)])
        (self.dir / "sin_runner/salida").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "runner.json"):
            te.main(["--corrida", f"x={self.dir / 'sin_runner'}", "--salida", str(self.dir / "otra")])

    def test_dos_corridas_validas_para_la_misma_celda_fallan(self):
        a = escribir_corrida(self.dir / "a", "original", 1)
        b = escribir_corrida(self.dir / "b", "original", 2)
        with self.assertRaisesRegex(ValueError, "misma configuración"):
            te.main(["--corrida", f"x={a}", "--corrida", f"x={b}", "--practica", str(self.practica),
                     "--salida", str(self.dir / "tabla")])
        self.assertFalse((self.dir / "tabla").exists())


if __name__ == "__main__":
    unittest.main()
