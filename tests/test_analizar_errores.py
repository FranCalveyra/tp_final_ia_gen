from contextlib import redirect_stdout
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from admision.casos import Caso
from admision.comparar import CONTROLES
from admision.entorno import Entorno
from scripts import analizar_errores as ae

RAIZ = Path(__file__).resolve().parents[1]
SECCIONES = {
    "NOR-TDE-01": "## NOR-TDE-01 · Dolor torácico\n\n§1. Se pide un ECG dentro de P-TDE-1.\nValores: P-TDE-1 = 10 min.",
    "NOR-GUA-02": "## NOR-GUA-02 · Ubicación en box según el nivel de triage\n\n§1. Texto de guardia.",
    "NOR-COB-02": "## NOR-COB-02 · Urgencias con cobertura\n\n§2. Se verifica la afiliación.",
}


class BuscadorFalso:
    """Devuelve secciones fijas según la consulta (sin BM25)."""
    k_entorno = 2

    def recuperar(self, consulta, k):
        if "torácico" in consulta:
            return [SECCIONES["NOR-TDE-01"]]
        if "box" in consulta:
            return [SECCIONES["NOR-GUA-02"]]  # la rectora (NOR-ADM-0x) no aparece: falla del buscador
        return [SECCIONES["NOR-COB-02"]]


def caso(id_, edad=50, nivel=3, motivo="dolor torácico", credenciales=(("prepaga", "Prepaga Sierra", "100/01"),),
         acompanante="su hija Laura", afiliaciones=None):
    return Caso.desde_dict({
        "id": id_, "paciente_id": f"P-{id_[-1]}000", "nombre": "Ana Paz", "edad": edad, "nivel": nivel,
        "motivo": motivo, "nota": "Nota.", "sepsis": "no", "llegada": "por sus propios medios",
        "documento_tipo": "DNI", "documento_numero": "12.345.678", "credenciales": [list(c) for c in credenciales],
        "puede_responder": True, "no_responde_porque": None, "acompanante": acompanante, "trabajo": None,
        "declarada": list(credenciales[0]) if credenciales else None, "art": None,
        "afiliaciones": afiliaciones or {f"{c[1]}|{c[2]}": True for c in credenciales}})


CASOS = {
    "c1": caso("c1"),                                                       # ECG simple
    "c2": caso("c2", edad=80, motivo="dolor abdominal", acompanante=None),  # mayor: box y aviso
    "c3": caso("c3"),                                                       # sin JSON
    "c4": caso("c4", edad=70, motivo="dolor abdominal"),                   # mayor: el buscador no trae ADM
    "c5": caso("c5", motivo="dolor abdominal",                              # dos credenciales
               credenciales=(("obra_social", "Obra Social Bermejo", "200/01"), ("prepaga", "Prepaga Cumbre", "300/01"))),
    "c6": caso("c6"),
}


def ficha(**cambios):
    base = {"registro": "definitivo", "numero_documento": "12.345.678", "box_antes_del_registro": False,
            "circuito": "ecg", "plazo_circuito_min": 10, "cobertura": "prepaga", "entidad_a_notificar": "Prepaga Sierra",
            "aviso_al_voluntariado": False}
    return {**base, **cambios}


NINGUNO = {"circuito": "ninguno", "plazo_circuito_min": None}
ESPERADAS = {
    "c1": (ficha(), ["Prepaga Sierra", "100/01"], []),
    "c2": (ficha(box_antes_del_registro=True, aviso_al_voluntariado=True, **NINGUNO), ["Prepaga Sierra", "100/01"], []),
    "c3": (ficha(), ["Prepaga Sierra", "100/01"], []),
    "c4": (ficha(box_antes_del_registro=True, **NINGUNO), ["Prepaga Sierra", "100/01"], []),
    "c5": (ficha(entidad_a_notificar="Prepaga Cumbre", **NINGUNO), ["Prepaga Cumbre", "300/01"], ["Obra Social Bermejo"]),
    "c6": (ficha(), ["Prepaga Sierra", "100/01"], []),
}


def jugar(c: Caso, pasos: list[tuple[str, dict]], sin_json: int = 0) -> dict:
    """Traza del kit: los pasos van por `Entorno.paso` real; `sin_json` respuestas sin JSON al final."""
    entorno = Entorno(c, BuscadorFalso())
    mensajes = [{"role": "system", "content": "sistema"}, {"role": "user", "content": f"Ingresó {c.paciente_id}."}]
    for herramienta, argumentos in pasos:
        mensajes.append({"role": "assistant", "content": json.dumps({"herramienta": herramienta,
                                                                     "argumentos": argumentos})})
        observacion, _ = entorno.paso(herramienta, argumentos)
        mensajes.append({"role": "user", "content": f"Resultado de {herramienta}:\n{observacion}"})
    for i in range(sin_json):
        mensajes.append({"role": "assistant", "content": "no sé qué hacer"})
        if i < sin_json - 1:
            mensajes.append({"role": "user", "content": "No entendí tu respuesta."})
    if entorno.decision is not None:
        mensajes.pop()  # la última observación termina el caso y no vuelve al modelo
    return {"caso": c.id, "mensajes": mensajes, "log": entorno.log, "decision": entorno.decision, "segundos": 1.5}


def guion(id_, decision, busqueda=None, verificar=None):
    c = CASOS[id_]
    pasos = [("leer_triage", {"paciente_id": c.paciente_id})]
    if busqueda:
        pasos.append(("buscar_normativa", {"consulta": busqueda}))
    if verificar:
        pasos.append(("verificar_afiliacion", {"entidad": verificar[0], "numero_afiliado": verificar[1]}))
    pasos.append(("registrar_decision", {**decision, "fundamentos": []}))
    return jugar(c, pasos)


def trazas_antes() -> list[dict]:
    return [
        guion("c1", ficha(plazo_circuito_min=15), "dolor torácico plazo", ("Prepaga Sierra", "100/01")),
        guion("c2", ficha(**NINGUNO), None, ("Prepaga Sierra", "100/01")),
        jugar(CASOS["c3"], [("leer_triage", {"paciente_id": CASOS["c3"].paciente_id})], sin_json=3),
        guion("c4", ficha(**NINGUNO), "ubicación en el box del paciente mayor", ("Prepaga Sierra", "100/01")),
        guion("c5", ficha(cobertura="obra_social", entidad_a_notificar="Obra Social Bermejo", **NINGUNO), None,
              ("Obra Social Bermejo", "200/01")),
        guion("c6", ficha(), "dolor torácico plazo", ("Prepaga Sierra", "100/01")),
    ]


def trazas_despues() -> list[dict]:
    trazas = trazas_antes()
    trazas[0] = guion("c1", ficha(), "dolor torácico plazo", ("Prepaga Sierra", "100/01"))
    trazas[5] = guion("c6", ficha(registro="provisorio"), "dolor torácico plazo", ("Prepaga Sierra", "100/01"))
    return trazas


def escribir_jsonl(ruta: Path, filas) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8")


class AnalizarErroresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.casos = self.tmp / "casos.jsonl"
        escribir_jsonl(self.casos, [c.como_dict() for c in CASOS.values()])
        self.esperadas = self.tmp / "esperadas.jsonl"
        escribir_jsonl(self.esperadas, [{"caso": k, "ficha": f, "verificacion": {"requerida": r, "prohibidas": p}}
                                        for k, (f, r, p) in ESPERADAS.items()])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def corrida_runner(self, nombre: str, trazas: list[dict]) -> Path:
        carpeta = self.tmp / nombre
        escribir_jsonl(carpeta / "salida/trazas.jsonl", trazas)
        (carpeta / "salida/tiempos.json").write_text(json.dumps({"jugar_lote_por_caso": 2.5}), encoding="utf-8")
        runner = {"variante": "original", "cantidad": len(trazas), "estado": "terminada", "prompt": None,
                  "buscador": {"variante": "baseline"}, "inicio": "a", "fin": "b",
                  "configuracion": {"modelo": "google/gemma", "revision": "r1", "lote": 4, "adapter_repo": None},
                  "maquina": {"commit": "abc123", "sucio": False, "gpus": ["Tesla T4"]}}
        (carpeta / "runner.json").write_text(json.dumps(runner), encoding="utf-8")
        return carpeta

    def corrida_manifiesto(self, nombre: str, trazas: list[dict]) -> Path:
        carpeta = self.tmp / nombre
        escribir_jsonl(carpeta / "trazas.jsonl", trazas)
        (carpeta / "entradas/normas").mkdir(parents=True)
        shutil.copy(self.casos, carpeta / "entradas/casos.jsonl")
        shutil.copy(self.esperadas, carpeta / "entradas/esperadas.jsonl")
        shutil.copy(RAIZ / "normas/original/parametros.yaml", carpeta / "entradas/normas/parametros.yaml")
        manifiesto = {"codigo": {"commit": "def456", "dirty": False}, "comparable": False, "proposito": "plomeria",
                      "estado": "completada", "prompt": {"nombre": "baseline", "version": 1, "sha256": "f" * 64},
                      "configuracion": {"modelo": "google/gemma", "revision": "r1", "buscador_variante": "baseline"}}
        (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
        return carpeta

    def analizar(self, carpeta: Path) -> dict:
        return ae.analizar(ae.cargar_corrida(carpeta, casos=self.casos, esperadas=self.esperadas))

    def test_cada_regla_de_fuente(self):
        an = self.analizar(self.corrida_runner("antes", trazas_antes()))
        fallas = {c["caso"]: c["fallas"] for c in an["casos"]}
        # Recuperó la rectora (NOR-TDE-01) y decidió mal.
        self.assertEqual(set(fallas["c1"]), {"plazo_circuito_min"})
        self.assertEqual((fallas["c1"]["plazo_circuito_min"]["fuente"], fallas["c1"]["plazo_circuito_min"]["subtipo"]),
                         ("modelo", "busco_y_decidio_mal"))
        # Nunca buscó: la rectora falta pero no hubo búsqueda relevante -> modelo, no buscador.
        self.assertEqual(fallas["c2"]["box_antes_del_registro"]["fuente"], "modelo")
        self.assertEqual(fallas["c2"]["box_antes_del_registro"]["subtipo"], "no_busco")
        self.assertEqual(fallas["c2"]["box_antes_del_registro"]["faltantes"], ["NOR-ADM-01", "NOR-ADM-02"])
        # Control decidido por costumbre.
        self.assertEqual(fallas["c2"]["aviso_al_voluntariado"]["fuente"], "costumbre")
        # Sin JSON: los 9 controles fallan por formato.
        self.assertEqual(set(fallas["c3"]), set(CONTROLES))
        self.assertEqual({(f["fuente"], f["subtipo"]) for f in fallas["c3"].values()}, {("formato", "sin_json")})
        # Buscó «box» y el buscador no trajo NOR-ADM-01/02.
        self.assertEqual(fallas["c4"]["box_antes_del_registro"]["fuente"], "buscador")
        self.assertEqual(fallas["c4"]["box_antes_del_registro"]["faltantes"], ["NOR-ADM-01", "NOR-ADM-02"])
        # Dos credenciales: verificó la que la costumbre no elige.
        self.assertEqual({k: f["fuente"] for k, f in fallas["c5"].items()},
                         {"cobertura": "costumbre", "entidad_a_notificar": "costumbre", "verificacion": "costumbre"})
        self.assertEqual(fallas["c6"], {})
        self.assertEqual(an["formato"]["sin_decision"], {"sin_json": 1})
        self.assertEqual(an["secciones_faltantes_buscador"], {"NOR-ADM-01": 1, "NOR-ADM-02": 1})

    def test_argumentos_invalidos_y_guarda_de_contexto_son_formato(self):
        c = CASOS["c6"]
        traza = jugar(c, [("leer_triage", {"paciente_id": c.paciente_id}),
                          ("preguntar", {"paciente_id": c.paciente_id, "a_quien": "medico", "tema": "cobertura"}),
                          ("registrar_decision", {**ficha(cobertura="sin_cobertura", entidad_a_notificar=None),
                                                  "fundamentos": []})])
        an = self.analizar(self.corrida_runner("invalidos", [traza]))
        self.assertEqual(an["casos"][0]["fallas"]["cobertura"]["subtipo"], "argumentos_invalidos:preguntar")
        guarda = jugar(c, [("leer_triage", {"paciente_id": c.paciente_id})])
        guarda["mensajes"].append({"role": "user", "content": "[agente] Caso terminado sin registrar decisión: x"})
        self.assertEqual(ae.formato_del_caso(guarda)["sin_decision"], "guarda_contexto")

    def test_por_control_igual_al_kit_y_a_los_resultados_previos(self):
        carpeta = self.corrida_runner("antes", trazas_antes())
        an = self.analizar(carpeta)
        self.assertTrue(an["cruce"]["por_control_igual_al_kit"])
        for control in CONTROLES:
            self.assertEqual(an["por_control"][control]["acierto"], an["resumen_kit"]["por_control"][control])
            self.assertEqual(sum(an["por_control"][control]["fuentes"].values()), an["por_control"][control]["fallas"])
        resultados = [{"caso": c["caso"], "resuelto": c["resuelto"], "controles": c["controles"]} for c in an["casos"]]
        resultados[0]["controles"] = {**resultados[0]["controles"], "registro": False}
        escribir_jsonl(carpeta / "puntuacion/resultados.jsonl", resultados)
        cruce = self.analizar(carpeta)["cruce"]["resultados_previos"]
        self.assertEqual((cruce["coincide"], cruce["casos_distintos"]), (False, ["c1"]))

    def test_agrupa_por_tipo_de_caso(self):
        grupos = self.analizar(self.corrida_runner("antes", trazas_antes()))["grupos"]
        self.assertEqual(grupos["credenciales"]["dos"]["ids"], ["c5"])
        self.assertEqual(grupos["edad_mayor_P-ADM-2"]["sí"]["ids"], ["c2", "c4"])
        self.assertEqual(grupos["edad_mayor_P-ADM-2"]["sí"]["por_control"]["box_antes_del_registro"], 2)
        self.assertEqual(grupos["circuito"]["ecg"]["casos"], 3)
        self.assertEqual(grupos["circuito"]["ecg"]["resueltos"], 1)

    def test_compara_ganados_perdidos_y_cambios(self):
        a, b = self.corrida_runner("antes", trazas_antes()), self.corrida_manifiesto("despues", trazas_despues())
        ca, cb = ae.cargar_corrida(a, casos=self.casos, esperadas=self.esperadas), ae.cargar_corrida(b)
        cmp = ae.comparar_corridas(ae.analizar(ca), ae.analizar(cb), trazas_antes=ca["trazas"],
                                   trazas_despues=cb["trazas"])
        self.assertEqual((cmp["ganados"], cmp["perdidos"]), (["c1"], ["c6"]))
        self.assertEqual(cmp["por_control"]["plazo_circuito_min"]["arreglados"], ["c1"])
        self.assertEqual(cmp["por_control"]["registro"]["rotos"], ["c6"])
        self.assertEqual(cmp["transiciones_de_fuente"], {"modelo→ok": 1, "ok→modelo": 1})
        self.assertEqual(sorted(cmp["conversaciones_distintas"]), ["c1", "c6"])
        self.assertEqual(cmp["costo"]["busquedas"]["antes"]["total"], 3)

    def test_corridas_identicas_no_tienen_diferencias(self):
        a, b = self.corrida_runner("a", trazas_antes()), self.corrida_manifiesto("b", trazas_antes())
        ca, cb = ae.cargar_corrida(a, casos=self.casos, esperadas=self.esperadas), ae.cargar_corrida(b)
        cmp = ae.comparar_corridas(ae.analizar(ca), ae.analizar(cb), trazas_antes=ca["trazas"],
                                   trazas_despues=cb["trazas"])
        self.assertEqual(cmp["diferencias"], 0)

    def test_rechaza_conjuntos_distintos_salvo_interseccion_explicita(self):
        a, b = self.corrida_runner("a", trazas_antes()), self.corrida_runner("b", trazas_antes()[:4])
        ca, cb = (ae.cargar_corrida(x, casos=self.casos, esperadas=self.esperadas) for x in (a, b))
        aa, ab = ae.analizar(ca), ae.analizar(cb)
        with self.assertRaises(ValueError):
            ae.comparar_corridas(aa, ab, trazas_antes=ca["trazas"], trazas_despues=cb["trazas"])
        cmp = ae.comparar_corridas(aa, ab, trazas_antes=ca["trazas"], trazas_despues=cb["trazas"],
                                   permitir_interseccion=True)
        self.assertTrue(cmp["interseccion"])
        self.assertEqual((cmp["casos_comparados"], cmp["solo_antes"]), (4, ["c5", "c6"]))

    def test_procedencia_de_runner_manifiesto_y_trazas_sueltas(self):
        carpeta = self.corrida_runner("antes", trazas_antes())
        p = self.analizar(carpeta)["procedencia"]
        self.assertEqual((p["origen"], p["commit"], p["sucio"], p["adapter"]), ("runner.json", "abc123", False,
                                                                               "sin entrenar"))
        self.assertEqual(p["buscador"], {"variante": "baseline"})
        self.assertEqual(p["trazas_sha256"],
                         hashlib.sha256((carpeta / "salida/trazas.jsonl").read_bytes()).hexdigest())
        m = ae.analizar(ae.cargar_corrida(self.corrida_manifiesto("m", trazas_antes())))
        self.assertEqual((m["procedencia"]["origen"], m["procedencia"]["prompt"]["sha256"]),
                         ("manifiesto.json", "f" * 64))
        self.assertTrue(any("no comparable" in a for a in m["avisos"]))
        suelta = self.tmp / "suelta"
        escribir_jsonl(suelta / "salida/trazas.jsonl", trazas_antes())
        with self.assertRaises(ae.CorridaInvalida):
            ae.cargar_corrida(suelta)
        an = ae.analizar(ae.cargar_corrida(suelta, version="original", casos=self.casos, esperadas=self.esperadas))
        self.assertIsNone(an["procedencia"]["origen"])
        self.assertTrue(any("sin runner.json" in a for a in an["avisos"]))

    def test_cli_escribe_reportes_y_no_pisa(self):
        carpeta = self.corrida_runner("antes", trazas_antes())
        salida = self.tmp / "analisis"
        argv = ["analizar", "--corrida", str(carpeta), "--casos", str(self.casos), "--esperadas",
                str(self.esperadas), "--salida", str(salida)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ae.main(argv), 0)
        datos = json.loads((salida / "analisis.json").read_text(encoding="utf-8"))
        self.assertEqual(datos["resumen_kit"]["resueltos"], 1)
        self.assertIn("comando", datos)
        self.assertIn("| box_antes_del_registro |", (salida / "analisis.md").read_text(encoding="utf-8"))
        antes = (salida / "analisis.json").read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            ae.main(argv)
        self.assertEqual((salida / "analisis.json").read_bytes(), antes)
        with self.assertRaises(FileExistsError):
            ae._crear_salida(salida)


if __name__ == "__main__":
    unittest.main()
