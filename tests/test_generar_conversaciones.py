"""Dataset v2 de conversaciones (issue #8), probado sobre los datos reales del repo en CPU.

Se genera una sola vez con un prompt distinto del baseline en los cuatro textos, para
comprobar que el script usa el PromptSpec recibido y no textos fijos.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from admision.casos import cargar_casos
from admision.entorno import nota_de_triage
from admision_tp.buscador import BuscadorBM25
from admision_tp.conversaciones import validar_archivo
from admision_tp.normas_variables import escribir_normas
from admision_tp.prompts import PromptSpec
from admision_tp.trazas_normativas import fila_desde_traza, parametros_observados, secciones_en
from scripts.clasificar_historial import razones_normativas
from scripts.generar_conversaciones import (
    Excluida,
    completar_parametros,
    curar_traza,
    secciones_rectoras,
    generar_dataset,
    huellas_practica,
    observar_busqueda,
    valores_de_traza,
    verificar_no_practica,
)

RAIZ = Path(__file__).resolve().parents[1]
PROMPT = PromptSpec(version=1, nombre="prueba-issue8", system_prompt="SISTEMA DE PRUEBA",
                    mensaje_inicial="Paciente {paciente_id}: completá la ficha.",
                    formato_observacion="<{herramienta}>\n{observacion}\n</{herramienta}>",
                    mensaje_json_invalido="Respondé con JSON.")


def leer_jsonl(ruta):
    return [json.loads(linea) for linea in Path(ruta).read_text(encoding="utf-8").splitlines()]


def unidades():
    import yaml
    datos = yaml.safe_load((RAIZ / "normas/original/parametros.yaml").read_text(encoding="utf-8"))
    return {p["codigo"]: p["unidad"] for p in datos["parametros"]}


def como_traza(conversacion):
    """Pasa una conversación del prompt de prueba al formato de pasos del kit."""
    mensajes, pasos = conversacion["messages"], []
    for i in range(2, len(mensajes), 2):
        accion = json.loads(mensajes[i]["content"])
        resultado = "Ficha registrada."
        if i + 1 < len(mensajes):
            herramienta = accion["herramienta"]
            contenido = mensajes[i + 1]["content"]
            prefijo, sufijo = f"<{herramienta}>\n", f"\n</{herramienta}>"
            assert contenido.startswith(prefijo) and contenido.endswith(sufijo)
            resultado = contenido[len(prefijo):-len(sufijo)]
        pasos.append({"orden": len(pasos) + 1, "herramienta": accion["herramienta"],
                      "argumentos": accion["argumentos"], "resultado": resultado})
    return {"id": conversacion["id"], "pasos": pasos}


class DatasetGeneradoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporal = tempfile.TemporaryDirectory()
        carpeta = Path(cls.temporal.name)
        ruta_prompt = carpeta / "prompt.json"
        ruta_prompt.write_text(json.dumps(PROMPT.como_dict(), ensure_ascii=False), encoding="utf-8")
        cls.salida = carpeta / "dataset"
        cls.manifiesto = generar_dataset(cls.salida, ruta_prompt=ruta_prompt, variante="baseline")
        cls.conversaciones = leer_jsonl(cls.salida / "conversaciones.jsonl")
        cls.auditoria = {a["id"]: a for a in leer_jsonl(cls.salida / "auditoria.jsonl")}

    @classmethod
    def tearDownClass(cls):
        cls.temporal.cleanup()

    def test_tres_origenes_validos_con_el_formato_del_prompt(self):
        informe = validar_archivo(self.salida / "conversaciones.jsonl", PROMPT.formato_observacion)
        self.assertEqual(informe["errores"], [])
        self.assertEqual(set(informe["por_origen"]), {"referencia", "historial", "sintetico"})
        self.assertEqual(informe["por_origen"], self.manifiesto["por_origen"])
        self.assertEqual(self.manifiesto["prompt"]["sha256"], PROMPT.sha256)
        self.assertEqual(self.manifiesto["buscador"]["variante"], "baseline")
        self.assertEqual(self.manifiesto["salida"]["conversaciones_sha256"], informe["sha256"])
        excluidas = sum(sum(m.values()) for m in self.manifiesto["exclusiones"].values())
        self.assertEqual(sum(self.manifiesto["entradas_por_origen"].values()) - excluidas, informe["validos"])

    def test_prompt_spec_en_todas_las_conversaciones(self):
        for conversacion in self.conversaciones:
            mensajes = conversacion["messages"]
            paciente = json.loads(mensajes[2]["content"])["argumentos"]["paciente_id"]
            self.assertEqual(mensajes[0]["content"], PROMPT.system_prompt)
            self.assertEqual(mensajes[1]["content"], PROMPT.mensaje_inicial_para(paciente))
            self.assertNotIn("Resultado de ", "".join(m["content"] for m in mensajes[3::2]))

    def test_busquedas_iguales_al_buscador_real_con_los_valores_de_cada_conversacion(self):
        with tempfile.TemporaryDirectory() as carpeta:
            buscadores = {}
            for indice, conversacion in enumerate(self.conversaciones[::9]):
                valores = self.auditoria[conversacion["id"]]["parametros"]
                clave = tuple(sorted(valores.items()))
                if clave not in buscadores:
                    ruta = escribir_normas(RAIZ / "normas/original", Path(carpeta) / str(indice), valores)
                    buscadores[clave] = BuscadorBM25(ruta, variante="baseline")
                for paso in como_traza(conversacion)["pasos"]:
                    if paso["herramienta"] == "buscar_normativa":
                        esperado = observar_busqueda(buscadores[clave], paso["argumentos"]["consulta"])
                        self.assertEqual(paso["resultado"], esperado)

    def test_valores_de_referencia_conservados_y_historial_original(self):
        trazas = {t["id"]: t for t in leer_jsonl(RAIZ / "entrenamiento/trazas.jsonl")}
        originales = self.manifiesto["parametros"]["originales"]
        for conversacion in self.conversaciones:
            visibles = valores_de_traza(como_traza(conversacion), unidades())
            auditados = self.auditoria[conversacion["id"]]["parametros"]
            self.assertTrue(all(auditados.get(c, originales[c]) == v for c, v in visibles.items()))
            if conversacion["origen"] == "referencia":
                propios = valores_de_traza(trazas[conversacion["id"]], unidades())
                self.assertEqual(propios, {c: auditados[c] for c in propios})
            if conversacion["origen"] == "historial":
                self.assertTrue(all(v == originales[c] for c, v in auditados.items()))

    def test_fichas_correctas_para_sus_parametros_y_sinteticos_variados(self):
        originales = self.manifiesto["parametros"]["originales"]
        invalidas_con_originales = 0
        for conversacion in self.conversaciones:
            fila = fila_desde_traza(como_traza(conversacion))
            parametros = {**originales, **self.auditoria[conversacion["id"]]["parametros"]}
            self.assertEqual(razones_normativas(fila, parametros), [], conversacion["id"])
            if conversacion["origen"] == "sintetico":
                invalidas_con_originales += razones_normativas(fila, originales) != []
        sinteticos = [a for a in self.auditoria.values() if a["origen"] == "sintetico"]
        self.assertGreater(len({tuple(sorted(a["parametros"].items())) for a in sinteticos}), 100)
        # Los valores variados cambian fichas: muchas serían erróneas con normas/original.
        self.assertGreater(invalidas_con_originales, 100)

    def test_todas_recuperan_rectoras_y_solo_citan_fundamentos_recuperados(self):
        for auditoria in self.auditoria.values():
            self.assertEqual(auditoria["rectoras_faltantes"], [], auditoria["id"])
            self.assertEqual(auditoria["fundamentos_faltantes"], [], auditoria["id"])
        for conversacion in self.conversaciones:
            pasos = como_traza(conversacion)["pasos"]
            recuperadas = {s for p in pasos if p["herramienta"] == "buscar_normativa"
                           for s in secciones_en(p["resultado"])}
            self.assertLessEqual(set(pasos[-1]["argumentos"].get("fundamentos", [])), recuperadas)
        insertadas = sum(len(a["busquedas_insertadas"]) for a in self.auditoria.values())
        self.assertGreater(insertadas, 0)
        self.assertEqual(insertadas, sum(e["curado"]["busquedas_insertadas"]
                                         for e in self.manifiesto["estadisticas"].values()))

    def test_no_comparte_casos_con_practica(self):
        practica = cargar_casos(RAIZ / "practica/casos.jsonl")
        triages = {"\n".join(nota_de_triage(c).splitlines()[1:]) for c in practica}
        for conversacion in self.conversaciones:
            nota = como_traza(conversacion)["pasos"][0]["resultado"]
            self.assertNotIn("\n".join(nota.splitlines()[1:]), triages)

    def test_rechaza_salida_existente_sin_tocarla(self):
        with tempfile.TemporaryDirectory() as carpeta:
            salida = Path(carpeta) / "dataset"
            salida.mkdir()
            (salida / "conversaciones.jsonl").write_text("no tocar", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generar_dataset(salida)
            comando = [sys.executable, "-m", "scripts.generar_conversaciones", "--salida", str(salida)]
            resultado = subprocess.run(comando, capture_output=True, text=True, cwd=RAIZ)
            self.assertEqual(resultado.returncode, 1)
            self.assertEqual(list(salida.iterdir()), [salida / "conversaciones.jsonl"])
            self.assertEqual((salida / "conversaciones.jsonl").read_text(encoding="utf-8"), "no tocar")


class CuradoTest(unittest.TestCase):
    """Búsqueda de ART agregada y fundamentos filtrados, con trazas y buscador reales."""

    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        trazas = leer_jsonl(RAIZ / "entrenamiento/trazas.jsonl")
        self.traza = next(t for t in trazas if any(
            p["argumentos"].get("consulta") == "accidente de trabajo ART dolor torácico plazo" for p in t["pasos"])
            and not any(p["argumentos"].get("consulta") == "accidente de trabajo ART" for p in t["pasos"]))
        valores = valores_de_traza(self.traza, unidades())
        ruta = escribir_normas(RAIZ / "normas/original", Path(self.temporal.name) / "n", valores)
        self.buscador = BuscadorBM25(ruta, variante="baseline")
        import yaml
        datos = yaml.safe_load((ruta / "parametros.yaml").read_text(encoding="utf-8"))
        self.rectoras = secciones_rectoras(fila_desde_traza(self.traza),
                                           {p["codigo"]: p["valor"] for p in datos["parametros"]})

    def test_agrega_busqueda_de_art_despues_de_la_combinada_sin_tocar_las_originales(self):
        self.assertIn("NOR-COB-02", self.rectoras)
        antes = json.loads(json.dumps(self.traza))
        curada, informe = curar_traza(self.traza, self.buscador, self.rectoras)
        self.assertEqual(self.traza, antes)
        consultas = [p["argumentos"].get("consulta") for p in curada["pasos"]]
        combinada = consultas.index("accidente de trabajo ART dolor torácico plazo")
        self.assertEqual(consultas[combinada + 1], "accidente de trabajo ART")
        self.assertEqual([c for i, c in enumerate(consultas) if i != combinada + 1],
                         [p["argumentos"].get("consulta") for p in antes["pasos"]])
        nuevo = curada["pasos"][combinada + 1]
        self.assertIn("NOR-COB-02", secciones_en(nuevo["resultado"]))
        self.assertEqual(nuevo["resultado"], observar_busqueda(self.buscador, "accidente de trabajo ART"))
        self.assertEqual([p["orden"] for p in curada["pasos"]], list(range(1, len(curada["pasos"]) + 1)))
        self.assertEqual(len(informe["busquedas_insertadas"]), 1)

    def test_sin_consulta_de_art_donde_insertar_se_excluye(self):
        sin_art = {**self.traza, "pasos": [p for p in self.traza["pasos"]
                                           if not p["argumentos"].get("consulta", "").startswith("accidente")]}
        with self.assertRaises(Excluida):
            curar_traza(sin_art, self.buscador, self.rectoras)

    def test_quita_fundamentos_no_recuperados_y_la_clave_si_queda_vacia(self):
        curada, informe = curar_traza(self.traza, self.buscador, self.rectoras)
        recuperadas = {s for p in curada["pasos"] if p["herramienta"] == "buscar_normativa"
                       for s in secciones_en(p["resultado"])}
        originales = self.traza["pasos"][-1]["argumentos"]["fundamentos"]
        final = curada["pasos"][-1]["argumentos"]["fundamentos"]
        self.assertEqual(final, [s for s in originales if s in recuperadas])
        self.assertEqual(informe["fundamentos_quitados"], [s for s in originales if s not in recuperadas])
        vacia = json.loads(json.dumps(self.traza))
        vacia["pasos"][-1]["argumentos"]["fundamentos"] = ["NOR-INT-01"]
        curada, informe = curar_traza(vacia, self.buscador, self.rectoras)
        self.assertNotIn("fundamentos", curada["pasos"][-1]["argumentos"])
        self.assertEqual(informe["fundamentos_quitados"], ["NOR-INT-01"])


class ReconstruccionTest(unittest.TestCase):
    def test_parametros_y_fila_de_una_traza(self):
        traza = leer_jsonl(RAIZ / "entrenamiento/trazas.jsonl")[0]
        self.assertEqual(parametros_observados(traza)["P-ADM-2"], "60 años")
        self.assertEqual(valores_de_traza(traza, unidades())["P-ADM-2"], 60)
        contradictoria = json.loads(json.dumps(traza))
        contradictoria["pasos"][2]["resultado"] += "\nValores: P-ADM-2 = 99 años."
        with self.assertRaises(Excluida) as error:
            valores_de_traza(contradictoria, unidades())
        self.assertEqual(error.exception.motivo, "parametros_contradictorios")

    def test_fila_reconstruida_coincide_con_el_csv_del_historial(self):
        with (RAIZ / "historial/historial.csv").open(encoding="utf-8", newline="") as archivo:
            filas = {f["id"]: f for f in csv.DictReader(archivo)}
        for traza in leer_jsonl(RAIZ / "historial/trazas.jsonl"):
            fila = fila_desde_traza(traza)
            esperado = {k: v for k, v in filas[traza["id"]].items() if k not in ("fecha", "operador")}
            self.assertEqual({k: v for k, v in fila.items() if k != "nombre"}, esperado, traza["id"])

    def test_parametro_relevante_no_observado_excluye_y_el_irrelevante_usa_el_original(self):
        trazas = leer_jsonl(RAIZ / "entrenamiento/trazas.jsonl")
        candidatos = {"P-TDE-1": [10, 15, 20, 25], "P-TDE-5": [2, 3, 4, 5, 6]}
        originales = {"P-ADM-1": 2, "P-ADM-2": 67, "P-ADM-3": 2, "P-COB-2": 7, "P-TDE-1": 10,
                      "P-TDE-2": 10, "P-TDE-3": 15, "P-TDE-4": 60, "P-TDE-5": 5}
        ecg = next(t for t in trazas if t["ficha"]["circuito"] == "ecg")
        sin_circuito = next(t for t in trazas if t["ficha"]["circuito"] == "ninguno")
        observados = valores_de_traza(ecg, unidades())
        observados.pop("P-TDE-1")
        with self.assertRaises(Excluida) as error:
            completar_parametros(fila_desde_traza(ecg), observados, originales, candidatos)
        self.assertIn("P-TDE-1", error.exception.motivo)
        observados = {c: v for c, v in valores_de_traza(sin_circuito, unidades()).items() if not c.startswith("P-TDE")}
        completos, supuestos = completar_parametros(fila_desde_traza(sin_circuito), observados, originales, candidatos)
        self.assertEqual(supuestos, ["P-TDE-1", "P-TDE-5"])
        self.assertEqual((completos["P-TDE-1"], completos["P-TDE-5"]), (10, 5))

    def test_detecta_caso_de_practica_aunque_cambie_el_id(self):
        huellas = huellas_practica(RAIZ / "practica/casos.jsonl")
        caso = cargar_casos(RAIZ / "practica/casos.jsonl")[0]
        traza = {"id": "X-1", "pasos": [
            {"herramienta": "leer_triage", "argumentos": {},
             "resultado": nota_de_triage(caso).replace(caso.paciente_id, "P-0000")},
            {"herramienta": "registrar_decision", "argumentos": {
                "registro": "provisorio", "numero_documento": None, "box_antes_del_registro": False,
                "circuito": "ninguno", "plazo_circuito_min": None, "cobertura": "sin_cobertura",
                "entidad_a_notificar": None, "aviso_al_voluntariado": False}, "resultado": ""}]}
        fila = fila_desde_traza(traza)
        with self.assertRaises(Excluida) as error:
            verificar_no_practica(traza["pasos"][0]["resultado"], fila, huellas)
        self.assertEqual(error.exception.motivo, "practica_triage_identico")
        otra = {**fila, "credenciales": ""}
        if caso.documento_numero:
            nota = traza["pasos"][0]["resultado"].replace(caso.nombre, "Otra Persona")
            with self.assertRaises(Excluida) as error:
                verificar_no_practica(nota, otra, huellas)
            self.assertEqual(error.exception.motivo, "practica_documento_o_afiliado_compartido")


if __name__ == "__main__":
    unittest.main()
