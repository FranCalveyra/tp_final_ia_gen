"""Experimento léxico CPU; ejecutar como python -m scripts.comparar_buscadores."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from admision.evaluar_rag import cargar_dataset, evaluar_buscador
from admision.normas import cargar_parametros
from admision_tp.buscador import BuscadorBM25, ConfigBM25, PALABRAS_COMUNES, VARIANTES, _tokens

# Ensayos medidos y documentados, no seleccionables desde entrega.py.
EXPLORATORIAS = {
    "prefijos_k2_sin_umbral": ConfigBM25(sin_comunes=True, prefijo=4),
    "prefijos_k2": ConfigBM25(sin_comunes=True, prefijo=4, umbral_relativo=0.6),
    "prefijo5_k3": ConfigBM25(sin_comunes=True, prefijo=5, umbral_relativo=0.6, k_entorno=3),
}
VALOR_CITADO = re.compile(r"^(P-[A-Z]{3}-\d+) = .+$")


def ngramas(texto):
    """Caracteres 3..5 dentro de cada palabra alfanumérica, con espacios de borde."""
    cuentas = Counter()
    for palabra in _tokens(texto):
        palabra = f" {palabra} "
        for n in range(3, 6):
            # Las palabras más cortas que n aportan la palabra acolchada una vez.
            if len(palabra) < n:
                continue
            cuentas.update(palabra[i:i + n] for i in range(len(palabra) - n + 1))
    return cuentas


class FusionLexica:
    """Alternativa experimental, no seleccionable desde entrega.py. RRF igual, c=60."""

    k_entorno = 2

    def __init__(self, ruta):
        self.bm25 = BuscadorBM25(ruta, variante="sin_comunes")
        self.secciones = self.bm25._secciones
        cuentas = [ngramas(s.texto) for s in self.secciones]
        df = Counter(g for c in cuentas for g in c)
        self.idf = {g: math.log((1 + len(cuentas)) / (1 + f)) + 1 for g, f in df.items()}
        self.vectores = [self.vector(c) for c in cuentas]

    def vector(self, cuentas):
        pesos = {g: f * self.idf[g] for g, f in cuentas.items() if g in self.idf}
        norma = math.sqrt(sum(p * p for p in pesos.values()))
        return {g: p / norma for g, p in pesos.items()} if norma else {}

    def recuperar(self, consulta, k):
        if k <= 0:
            return []
        q = self.vector(ngramas(consulta))
        cosenos = [(s.id, sum(v.get(g, 0) * p for g, p in q.items()))
                   for s, v in zip(self.secciones, self.vectores)]
        cosenos = sorted(((i, p) for i, p in cosenos if p > 0), key=lambda x: (-x[1], x[0]))
        rankings = [[s.id for s in self.bm25.recuperar(consulta, len(self.secciones))],
                    [i for i, _ in cosenos]]
        rrf = Counter()
        for ranking in rankings:
            for puesto, codigo in enumerate(ranking, 1):
                rrf[codigo] += 1 / (60 + puesto)
        orden = sorted(rrf, key=lambda i: (-rrf[i], i))[:k]
        por_id = {s.id: s for s in self.secciones}
        return [por_id[i] for i in orden]


def ajustar_evidencia(dataset, ruta_normas):
    """Reescribe la evidencia «P-XXX-N = valor» con el valor de la carpeta recibida.

    En las normas originales no cambia nada; en b y c permite medir las consultas
    numéricas con los valores que el agente vería en esa versión.
    """
    valores = cargar_parametros(ruta_normas)
    def ajustar(e):
        m = VALOR_CITADO.match(e["texto"])
        return {**e, "texto": f"{m[1]} = {valores[m[1]]}"} if m and m[1] in valores else e
    return [{**item, "evidencia": [ajustar(e) for e in item["evidencia"]]} for item in dataset]


def diagnosticar(buscador, dataset):
    normalizar = lambda t: re.sub(r"\s+", " ", t).strip()
    resultados = []
    for item in dataset:
        secciones = buscador.recuperar(item["consulta"], getattr(buscador, "k_entorno", 2))
        textos = [normalizar(s.texto) for s in secciones]
        evidencia = [{**e, "recuperada": any(normalizar(e["texto"]) in t for t in textos)}
                     for e in item["evidencia"]]
        resultados.append({"id": item["id"], "consulta": item["consulta"],
                           "traidas": [s.id for s in secciones], "evidencia": evidencia,
                           "completa": all(e["recuperada"] for e in evidencia)})
    return {"metricas": evaluar_buscador(buscador, dataset), "consultas": resultados}


def comparar(ruta_normas, ruta_consultas, evidencia_ajustada=False):
    dataset = cargar_dataset(ruta_consultas)
    if evidencia_ajustada:
        dataset = ajustar_evidencia(dataset, ruta_normas)
    buscadores = {nombre: BuscadorBM25(ruta_normas, nombre) for nombre in VARIANTES}
    buscadores["fusion_lexica"] = FusionLexica(ruta_normas)
    buscadores.update({n: BuscadorBM25(ruta_normas, c) for n, c in EXPLORATORIAS.items()})
    resultados = {nombre: diagnosticar(b, dataset) for nombre, b in buscadores.items()}
    completos = {n: {x["id"] for x in r["consultas"] if x["completa"]} for n, r in resultados.items()}
    for nombre, resultado in resultados.items():
        for referencia in ("baseline", "sin_comunes"):
            resultado[f"ganadas_vs_{referencia}"] = sorted(completos[nombre] - completos[referencia])
            resultado[f"perdidas_vs_{referencia}"] = sorted(completos[referencia] - completos[nombre])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(ruta_normas).iterdir()) if p.is_file()}
    variantes = {n: vars(c) for n, c in {**VARIANTES, **EXPLORATORIAS}.items()}
    return {"configuracion": {"variantes_bm25": variantes, "evidencia_ajustada": evidencia_ajustada,
             "fusion_k": 2, "bm25_k1": 1.5, "bm25_b": 0.75,
             "palabras_comunes": sorted(PALABRAS_COMUNES), "ngramas": [3, 5],
             "idf": "log((1+N)/(1+df))+1", "tf": "frecuencia cruda", "norma": "L2",
             "rrf_constante": 60, "rrf_pesos": [1, 1], "normas_sha256": hashes,
             "consultas_sha256": hashlib.sha256(Path(ruta_consultas).read_bytes()).hexdigest()},
            "resultados": resultados}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normas", type=Path, default=Path("normas/original"))
    parser.add_argument("--consultas", type=Path, default=Path("rag/consultas_dev.jsonl"))
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--evidencia-ajustada", action="store_true",
                        help="usa los valores de --normas en la evidencia numérica (para b y c)")
    args = parser.parse_args()
    informe = comparar(args.normas, args.consultas, args.evidencia_ajustada)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    print(json.dumps({n: {"metricas": r["metricas"], "ganadas": r["ganadas_vs_baseline"],
                         "perdidas": r["perdidas_vs_baseline"],
                         "ganadas_vs_sin_comunes": r["ganadas_vs_sin_comunes"],
                         "perdidas_vs_sin_comunes": r["perdidas_vs_sin_comunes"]}
                      for n, r in informe["resultados"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
