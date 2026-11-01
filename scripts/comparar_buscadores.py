"""Experimento léxico CPU; ejecutar como python -m scripts.comparar_buscadores."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from admision.evaluar_rag import cargar_dataset, evaluar_buscador
from admision_tp.buscador import BuscadorBM25, PALABRAS_COMUNES, _tokens


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


def diagnosticar(buscador, dataset):
    normalizar = lambda t: re.sub(r"\s+", " ", t).strip()
    resultados = []
    for item in dataset:
        secciones = buscador.recuperar(item["consulta"], 2)
        textos = [normalizar(s.texto) for s in secciones]
        evidencia = [{**e, "recuperada": any(normalizar(e["texto"]) in t for t in textos)}
                     for e in item["evidencia"]]
        resultados.append({"id": item["id"], "consulta": item["consulta"],
                           "traidas": [s.id for s in secciones], "evidencia": evidencia,
                           "completa": all(e["recuperada"] for e in evidencia)})
    return {"metricas": evaluar_buscador(buscador, dataset), "consultas": resultados}


def comparar(ruta_normas, ruta_consultas):
    dataset = cargar_dataset(ruta_consultas)
    buscadores = {"baseline": BuscadorBM25(ruta_normas),
                  "sin_comunes": BuscadorBM25(ruta_normas, "sin_comunes"),
                  "fusion_lexica": FusionLexica(ruta_normas)}
    resultados = {nombre: diagnosticar(b, dataset) for nombre, b in buscadores.items()}
    base = {x["id"] for x in resultados["baseline"]["consultas"] if x["completa"]}
    for resultado in resultados.values():
        completos = {x["id"] for x in resultado["consultas"] if x["completa"]}
        resultado["ganadas_vs_baseline"] = sorted(completos - base)
        resultado["perdidas_vs_baseline"] = sorted(base - completos)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(ruta_normas).iterdir()) if p.is_file()}
    return {"configuracion": {"k": 2, "bm25_k1": 1.5, "bm25_b": 0.75,
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
    args = parser.parse_args()
    informe = comparar(args.normas, args.consultas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    print(json.dumps({n: {"metricas": r["metricas"], "ganadas": r["ganadas_vs_baseline"],
                         "perdidas": r["perdidas_vs_baseline"]}
                      for n, r in informe["resultados"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
