"""Exploración léxica CPU del issue #6, incluidos los intentos rechazados.

Ejecutar como python -m scripts.explorar_buscadores --salida corridas/<nueva>/exploracion.json.
Lo que `ConfigBM25` expresa (palabras comunes, prefijo, umbral relativo, k) se mide con
`BuscadorBM25`. `BuscadorExploratorio` cubre solo lo que el buscador no ofrece: párrafos como
unidad, recorte de párrafos, otros stemmers, peso del título y k1/b. Con sus valores por
defecto reproduce `baseline` (lo verifica tests/test_explorar_buscadores.py).
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from admision.evaluar_rag import _normalizar, cargar_dataset, evaluar_buscador
from admision.normas import cargar_parametros, linea_de_valores
from admision_tp.buscador import PALABRAS_COMUNES, BuscadorBM25, ConfigBM25, _tokens

SUFIJOS = sorted("amientos imientos amiento imiento aciones iciones acion icion mente ando iendo "
                 "ados idos adas idas ado ido ada ida ar er ir es as os s a o e".split(),
                 key=len, reverse=True)


def raiz(termino: str, stemmer: str) -> str:
    """none, plural, sufijos o prefijoN; los números no cambian."""
    if stemmer == "none" or termino.isdigit():
        return termino
    if stemmer == "plural":
        if len(termino) > 4 and termino.endswith("es"):
            return termino[:-2]
        return termino[:-1] if len(termino) > 3 and termino.endswith("s") else termino
    if stemmer == "sufijos":
        for sufijo in SUFIJOS:
            if termino.endswith(sufijo) and len(termino) - len(sufijo) >= 4:
                return termino[:-len(sufijo)]
        return termino
    if stemmer.startswith("prefijo"):
        return termino[:int(stemmer[len("prefijo"):])]
    raise ValueError(f"stemmer desconocido: {stemmer}")


@dataclass(frozen=True)
class Fragmento:
    id: str
    texto: str


class _IndiceBM25:
    def __init__(self, documentos, k1, b):
        self.k1, self.b = k1, b
        self.frecuencias = [Counter(d) for d in documentos]
        self.longitudes = [sum(f.values()) for f in self.frecuencias]
        self.promedio = sum(self.longitudes) / len(self.longitudes)
        df = Counter(t for f in self.frecuencias for t in f)
        n = len(documentos)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def puntajes(self, terminos):
        resultado = []
        for f, largo in zip(self.frecuencias, self.longitudes):
            norma = self.k1 * (1 - self.b + self.b * largo / self.promedio)
            resultado.append(sum(self.idf[t] * f[t] * (self.k1 + 1) / (f[t] + norma)
                                 for t in terminos if t in f))
        return resultado


class BuscadorExploratorio:
    """Solo experimento. unidad: seccion o parrafo; el texto devuelto es literal de la norma
    (encabezado + párrafos elegidos en su orden) más la línea de valores de lo que cita."""

    def __init__(self, ruta, unidad="seccion", stemmer="none", sin_comunes=False, titulo=1,
                 k1=1.5, b=0.75, k_entorno=2, umbral=0.0, n_parrafos=None, alfa=0.0, recorte=None):
        self.stemmer, self.sin_comunes, self.unidad = stemmer, sin_comunes, unidad
        self.k_entorno, self.umbral, self.n_parrafos = k_entorno, umbral, n_parrafos
        self.alfa, self.recorte = alfa, recorte
        self.valores = cargar_parametros(Path(ruta))
        self.secciones = []
        for archivo in sorted(Path(ruta).glob("NOR-*.md")):
            bloques = archivo.read_text(encoding="utf-8").strip().split("\n\n")
            self.secciones.append((archivo.stem, bloques[0], bloques[1:]))
        self.unidades, documentos = [], []
        for si, (_, encabezado, parrafos) in enumerate(self.secciones):
            if unidad == "seccion":
                self.unidades.append((si, None))
                cuerpo = "\n\n".join(parrafos)
                documentos.append(self.terminos(encabezado) * titulo + self.terminos(
                    cuerpo + "\n" + linea_de_valores("\n\n".join([encabezado] + parrafos), self.valores)))
            else:
                for pi, parrafo in enumerate(parrafos):
                    self.unidades.append((si, pi))
                    documentos.append(self.terminos(encabezado) * titulo + self._parrafo(parrafo))
        self.indice = _IndiceBM25(documentos, k1, b)
        if recorte is not None:
            self.por_seccion = [len(p) for _, _, p in self.secciones]
            self.indice_parrafos = _IndiceBM25(
                [self._parrafo(p) for _, _, ps in self.secciones for p in ps], k1, b)
        if alfa:
            self.indice_secciones = _IndiceBM25(
                [self.terminos("\n\n".join([e] + ps)) for _, e, ps in self.secciones], k1, b)

    def terminos(self, texto):
        return [raiz(t, self.stemmer) for t in _tokens(texto)
                if not (self.sin_comunes and t in PALABRAS_COMUNES)]

    def _parrafo(self, parrafo):
        return self.terminos(parrafo + "\n" + linea_de_valores(parrafo, self.valores))

    def _fragmento(self, si, elegidos=None):
        codigo, encabezado, parrafos = self.secciones[si]
        elegidos = range(len(parrafos)) if elegidos is None else sorted(elegidos)
        cuerpo = "\n\n".join([encabezado] + [parrafos[i] for i in elegidos])
        valores = linea_de_valores(cuerpo, self.valores)
        return Fragmento(codigo, f"{cuerpo}\n{valores}" if valores else cuerpo)

    def recuperar(self, consulta, k):
        if k <= 0:
            return []
        terminos = sorted(set(self.terminos(consulta)))
        puntajes = self.indice.puntajes(terminos)
        if self.alfa:
            de_seccion = self.indice_secciones.puntajes(terminos)
            escala = (max(puntajes) or 1) / (max(de_seccion) or 1)
            puntajes = [p + self.alfa * de_seccion[self.unidades[i][0]] * escala
                        for i, p in enumerate(puntajes)]
        orden = sorted(((i, p) for i, p in enumerate(puntajes) if p > 0), key=lambda x: (-x[1], x[0]))
        if self.umbral and orden:
            orden = [(i, p) for i, p in orden if p >= self.umbral * orden[0][1]]
        if self.unidad == "seccion":
            return [self._recortada(self.unidades[i][0], terminos) for i, _ in orden[:k]]
        grupos, tomados = {}, 0
        for i, _ in orden:
            if tomados >= self.n_parrafos:
                break
            si, pi = self.unidades[i]
            if si not in grupos:
                if len(grupos) >= k:
                    continue
                grupos[si] = []
            grupos[si].append(pi)
            tomados += 1
        return [self._fragmento(si, pis) for si, pis in grupos.items()]

    def _recortada(self, si, terminos):
        """Sin recorte, la sección entera; con recorte, párrafos con puntaje >= recorte × el mejor
        (recorte 0: cualquier coincidencia). Si ninguno coincide, la sección entera."""
        if self.recorte is None:
            return self._fragmento(si)
        inicio = sum(self.por_seccion[:si])
        locales = self.indice_parrafos.puntajes(terminos)[inicio:inicio + self.por_seccion[si]]
        mejor = max(locales)
        elegidos = [j for j, p in enumerate(locales) if mejor > 0 and p > 0 and p >= self.recorte * mejor]
        return self._fragmento(si, elegidos or None)


def _cfg(**kw):
    return ("config", kw)


def _exp(**kw):
    return ("exploratorio", kw)


INTENTOS = {
    "baseline": _cfg(),
    "baseline_exploratorio": _exp(),
    "sin_comunes": _cfg(sin_comunes=True),
    **{f"parrafos_top{n}": _exp(unidad="parrafo", n_parrafos=n) for n in (2, 3, 4, 5)},
    **{f"parrafos_top{n}_comunes": _exp(unidad="parrafo", n_parrafos=n, sin_comunes=True) for n in (3, 4, 5, 6)},
    **{f"parrafos_top{n}_prefijo4": _exp(unidad="parrafo", n_parrafos=n, sin_comunes=True,
                                         stemmer="prefijo4") for n in (4, 5)},
    **{f"parrafos_top{n}_prefijo4_alfa{a}": _exp(unidad="parrafo", n_parrafos=n, sin_comunes=True,
                                                 stemmer="prefijo4", alfa=a)
       for n, a in ((4, 0.5), (4, 1.0), (5, 1.0), (3, 1.0))},
    **{f"recorte{r}_prefijo4": _exp(sin_comunes=True, stemmer="prefijo4", recorte=r) for r in (0, 0.25, 0.5)},
    **{f"recorte{r}_prefijo5": _exp(sin_comunes=True, stemmer="prefijo5", recorte=r) for r in (0, 0.25)},
    "recorte0_comunes": _exp(sin_comunes=True, recorte=0),
    "recorte0": _exp(recorte=0),
    "recorte0.5_prefijo4_k3": _exp(sin_comunes=True, stemmer="prefijo4", recorte=0.5, k_entorno=3),
    "recorte0.25_prefijo4_k3": _exp(sin_comunes=True, stemmer="prefijo4", recorte=0.25, k_entorno=3),
    "plural": _exp(stemmer="plural"),
    "plural_comunes": _exp(stemmer="plural", sin_comunes=True),
    "sufijos_comunes": _exp(stemmer="sufijos", sin_comunes=True),
    "sufijos_comunes_umbral0.6": _exp(stemmer="sufijos", sin_comunes=True, umbral=0.6),
    "sufijos_comunes_umbral0.6_k3": _exp(stemmer="sufijos", sin_comunes=True, umbral=0.6, k_entorno=3),
    **{f"prefijo{p}": _cfg(prefijo=p) for p in (5,)},
    **{f"prefijo{p}_comunes": _cfg(sin_comunes=True, prefijo=p) for p in (4, 5, 6)},
    **{f"titulo{t}_comunes": _exp(sin_comunes=True, titulo=t) for t in (2, 3)},
    "titulo2_comunes_prefijo5": _exp(sin_comunes=True, titulo=2, stemmer="prefijo5"),
    "k1_1.2_comunes": _exp(sin_comunes=True, k1=1.2),
    "b0.5_comunes": _exp(sin_comunes=True, b=0.5),
    "b0.9_comunes": _exp(sin_comunes=True, b=0.9),
    "k1_0.9_b0.4_comunes": _exp(sin_comunes=True, k1=0.9, b=0.4),
    "prefijo4_k1": _cfg(sin_comunes=True, prefijo=4, k_entorno=1),
    "prefijo4_k3_sin_umbral": _cfg(sin_comunes=True, prefijo=4, k_entorno=3),
    **{f"prefijo4_k{k}_umbral{u}": _cfg(sin_comunes=True, prefijo=4, umbral_relativo=u, k_entorno=k)
       for k in (2, 3, 4) for u in (0.4, 0.5, 0.6, 0.7, 0.8)},
    **{f"{nombre}_k3_umbral{u}": _cfg(**cfg, umbral_relativo=u, k_entorno=3)
       for u in (0.5, 0.6, 0.7)
       for nombre, cfg in (("baseline", {}), ("comunes", {"sin_comunes": True}),
                           ("prefijo5", {"sin_comunes": True, "prefijo": 5}))},
}


def construir(ruta_normas, intento):
    tipo, parametros = intento
    if tipo == "config":
        return BuscadorBM25(ruta_normas, ConfigBM25(**parametros))
    return BuscadorExploratorio(ruta_normas, **parametros)


def completas(buscador, dataset):
    ids = []
    for item in dataset:
        textos = [_normalizar(f.texto) for f in buscador.recuperar(item["consulta"], buscador.k_entorno)]
        if all(any(_normalizar(e["texto"]) in t for t in textos) for e in item["evidencia"]):
            ids.append(item["id"])
    return ids


def explorar(ruta_normas, ruta_consultas, nombres=None):
    dataset = cargar_dataset(ruta_consultas)
    resultados = {}
    for nombre in nombres or INTENTOS:
        buscador = construir(ruta_normas, INTENTOS[nombre])
        tipo, parametros = INTENTOS[nombre]
        resultados[nombre] = {"tipo": tipo, "parametros": parametros,
                              "metricas": evaluar_buscador(buscador, dataset),
                              "completas": completas(buscador, dataset)}
    if "baseline" in resultados:
        base = set(resultados["baseline"]["completas"])
        for r in resultados.values():
            r["ganadas_vs_baseline"] = sorted(set(r["completas"]) - base)
            r["perdidas_vs_baseline"] = sorted(base - set(r["completas"]))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(ruta_normas).iterdir()) if p.is_file()}
    return {"configuracion": {"normas": str(ruta_normas), "normas_sha256": hashes,
                              "consultas_sha256": hashlib.sha256(Path(ruta_consultas).read_bytes()).hexdigest(),
                              "palabras_comunes": sorted(PALABRAS_COMUNES), "sufijos": SUFIJOS},
            "resultados": resultados}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normas", type=Path, default=Path("normas/original"))
    parser.add_argument("--consultas", type=Path, default=Path("rag/consultas_dev.jsonl"))
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    informe = explorar(args.normas, args.consultas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    for nombre, r in informe["resultados"].items():
        m = r["metricas"]
        print(f"{nombre:34s} k={m['k_entorno']} {m['consultas_con_toda_la_evidencia']:2d}/60 "
              f"recall={m['recall_evidencia']:.3f} precision={m['precision']:.3f} "
              f"tokens={m['tokens_por_consulta']}")


if __name__ == "__main__":
    main()
