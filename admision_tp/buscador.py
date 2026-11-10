"""BM25 sobre secciones completas, incluidos sus parámetros vigentes."""

from dataclasses import dataclass
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from admision.normas import Seccion, cargar_normas

# Lista fija del experimento; no/sin/ni se conservan deliberadamente.
PALABRAS_COMUNES = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquellos
aqui asi aun aunque bajo cada como con contra cual cuales cuando de del desde donde
durante e el ella ellas ello ellos en entre era eran es esa esas ese esos esta estaba
estaban estas este esto estos fue fueron ha habia han hasta hay la las le les lo los
mas me mi mis muy o os otra otras otro otros para pero por porque que quien quienes
se ser si sido sobre son su sus tal tambien te tener tiene tienen todo todos tras tu
tus un una unas uno unos usted ustedes ya yo
""".split())


def _tokens(texto: str) -> list[str]:
    normalizado = unicodedata.normalize("NFD", texto.casefold())
    sin_tildes = "".join(c for c in normalizado if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", sin_tildes)


@dataclass(frozen=True)
class ConfigBM25:
    """Cómo se indexa y cuántas secciones ve el agente.

    prefijo: trunca cada término no numérico a sus primeras letras (stemming por prefijo).
    umbral_relativo: descarta secciones con puntaje menor que esa fracción del primero.
    """

    sin_comunes: bool = False
    prefijo: int | None = None
    umbral_relativo: float = 0.0
    k_entorno: int = 2


VARIANTES = {
    "baseline": ConfigBM25(),
    "sin_comunes": ConfigBM25(sin_comunes=True),
    "prefijos": ConfigBM25(sin_comunes=True, prefijo=4, umbral_relativo=0.6, k_entorno=3),
}


class BuscadorBM25:
    """Una sección por resultado; empates en orden de nombre de archivo.

    Consultas sin términos, sin coincidencias o con k <= 0 devuelven [].
    No se altera el texto original al normalizar el índice.
    """

    def __init__(self, ruta_normas: Path, variante: str | ConfigBM25 = "baseline"):
        if isinstance(variante, ConfigBM25):
            self.variante, self.config = "personalizada", variante
        elif variante in VARIANTES:
            self.variante, self.config = variante, VARIANTES[variante]
        else:
            raise ValueError(f"BUSCADOR_VARIANTE debe ser una de: {', '.join(VARIANTES)}")
        self.k_entorno = self.config.k_entorno
        self._secciones = cargar_normas(Path(ruta_normas))
        self._frecuencias = [Counter(self._terminos(s.texto)) for s in self._secciones]
        self._longitudes = [sum(f.values()) for f in self._frecuencias]
        self._promedio = sum(self._longitudes) / len(self._longitudes) if self._longitudes else 0
        documentos = Counter(t for f in self._frecuencias for t in f)
        n = len(self._secciones)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in documentos.items()}

    def _terminos(self, texto: str) -> list[str]:
        c = self.config
        terminos = [t for t in _tokens(texto) if not (c.sin_comunes and t in PALABRAS_COMUNES)]
        if c.prefijo:
            terminos = [t if t.isdigit() else t[:c.prefijo] for t in terminos]
        return terminos

    def recuperar(self, consulta: str, k: int) -> list[Seccion]:
        """Devuelve hasta k secciones por relevancia BM25 (k1=1.5, b=0.75)."""
        if k <= 0 or not self._promedio:
            return []
        terminos = sorted(set(self._terminos(consulta)))
        puntuados = []
        for i, frecuencias in enumerate(self._frecuencias):
            normalizador = 1.5 * (0.25 + 0.75 * self._longitudes[i] / self._promedio)
            puntaje = sum(
                self._idf[t] * frecuencias[t] * 2.5 / (frecuencias[t] + normalizador)
                for t in terminos if t in frecuencias
            )
            if puntaje > 0:
                puntuados.append((i, puntaje))
        puntuados.sort(key=lambda x: (-x[1], x[0]))
        if puntuados:
            minimo = self.config.umbral_relativo * puntuados[0][1]
            puntuados = [(i, p) for i, p in puntuados if p >= minimo]
        return [self._secciones[i] for i, _ in puntuados[:k]]
