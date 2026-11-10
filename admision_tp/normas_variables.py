"""Carpetas de normas con otros valores de parámetros: mismos textos, otro `parametros.yaml`.

Sirven para re-ejecutar búsquedas con los valores que vio cada traza y para generar casos
sintéticos con valores variados. Los textos se copian byte a byte de la carpeta base; solo
cambian los valores pedidos. Nunca se fija un valor en código: llegan como datos.
"""

import re
import shutil
from pathlib import Path

import yaml

from admision.normas import cargar_parametros, valor_como_texto


def parametros_base(ruta_normas: Path) -> list[dict]:
    """Entradas de `parametros.yaml` ({codigo, valor, unidad}) en su orden original."""
    datos = yaml.safe_load((Path(ruta_normas) / "parametros.yaml").read_text(encoding="utf-8"))
    return [dict(p) for p in datos["parametros"]]


def valor_desde_texto(codigo: str, texto: str, unidades: dict[str, str]) -> int:
    """Entero de «nivel 2», «60 años», «15 min» o «7»; exige que el texto sea el que renderiza el kit."""
    numeros = re.findall(r"\d+", texto)
    if codigo not in unidades or len(numeros) != 1:
        raise ValueError(f"valor no interpretable para {codigo}")
    valor = int(numeros[0])
    if valor_como_texto(valor, unidades[codigo]) != texto:
        raise ValueError(f"valor de {codigo} con formato distinto del de las normas")
    return valor


def escribir_normas(base: Path, destino: Path, valores: dict[str, int]) -> Path:
    """Copia las secciones de `base` a `destino` (nueva) con `valores` aplicados a sus parámetros."""
    base, destino = Path(base), Path(destino)
    entradas = parametros_base(base)
    desconocidos = set(valores) - {p["codigo"] for p in entradas}
    if desconocidos:
        raise ValueError(f"parámetros inexistentes: {sorted(desconocidos)}")
    destino.mkdir(parents=True)
    for archivo in sorted(base.glob("NOR-*.md")):
        shutil.copyfile(archivo, destino / archivo.name)
    for entrada in entradas:
        if entrada["codigo"] in valores:
            valor = valores[entrada["codigo"]]
            if type(valor) is not int or valor < 0:
                raise ValueError("los valores de parámetros son enteros no negativos")
            entrada["valor"] = valor
    (destino / "parametros.yaml").write_text(
        yaml.safe_dump({"parametros": entradas}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cargar_parametros(destino)  # falla aquí si el archivo escrito no es legible por el kit
    return destino


class CarpetasNormas:
    """Una carpeta por conjunto de valores, creada una sola vez bajo `raiz`, con su buscador.

    `fabrica(ruta)` arma el buscador igual que la entrega (por ejemplo `entrega.construir_buscador`).
    """

    def __init__(self, base: Path, raiz: Path, fabrica):
        self.base, self.raiz, self.fabrica = Path(base), Path(raiz), fabrica
        self.originales = {p["codigo"]: p["valor"] for p in parametros_base(self.base)}
        self._carpetas: dict[tuple, tuple[Path, object]] = {}

    def completar(self, valores: dict[str, int]) -> dict[str, int]:
        """Valores originales con `valores` aplicados encima."""
        return {**self.originales, **valores}

    def obtener(self, valores: dict[str, int]):
        """(ruta, buscador) de la carpeta con esos valores sobre los originales."""
        completos = self.completar(valores)
        clave = tuple(sorted(completos.items()))
        if clave not in self._carpetas:
            ruta = escribir_normas(self.base, self.raiz / f"normas-{len(self._carpetas):05d}", completos)
            self._carpetas[clave] = (ruta, self.fabrica(ruta))
        return self._carpetas[clave]

    def __len__(self):
        return len(self._carpetas)
