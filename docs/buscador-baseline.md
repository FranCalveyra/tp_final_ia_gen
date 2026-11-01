# Buscador inicial

`entrega.py` permite evaluar el buscador sin cargar un modelo. La implementación
del agente y su validación pendiente están en [modelo-modal.md](modelo-modal.md).

Se indexa una sección completa por archivo `NOR-*.md`, sin excluir otras
áreas. `admision.normas.cargar_normas` agrega los valores de `parametros.yaml`
de la carpeta recibida. El texto devuelto se conserva; solo el índice y la
consulta se normalizan a minúsculas, sin tildes y con términos alfanuméricos.

BM25 usa k1=1.5, b=0.75 e IDF positiva `log(1 + (N-df+0.5)/(df+0.5))`.
No usa embeddings, sinónimos ni servicios externos. Devuelve hasta dos
secciones con coincidencias; los empates conservan el orden de archivo.

## Reproducir sin construir ni instalar el proyecto

Desde la raíz del repositorio (requiere `uv`; PyYAML es dependencia del kit):

```bash
uv run --no-project --with pyyaml python -m unittest discover -s tests -v
uv run --no-project --with pyyaml python -m admision.evaluar_rag \
  --entrega entrega.py --normas normas/original \
  --consultas rag/consultas_dev.jsonl
```

Resultado inicial sobre las 60 consultas, con dos secciones:

| Métrica | Nuestro baseline | Referencia del enunciado |
|---|---:|---:|
| Recall de evidencia | 0,817 | 0,875 |
| Precisión | 0,467 | 0,492 |
| Tokens por consulta (estimados) | 618 | 580 |
| Consultas con toda la evidencia | 48/60 | 51/60 |

Es una primera medición, no una mejora sobre la referencia. Las consultas
coloquiales tienen recall 0,25: el buscador depende de coincidencias de palabras.
No se midió todavía resolución de pacientes ni tiempo de inferencia.
El entorno de ejecución definitivo y las versiones fijadas siguen pendientes.

## Candidato experimental: sin palabras comunes

El **default sigue siendo `baseline` (48/60)**. Para seleccionar el candidato:

```bash
BUSCADOR_VARIANTE=sin_comunes python -m admision.evaluar_rag \
  --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl
```

También se puede construir `BuscadorBM25(ruta, variante="sin_comunes")`.
Un nombre desconocido produce `ValueError`. La lista fija está en
`PALABRAS_COMUNES`, se aplica al índice y a las consultas y conserva `no`, `sin`,
`ni`, números y términos del dominio. No altera el texto devuelto.

### Auditoría de experimentos (60 consultas de desarrollo, k=2)

| Variante | Completas | Recall | Precisión | Tokens/consulta |
|---|---:|---:|---:|---:|
| Baseline | 48/60 | 0,817 | 0,467 | 618 |
| Sin comunes (candidato seleccionable) | 50/60 | 0,850 | 0,483 | 591 |
| Peso 0,25 para comunes (rechazado) | 49/60 | 0,833 | 0,475 | 605 |
| Fusión léxica (solo experimento) | 48/60 | 0,817 | 0,467 | 612 |

Sin comunes gana `d022,d026,d028` y pierde `d011`. No es una mejora sin
regresiones: se conserva como candidato por decisión explícita, sin reemplazar
el baseline. El ensayo de peso 0,25 preservó índice/longitudes/IDF originales,
pero perdió también `d011`, por eso no se incorporó.

La alternativa usa n-gramas de caracteres 3..5 dentro de cada palabra
alfanumérica, con espacios en los bordes y la misma normalización sin tildes.
TF es frecuencia cruda; IDF suavizada es `log((1+N)/(1+df))+1`; los vectores
se normalizan con L2 y se ordenan por coseno. Estos conceptos siguen la
[documentación de TF-IDF de scikit-learn](https://sklearn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
pero se implementan con biblioteca estándar, sin instalar scikit-learn.
No cruza bordes de palabras ni usa rasgos fuera del vocabulario del corpus.
Se fusiona el ranking completo positivo con BM25 sin comunes mediante
`RRF(d) = sum(1/(60+puesto(d)))`, pesos iguales y puestos desde 1;
los empates se resuelven por código de norma. Solo se devuelven dos normas.
Es **léxico, no semántico**; no entiende sinónimos.

Fusión gana `d022,d024,d032`, pierde `d011,d025,d027` frente al baseline.
Frente al candidato pierde `d026,d028` y gana `d024,d032`, además pierde
`d025,d027`: no lo supera y no se ofrece desde `entrega.py`.

### Reproducir baseline, candidato y alternativa

Con Python y PyYAML disponibles, sin construir ni instalar el proyecto:

```bash
python -m scripts.comparar_buscadores \
  --salida corridas/rag-variantes-nueva/comparacion.json
python -m unittest discover -s tests -v
```

La salida debe ser nueva: el script no pisa resultados. El JSON contiene todas
las consultas, normas traídas, evidencia recuperada/faltante, ganancias y pérdidas,
configuración exacta y hashes SHA256 de los datos. La corrida registrada está en
`corridas/rag-variantes-20261101/comparacion.json` (artefacto local regenerable).
No hubo barrido de parámetros ni reglas por consulta. Los resultados se usaron
para desarrollo: **no son evaluación independiente ni resolución de pacientes**.
