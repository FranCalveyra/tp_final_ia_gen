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

## Candidato `prefijos` (2026-11-10, issue #6)

Seleccionable con `BUSCADOR_VARIANTE=prefijos`; **el default sigue siendo `baseline`**
hasta que el equipo decida (propuesta abajo). Misma unidad (sección completa con su
línea `Valores:` de la carpeta recibida), mismo BM25 (k1=1,5, b=0,75) y texto devuelto sin
cambios. Cambian tres cosas, todas generales, sin reglas por consulta ni valores fijos:

1. **Sin palabras comunes**, la misma lista de `sin_comunes`.
2. **Stemming por prefijo de 4 letras** en índice y consulta (`registrarlo`, `registro`,
   `registra` → `regi`); los números no se truncan. Une flexiones del español sin diccionario.
3. **k adaptativo**: `k_entorno=3`, pero se descarta toda sección con puntaje menor que
   0,6 × el de la primera. Con consultas claras vuelve 1 sección; con dudosas, hasta 3.
   El umbral también rige en `recuperar(consulta, k)` con k grande (modo presupuesto).

Configuración en `admision_tp/buscador.py` (`VARIANTES`, `ConfigBM25`).

### Resultados (60 consultas de desarrollo)

Comandos (tests: 121, 120 OK y 1 omitido opcional):

```bash
BUSCADOR_VARIANTE=prefijos rtk proxy uv run --locked --no-sync python -m admision.evaluar_rag \
  --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl
rtk proxy uv run --locked --no-sync python -m scripts.comparar_buscadores \
  --salida corridas/rag-prefijos-20261110/original.json
rtk proxy uv run --locked --no-sync python -m scripts.comparar_buscadores --normas normas/b \
  --evidencia-ajustada --salida corridas/rag-prefijos-20261110/b-evidencia-ajustada.json
```

| Variante | k | Completas | Recall | Precisión | Tokens/consulta | Estado |
|---|---:|---:|---:|---:|---:|---|
| Referencia de la cátedra | 2 | 51/60 | 0,875 | 0,492 | 580 | — |
| `baseline` | 2 | 48/60 | 0,817 | 0,467 | 618 | Default |
| `sin_comunes` | 2 | 50/60 | 0,850 | 0,483 | 591 | Seleccionable |
| **`prefijos`** | 3, umbral 0,6 | **55/60** | **0,933** | **0,686** | **549** | Seleccionable, propuesto |
| `prefijos_k2` | 2, umbral 0,6 | 53/60 | 0,900 | 0,717 | **449** | Solo script |
| `prefijos_k2_sin_umbral` | 2 | 53/60 | 0,900 | 0,508 | 580 | Solo script |
| `prefijo5_k3` | 3, umbral 0,6 | 53/60 | 0,900 | 0,667 | 542 | Solo script |

Recall por tipo de `prefijos`: coloquial 0,75 (baseline 0,25), directa 1,0, distractora 1,0,
dos secciones 0,875, numérica 1,0. Trae 1,82 secciones por consulta en promedio.

**Por consulta.** Frente a `baseline` gana `d021,d022,d024,d026,d028,d031,d032,d034,d049`
y pierde `d027,d052`. Frente a `sin_comunes` gana `d011,d021,d024,d031,d032,d034,d049` y
pierde `d027,d052`. Las dos pérdidas son colisiones de prefijo, no reglas: en d027
`pasar`/`senora`/`hacemos` → `pasa`/`seno`/`hace` suben NOR-ADM-01/ADM-03 sobre NOR-COB-03;
en d052 `registro` → `regi` favorece NOR-ADM-02 y GUA-02 sobre NOR-ADM-01, y además NOR-ADM-01
queda bajo el umbral (6,9 contra 0,6 × 15,5), así que solo vuelve NOR-COB-01. Siguen sin resolverse
d023, d051, d030 y las dos anteriores. En d030 (pecho ↔ dolor torácico) no hay término compartido y el
agente recibe «No se encontró ninguna sección.», igual que con `sin_comunes`. Distribución: 25 consultas
reciben 1 sección, 16 reciben 3.

**Robustez b/c.** El texto de las normas b y c es idéntico; solo cambia `parametros.yaml`, así que
el ranking es el mismo. Con la evidencia del dataset tal cual, las 12 consultas numéricas
citan los valores originales y fallan para todas las variantes (`prefijos` 44/60, recall 0,750, contra
`baseline` 38/60, 0,650). Con `--evidencia-ajustada` (reemplaza `P-XXX-N = valor` por el valor
de la carpeta recibida) b y c dan exactamente lo mismo que original para todas las variantes:
los valores llegan correctos al texto. Artefactos locales en `corridas/rag-prefijos-20261110/`
(`original.json`, `{b,c}-evidencia-{ajustada,original}.json`).

**Modo presupuesto** (evaluar_rag, `prefijos` / `baseline`): 256 tokens 35/33 completas,
512 tokens 44/39, 1024 tokens 54/50 usando 536 contra 1020 tokens, porque el umbral corta antes.

### Exploración descartada

Todas las filas salen de `scripts/explorar_buscadores.py`, que registra la configuración de cada
intento, sus consultas completas, ganancias/pérdidas contra `baseline` y los SHA256 de normas y
consultas. Lo expresable con `ConfigBM25` se mide con `BuscadorBM25`; `BuscadorExploratorio`
(solo en el script) cubre párrafos, recorte, otros stemmers, peso del título y k1/b, y con sus
valores por defecto da las mismas métricas y consultas que `baseline` (test incluido).
Todas las filas con k=2 salvo indicación.

```bash
rtk proxy uv run --locked --no-sync python -m scripts.explorar_buscadores \
  --salida corridas/rag-exploracion-20261110/original.json
```

La salida debe ser nueva (el script no pisa archivos). La corrida registrada reprodujo
exactamente todas las filas de esta tabla, que se habían medido antes con un script temporal.

| Intento | Completas | Recall | Tokens | Por qué se descartó |
|---|---:|---:|---:|---|
| Párrafos `§` + encabezado + sus valores, top 2–5 párrafos | 39–46 | 0,683–0,792 | 176–389 | Pierde contexto de sección; coloquiales ≤0,17 |
| Ídem con stop + prefijo 4, top 4–5 | 49 | 0,833 | 291–341 | Menos tokens pero menor recall que la referencia |
| Párrafo + puntaje de sección (α 0,5–1) | 44–49 | 0,767–0,825 | 210–347 | Sin mejora |
| Sección recortada a párrafos con coincidencia (prefijo 4) | 51 | 0,867 | 535 | Pierde evidencia de párrafos sin términos de la consulta |
| Ídem, recorte ≥0,5 × mejor párrafo | 49 | 0,825 | 353 | Idem, más pérdida |
| Plurales (`-s`, `-es`) + stop | 49 | 0,833 | 589 | Peor que prefijo |
| Sufijos españoles livianos + stop | 51 | 0,867 | 586 | Peor que prefijo 4 |
| Prefijo 5 / 6 + stop | 51 / 51 | 0,867 | 583 / 586 | Peor que prefijo 4 |
| Título ×2 / ×3 (stop) | 49 / 49 | 0,842 | 585 / 590 | Peor que `sin_comunes` en dos secciones |
| BM25 k1=1,2 · b=0,5 · b=0,9 · k1=0,9 b=0,4 (stop) | 50–51 | 0,850–0,867 | 584–610 | Ganancia ≤1 consulta, sin barrido fino |
| `prefijos` con k=3 sin umbral | 56 | 0,942 | 895 | +45 % de tokens por una consulta |
| `prefijos` umbral 0,4 / 0,5 / 0,7 / 0,8 (k=3) | 56/55/53/52 | 0,942–0,892 | 754/671/471/423 | 0,6 es el codo; 0,5–0,6 estables |
| `prefijos` k=4, umbral 0,6 | 55 | 0,933 | 608 | Igual a k=3 con más tokens |

**Sobreajuste.** Prefijo (4), umbral (0,6) y k (3) se eligieron mirando estas 60 consultas;
los resultados son de desarrollo, no evaluación independiente. Los vecinos (umbral 0,5–0,6,
prefijo 5) dan 53–55/60, sin un pico aislado. Falta ver el efecto en el agente: con
`prefijos` cambia la cantidad de secciones que recibe por búsqueda.

### Fuera de alcance

Lo que queda (d030 pecho ↔ torácico, d023 abuela ↔ edad) necesita sinónimos o
embeddings densos: requieren dependencias o descargas nuevas no autorizadas.
