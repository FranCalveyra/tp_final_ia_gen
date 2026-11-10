# Informe: costumbres del hospital

## Resumen ejecutivo

El análisis usa `historial/historial.csv`, las 2.000 trazas correspondientes de `historial/trazas.jsonl` y las normas de `normas/original/`. Se confirmaron dos costumbres completas (aviso al voluntariado y elección de credencial con obra social y prepaga). Todas las filas `ruido` son desvíos de esas costumbres y provienen de tres operadores (`OP-04`, `OP-11`, `OP-17`).

Clasificación vigente (corrección del 2026-11-10, ver [bitácora](docs/progreso.md)):

| tipo | cantidad | antes de la corrección |
|---|---:|---:|
| correcta | 1.600 | 1.631 |
| error | 240 | 279 |
| ruido | 160 | 90 |
| total | 2.000 | 2.000 |

Una violación normativa siempre tiene prioridad sobre una costumbre: `error > ruido`.

## Fuentes y validación de trazas

El script usa tres fuentes:

- `historial/historial.csv`: atributos de ingreso, cobertura y ficha final.
- `historial/trazas.jsonl`: conversación completa, operador y ficha final de cada ID.
- `normas/original/`: secciones normativas y `parametros.yaml`.

Las trazas se validan antes de usarse como evidencia. Hay 2.000 IDs únicos, cada operador coincide con el CSV y las 2.000 fichas finales coinciden campo por campo con la fila correspondiente. La instrumentación contiene 2.000 lecturas de triage, 2.000 decisiones, 1.587 verificaciones de afiliación, 1.405 preguntas y 6.197 búsquedas normativas; el promedio es 6,5945 pasos por traza.

La coincidencia de las fichas no se toma como una etiqueta independiente: las trazas documentan cómo se llegó a la ficha. La fuente para determinar errores sigue siendo la comparación con las normas.

## Metodología

1. Se cargan los parámetros y las secciones observables de las normas originales.
2. Se calculan las violaciones normativas fila por fila: registro, documento, orden del box, circuito, plazo, cobertura, verificación y entidad a notificar. Para la terminación reservada (NOR-COB-04) se aplica el texto completo: la ficha registra `sin_cobertura` y se notifica a la entidad **esté activa o no la afiliación** (§3), salvo en accidentes de trabajo (§4).
3. Se excluyen de la inferencia de costumbres todas las filas con alguna violación.
4. Costumbre del voluntariado: se prueban todas las reglas `edad >= umbral` (cada edad observada), solas o combinadas con un dato observable del triage (acompañante, si puede responder, forma de llegada), y se elige la que explica más filas limpias. Ni el umbral ni la condición están fijados en el código.
5. Costumbre de credenciales: entre las filas limpias con obra social y prepaga se mide qué credencial se verificó y qué se registró cuando estaba inactiva; se exige una preferencia de al menos 90% para confirmarla.
6. Toda fila limpia que se aparta de una costumbre confirmada es `ruido`, en cualquier motivo o cobertura. Las trazas se usan para validar la correspondencia ID-operador-ficha y para contar verificaciones de afiliación.

El análisis completo, con frecuencias e IDs, queda en `historial_costumbres.json`. `costumbres.md` es la versión legible, con los IDs que respaldan cada lado de cada costumbre.

## Costumbres confirmadas

### 1. Avisar al voluntariado sólo a pacientes de 75 años o más sin acompañante

`aviso_al_voluntariado` es `sí` si y sólo si el paciente tiene 75 años o más y no tiene acompañante en la guardia. La regla explica 1.609 de 1.760 filas limpias; la mejor regla sólo por edad (también 75) explica 1.507, porque trata igual a los mayores acompañados.

| situación | aviso | filas limpias que la siguen | todas las filas de los 17 operadores sin desvíos |
|---|---|---:|---:|
| 75+ sin acompañante | `sí` | 167/186 | 185/185 |
| 75+ con acompañante | `no` | 114/126 | 141/141 |
| menos de 75 | `no` | 1.328/1.448 | 1.348/1.348 |

La versión anterior describía sólo "no avisar a menores de 75" y trataba al grupo de 75+ como mixto (175/306): esa mezcla era la suma de mayores solos y acompañados. Además, sólo marcaba `ruido` dentro de ocho estratos `motivo × cobertura` con muestra suficiente, por lo que desvíos idénticos quedaban `correcta` o `ruido` según el estrato. Esa compuerta se quitó: la costumbre se cumple al 100% en todas las situaciones para los operadores sin desvíos, así que no hace falta estratificar para separar costumbre de ruido.

Cruce con las 800 trazas de referencia de `entrenamiento/trazas.jsonl` (sólo verificación, no se usan como datos): 74/74 `sí` en 75+ sin acompañante, 67/67 `no` en 75+ con acompañante y 659/659 `no` en menores de 75.

### 2. Con obra social y prepaga, verificar sólo la prepaga, sin recurrir a la obra social

Cuando el paciente presenta las dos credenciales (sin lesión laboral, donde rige la ART), se verifica sólo la prepaga. Si está activa se registra y notifica según las normas; si está inactiva se registra `sin_cobertura` y no se verifica la obra social.

| evidencia | resultado |
|---|---:|
| filas limpias que verificaron la prepaga | 163/172 (94,8%) |
| prepaga inactiva, filas limpias → `sin_cobertura` | 39/39 |
| prepaga inactiva, todas las filas → `sin_cobertura` | 45/45 |
| trazas con dos credenciales y una sola verificación | 193/193 |
| trazas de referencia que verificaron sólo la prepaga | 65/65 |
| trazas de referencia con prepaga inactiva → `sin_cobertura` | 20/20 |

Las normas obligan a verificar la cobertura que se usa, pero no dicen cuál elegir: por eso es costumbre y no norma. Las 9 filas limpias que verificaron la obra social son de `OP-04`, `OP-11` y `OP-17` y son `ruido`.

## Desvíos por operador

| operador | ruido por voluntariado | ruido por credencial | total |
|---|---:|---:|---:|
| OP-04 | 38 | 1 | 39 |
| OP-11 | 60 | 4 | 64 |
| OP-17 | 53 | 4 | 57 |

Los tres operadores se apartan en ambas direcciones (avisan a menores o a mayores acompañados y omiten avisos a mayores solos), sin otra regla reconocible. No se confirma como otra costumbre: es variación individual y se etiqueta como `ruido`. Los otros 17 operadores no tienen desvíos.

## Cambios de etiqueta en la corrección del 2026-11-10

| causa | cambio | filas |
|---|---|---:|
| NOR-COB-04 con afiliación inactiva que notifica (antes se exigía no notificar) | `error` → `correcta` | 42 |
| ídem, con desvío de la costumbre del voluntariado | `error` → `ruido` | 3 |
| NOR-COB-04 con afiliación inactiva que **no** notifica | `correcta` → `error` | 6 |
| costumbre completa del voluntariado, sin compuerta de estratos | `correcta` → `ruido` | 67 |

Otras 3 filas con terminación reservada inactiva siguen `error` por otros campos. La costumbre de credenciales no cambió etiquetas: sólo se completó su descripción.

La ejecución reproducible es:

```text
python scripts/clasificar_historial.py --entrada historial/historial.csv --normas normas/original --trazas historial/trazas.jsonl --salida historial_clasificado.csv
```

## Archivos y resumen para el equipo

- [`costumbres.md`](costumbres.md): costumbres completas, método, desvíos por operador e IDs de respaldo de cada lado.
- [`historial_costumbres.json`](historial_costumbres.json): frecuencias, operadores e IDs completos.
- [`docs/resumen-costumbres-issue-11.md`](docs/resumen-costumbres-issue-11.md): versión breve para el equipo.
- [`scripts/clasificar_historial.py`](scripts/clasificar_historial.py): análisis reproducible.
- [`tests/test_clasificar_historial.py`](tests/test_clasificar_historial.py): pruebas de reglas, costumbres, trazas y determinismo.

## Verificación

```text
python -m unittest tests.test_clasificar_historial -v
```

## Issue #12: datos de entrenamiento

### Seleccion de historial

Se parte de las 2.000 filas clasificadas. Se excluyen las 240 filas `error` y las 160 `ruido`. Como todo desvío de la costumbre del voluntariado ya es `ruido`, se quitó la exclusión adicional de "ambiguas" (36 filas en la versión anterior); en su lugar, el selector falla si alguna fila `correcta` contradice la costumbre del informe.

Quedan **1.600 conversaciones reales** (antes 1.595: salen 31 por la costumbre completa y 6 por NOR-COB-04; entran 42 que antes eran `error` por NOR-COB-04). Cada una se obtiene de `historial/trazas.jsonl`, conserva el sistema, el mensaje inicial, todas las acciones y observaciones, y termina en `registrar_decision`. No se usan filas de `practica/` ni sus fichas esperadas.

### Casos sinteticos

El script usa `admision.generador.generar` con semilla **12012** y genera 5.000 candidatos. Selecciona 500 de forma determinista: 100 por circuito (`ecg`, `codigo_acv`, `sepsis_probable`, `sepsis_posible`, `ninguno`) y al menos 60 por cada dificultad especial (`art`, `dos_credenciales`, `acompanante`, `sin_nadie`). El balance efectivo es:

| dificultad | cantidad |
|---|---:|
| art | 70 |
| dos credenciales | 60 |
| acompanante | 63 |
| sin nadie | 63 |
| general | 244 |
| total | 500 |

Cada ficha sintetica se deduce desde los parametros cargados de `normas/original/`: documento, box, circuito, plazo, fuente de cobertura, afiliacion, digito reservado (con la misma regla de NOR-COB-04 que el clasificador) y notificacion. Las dos costumbres se leen de `historial_costumbres.json` y se aplican completas y deterministas: con dos credenciales se verifica sólo la preferida, y el aviso al voluntariado es `sí` sólo para 75+ sin acompañante (antes se muestreaba `sí` con probabilidad 175/306 en mayores). Resultado: 61 avisos (todos los 75+ sin acompañante) y ningún aviso en 50 mayores acompañados ni en 389 menores.

Las conversaciones se ejecutan sobre `admision.Entorno` con el buscador real del proyecto, construido con `entrega.construir_buscador` (misma variante por defecto, `baseline`, y misma carpeta de normas recibida) y con su `k_entorno` (2 secciones). Antes se usaba un buscador sintético propio que, por ejemplo, devolvía NOR-COB-02 + NOR-TDE-01 para la consulta de código ACV; ahora las 400 conversaciones con circuito recuperan su sección rectora (NOR-TDE-01/02/03). Las 500 fichas pasan la comparacion normativa y el validador de conversaciones v1.

Limitaciones: las normas siguen siendo `normas/original` (variar sus valores es una decisión pendiente del equipo). Algunas secciones citadas en `fundamentos` no aparecen en ninguna búsqueda de la conversación (NOR-COB-03 en 246, NOR-ADM-03 en 114 y NOR-COB-04 en 38 sintéticos), porque las consultas del guion no cambiaron.

### Dataset final y reproducibilidad

El dataset combinado tiene **2.100 conversaciones**:

| origen | cantidad | proporcion |
|---|---:|---:|
| historial | 1.600 | 76,2% |
| sintetico | 500 | 23,8% |
| total | 2.100 | 100% |

El comando reproducible es:

```text
python -m scripts.preparar_datos_entrenamiento --raiz . --salida datos/entrenamiento --semilla 12012 --candidatos 5000
```

La salida se valida con:

```text
python -m scripts.validar_conversaciones datos/entrenamiento/conversaciones.jsonl
```

Resultado: 2.100 registros, 2.100 validos, sin errores. Respecto de la versión anterior cambiaron 53 fichas sintéticas (45 por el aviso al voluntariado y 10 entidades a notificar por NOR-COB-04; dos fichas cambian en ambos campos) y los mensajes de las 500 conversaciones sintéticas (observaciones del buscador real y nueva referencia `buscador:`). `datos/entrenamiento/manifiesto.json` conserva procedencia, semilla, balance, buscador, cantidades y SHA-256; `auditoria_sinteticos.jsonl` conserva la ficha deducida de cada sintetico. El ensamblador no sobrescribe una salida existente.

### Archivos del issue

- [`datos/entrenamiento/conversaciones.jsonl`](datos/entrenamiento/conversaciones.jsonl): dataset combinado v1.
- [`datos/entrenamiento/auditoria_sinteticos.jsonl`](datos/entrenamiento/auditoria_sinteticos.jsonl): auditoria de fichas sinteticas.
- [`datos/entrenamiento/manifiesto.json`](datos/entrenamiento/manifiesto.json): procedencia, balance y hash.
- [`scripts/seleccionar_entrenamiento.py`](scripts/seleccionar_entrenamiento.py): curacion y conversion de trazas reales.
- [`scripts/generar_sinteticos_entrenamiento.py`](scripts/generar_sinteticos_entrenamiento.py): generacion, deduccion y balance sintetico.
- [`scripts/preparar_datos_entrenamiento.py`](scripts/preparar_datos_entrenamiento.py): ensamblado y validacion final.
- [`tests/test_seleccionar_entrenamiento.py`](tests/test_seleccionar_entrenamiento.py), [`tests/test_generar_sinteticos_entrenamiento.py`](tests/test_generar_sinteticos_entrenamiento.py) y [`tests/test_preparar_datos_entrenamiento.py`](tests/test_preparar_datos_entrenamiento.py): pruebas reproducibles, incluidas la costumbre completa en sintéticos, NOR-COB-04 con afiliación inactiva y búsquedas idénticas a las del buscador de la entrega.

## Parte 1: conversaciones

Dataset v2 (issue #8), generado por `scripts/generar_conversaciones.py` con el prompt `baseline`
(sha256 `944ae7f6…f474`) y el buscador de `entrega.construir_buscador` (variante `baseline`, k=2),
semilla 12012. Comando, reglas y salidas: [contrato de conversaciones](docs/conversaciones-entrenamiento.md#dataset-v2-scriptsgenerar_conversaciones-issue-8).

```text
uv run --locked --no-sync python -m scripts.generar_conversaciones
uv run --locked --no-sync python -m scripts.validar_conversaciones datos/conversaciones/v2-baseline/conversaciones.jsonl
```

| origen | entradas | excluidas | conversaciones |
|---|---:|---:|---:|
| reales: historial `correcta` | 1.600 (de 2.000; 240 `error` y 160 `ruido` no entran) | 0 | 1.600 |
| reales: referencia de la cátedra (`entrenamiento/trazas.jsonl`) | 800 | 0 | 800 |
| sintéticas (`admision.generador`, valores variados) | 500 | 0 | 500 |
| total | | | **2.900** |

Validación: 2.900 registros válidos, 0 errores (SHA-256 `3e6b8897…d2ee`). Tres generaciones
independientes dieron archivos idénticos byte a byte (`cmp`).

- **Búsquedas del buscador real.** Cada búsqueda se re-ejecutó sobre una copia de las normas con los
  valores de esa conversación. En las 800 de referencia hay 706 conjuntos de valores distintos; 584 de sus
  fichas serían erróneas con los valores de `normas/original` (469 por plazo, 152 por box), por eso no se
  re-ejecutaron contra la carpeta original. Ningún valor mostrado cambió respecto de las trazas; solo cambia
  el desempate de la segunda sección en dos consultas (NOR-COB-01 en lugar de NOR-COB-02).
- **Revalidación.** Las 2.400 fichas reales cumplen las normas con sus valores y las dos costumbres; las
  1.600 de historial muestran solo valores originales. Ningún parámetro relevante quedó sin observar.
- **Sintéticas variadas.** 500 conjuntos de valores distintos, sorteados entre los observados en la
  referencia; 365 fichas serían distintas con los valores originales. Cada una busca las secciones que la
  deciden y las que cita en `fundamentos` (500/500 con toda su evidencia).
- **Buscador `baseline`.** Se eligió por el resultado con el agente en Kaggle (50 casos de práctica,
  normas originales): `baseline` 10/50 contra `prefijos` 6/50.
- **Curado para que el modelo responda con la evidencia que recuperó, no de memoria.** Sin curar, con
  `baseline`, 49 conversaciones de referencia y 93 de historial no recuperaban NOR-COB-02 (accidente de
  trabajo con la consulta combinada «accidente de trabajo ART dolor torácico plazo»). En esas 142 se agregó
  una búsqueda «accidente de trabajo ART» justo después de la combinada, sin modificar las consultas
  originales. Además, los `fundamentos` reales citaban una sección no recuperada en 783 de referencia y
  1.543 de historial, una por conversación (NOR-COB-03 1.199, NOR-COB-02 659, NOR-COB-01 468): se quitaron
  esas 2.326 citas. Ninguna ficha quedó sin `fundamentos`.
- **Cobertura de evidencia.** Las conversaciones que recuperan todas sus secciones rectoras son 800/800 de
  referencia, 1.600/1.600 de historial y 500/500 sintéticas, y ninguna cita fundamentos no recuperados.
- **Práctica.** 0 conversaciones comparten triage, documento, afiliado ni CUIL con `practica/casos.jsonl`.
- **Largo.** Caracteres por conversación (no tokens; no se tokenizó): mediana 10.831 en historial, 10.826 en referencia y 15.618 en
  sintéticas (máximo 21.939).

## Parte 2: buscador

### Enfoque

El buscador (`admision_tp/buscador.py`, conectado por `entrega.construir_buscador`) se arma siempre con
la carpeta de normas recibida: indexa una sección completa por archivo `NOR-*.md` e incorpora su línea de
valores desde el `parametros.yaml` de esa carpeta, así los números de las normas b/c o del test llegan
por búsqueda y no por los pesos. Es BM25 (k1=1,5, b=0,75) sobre texto normalizado (minúsculas, sin
tildes); el texto devuelto no se modifica. No usa servicios externos, embeddings ni descargas. Las variantes
se eligen con `BUSCADOR_VARIANTE`; el default de la entrega es `baseline`. Detalle técnico y reproducción en
la [guía del buscador](docs/buscador-baseline.md).

### Evaluación por separado (`admision.evaluar_rag`)

Comando, sobre las 60 consultas de desarrollo y normas originales, con el código de `4e3613a`
(repetido sin cambios sobre `d37f492`):

```text
BUSCADOR_VARIANTE=<variante> uv run --locked --no-sync python -m admision.evaluar_rag --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl
```

| Buscador | Secciones por búsqueda | Completas | Recall de evidencia | Precisión | Tokens/consulta |
|---|---|---:|---:|---:|---:|
| Referencia BM25 de la cátedra | 2 | 51/60 | 0,875 | 0,492 | 580 |
| `baseline` (default de la entrega) | 2 | 48/60 | 0,817 | 0,467 | 618 |
| `sin_comunes` | 2 | 50/60 | 0,850 | 0,483 | 591 |
| `prefijos` | 1 a 3 (umbral 0,6) | **55/60** | **0,933** | **0,686** | **549** |

- `baseline`: BM25 sobre la sección entera, 2 secciones. Queda por debajo de la referencia; las consultas
  coloquiales tienen recall 0,25 porque depende de compartir palabras.
- `sin_comunes`: quita una lista fija de palabras comunes del índice y de la consulta (conserva `no`, `sin`,
  `ni` y números). Gana 3 consultas y pierde 1 contra `baseline`.
- `prefijos`: sin palabras comunes, cada palabra recortada a sus primeras 4 letras (une `registro`,
  `registrarlo`, `registra`) y hasta 3 secciones, descartando las que puntúan menos de 0,6 × la primera.
  Supera la referencia en las cuatro métricas; gana 9 consultas y pierde 2 (d027 y d052, por colisiones de
  prefijo). Coloquiales: recall 0,75.

**Variantes probadas y descartadas.** Además de las tres seleccionables se midieron 68 configuraciones,
reproducibles con `python -m scripts.explorar_buscadores --salida <carpeta nueva>`; la tabla completa está en
la guía. En resumen:

| Familia | Mejor resultado | Por qué se descartó |
|---|---|---|
| Fragmentos por párrafo (top 2–5, con o sin puntaje de sección) | 49/60, 0,833, 291–341 tokens | Menos tokens, pero menos recall que la referencia |
| Sección recortada a los párrafos con coincidencia | 51/60, 0,867, 535 tokens | Pierde evidencia de párrafos sin palabras de la consulta |
| Otros stemmers (plurales, sufijos, prefijo 5 o 6) | 51/60, 0,867 | Peores que prefijo 4 |
| Título con más peso, ajustes de k1/b | 49–51/60 | Ganancia de 1 consulta como mucho |
| Peso 0,25 para palabras comunes | 49/60, 0,833, 605 tokens | Pierde d011 igual que `sin_comunes`, gana menos |
| Fusión BM25 + n-gramas de caracteres (RRF) | 48/60, 0,817, 612 tokens | No supera a `sin_comunes` |
| `prefijos` sin umbral (k=3) | 56/60, 0,942, 895 tokens | +45 % de tokens por 1 consulta |
| `prefijos` con umbral 0,4 / 0,5 / 0,7 / 0,8 | 56 / 55 / 53 / 52 de 60 | 0,6 es el codo entre recall y tokens |

Prefijo, umbral y k se eligieron mirando estas mismas 60 consultas: son números de desarrollo, no una
evaluación independiente. Con normas b y c el texto es el mismo y solo cambian los valores; con la evidencia
ajustada a esos valores las tres variantes dan lo mismo que con original. Densos, híbridos con embeddings y
rerankers quedaron fuera: requieren descargas o dependencias nuevas no autorizadas. Lo que sigue sin
resolverse (d030 pecho ↔ dolor torácico, d023 abuela ↔ edad) necesita sinónimos.

### Cuánto rinde el agente con cada buscador

Como lo que cuenta es lo que le sirve al agente, antes de cambiar el default se compararon `baseline` y
`prefijos` jugando los 50 casos de práctica con normas originales. Configuración: Kaggle T4 x2, notebook
`franciscocalveyra/tp-admision-baseline` versión 22, commit `4e3613a` limpio, Gemma 4 E4B **sin entrenar**,
float16, `device_map=auto`, `MODELO_LOTE=4`, prompt sin optimizar de `AgenteJSON`. Comando por
variante: `BUSCADOR_VARIANTE=<variante> MODELO_LOTE=4 python infra/kaggle_baseline.py --variante original
--cantidad 50 --salida <carpeta nueva>`; puntuado local con `python -m admision.correr puntuar --esperadas
practica/esperadas-original.jsonl`.

| Métrica | `baseline` | `prefijos` |
|---|---:|---:|
| Casos resueltos | **10/50** | 6/50 |
| Parcial medio | **0,831** | 0,796 |
| registro / box / circuito / plazo | 0,76 / 0,72 / 0,98 / 0,92 | 0,62 / 0,68 / 0,94 / 0,84 |
| cobertura / entidad / aviso / verificación | 0,80 / 0,82 / 0,72 / 0,76 | 0,78 / 0,82 / 0,72 / 0,76 |
| Llamadas / búsquedas por caso | 4,9 / 1,34 | 5,0 / 1,40 |
| Caracteres de contexto por caso | 19.954 | 20.689 |
| Segundos por caso (T4, lote 4) | 15,2 | 15,7 |

`baseline` reproduce exactamente la corrida anterior del modelo sin entrenar (10/50, 0,831; commit `a89f5b9`).
Por caso, `prefijos` gana 3 (pev-005, 019, 037) y pierde 7 (pev-010, 012, 017, 021, 024, 049, 050); con 10
casos discordantes la prueba de signos da p≈0,34, así que la diferencia **no es concluyente**, pero tampoco
hay señal de mejora.

**Por qué el mejor recall no ayudó.** `baseline` devuelve siempre 2 secciones; `prefijos` devolvió 1 en 30
búsquedas, 3 en 26 y 2 en 14. Con 3 suele sumar secciones parecidas de otras áreas (por ejemplo NOR-DER-01 o
NOR-GUA-02), justamente las que no deciden la ficha. Por eso el contexto por caso subió 3,7 % en vez de bajar
como en la evaluación separada, y cambió el recorrido del agente (en pev-021 nunca buscó el circuito). Es lo
que muestran las trazas, no una causa probada.

**Costo de contexto.** Con las dos variantes el agente hace pocas búsquedas (1,3–1,4 por caso) y cada caso usa
unos 20.000 caracteres de contexto (~5.000 tokens a 4 caracteres por token). `prefijos` trae menos tokens por
consulta en `evaluar_rag` (549 contra 618) pero más contexto por caso con el agente. El costo de GPU de la
comparación fue 2.968 s de sesión T4 (≈ 49,5 min, ~20 min de instalación de dependencias); tiempos T4 **no
comparables** con la A100-40GB de la entrega.

### Decisión

El default de la entrega sigue siendo `baseline` (`entrega.py` no cambia): es el que mejor resolvió con el
agente y el que se usó para generar las conversaciones de entrenamiento. `prefijos` queda documentado y
seleccionable como resultado negativo: mejora la evaluación separada pero no al agente sin entrenar. Si
alcanzan tiempo y cuota, se repetirá la comparación con el modelo entrenado.

### Límites

- Una sola corrida por variante, 50 casos, solo normas originales y modelo sin entrenar.
- Los hiperparámetros de `prefijos` se ajustaron con las consultas de desarrollo; la práctica se usó solo para
  medir, nunca para ajustar.
- Resultados de agente en T4; falta medir en A100-40GB.
- Artefactos de las corridas locales e ignorados por Git: `corridas/kaggle/v22/` (runner, métricas, trazas
  y puntuación) y `corridas/rag-*-20261110/` (comparaciones de `evaluar_rag`).
