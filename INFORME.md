# Informe: agente de admisión de guardia

## Resumen

**Entrega.** `entrega.py` arma el agente con Gemma 4 E4B (`google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`),
el adapter LoRA **`FranCalveyra/tp-admision@770c9fac84d5718b0b2cf4c40c3451cad322ac9e`, carpeta `final`** (v39), el
prompt `procedimiento-v1` (sha256 `8c692c2c…1ecdd`), el buscador `baseline` (BM25, 2 secciones completas) y lote 4.

| Resultado (práctica, 50 casos; solo para reportar) | original | b | c | Total |
|---|---:|---:|---:|---:|
| **Entrega (`ambos`: adapter v39 + `procedimiento-v1`)** | **36/50** | **28/50** | **36/50** | **100/150** |
| Referencia de la cátedra, SFT con datos perfectos | 25/50 | 24/50 | 25/50 | 74/150 |
| Referencia de la cátedra, sin entrenar | 12/50 | 15/50 | 12/50 | 39/150 |
| Nuestro modelo sin entrenar (prompt de la cátedra) | 10/50 | 10/50 | 7/50 | 27/150 |

- **Cómo se eligió:** prompt y adapter se eligieron solo con un conjunto de desarrollo propio (`datos/prompt-dev/v1`,
  40 casos); el adapter v39 dio 31/40 contra 26/40 del anterior. La práctica se corrió después (Kaggle v40) y no
  cambió ninguna decisión.
- **Costo por caso:** 27,8–28,8 s/caso y 19,94 GB de VRAM pico en Kaggle 2× T4, fp16, lote 4 (Parte 5).
- **Limitación principal:** **nada se validó en una A100 de 40 GB**; todo es Kaggle T4
  ([KI-A100-001](README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb)).

**Cómo leer los números.** Cada tabla dice el comando, el punto de control y el prompt, o enlaza la procedencia
generada por el script. Las evaluaciones están versionadas sin editar en [`evaluaciones/`](evaluaciones/README.md):
conversaciones completas, tiempos, métricas por micro-lote y `runner.json` con commit, modelo, adapter, prompt y
buscador. La bitácora completa de cada pasada está en [`docs/progreso.md`](docs/progreso.md).

Se distinguen cuatro niveles: *implementado*, *probado con dobles* (tests CPU), *validado con dependencia real* y
*validado en GPU (T4)*. Ningún resultado está *validado en A100*.

---

## Parte 1: curado

### 1.1 Clasificación del historial

**Fuentes.** El script usa tres fuentes:

- `historial/historial.csv`: atributos de ingreso, cobertura y ficha final.
- `historial/trazas.jsonl`: la conversación completa, el operador y la ficha final de cada ID.
- `normas/original/`: las secciones normativas y `parametros.yaml`.

Las trazas se validan antes de usarse como evidencia:

- Hay 2.000 IDs únicos.
- Cada operador coincide con el CSV.
- Las 2.000 fichas finales coinciden campo por campo con su fila.
- La instrumentación tiene 2.000 lecturas de triage, 2.000 decisiones, 1.587 verificaciones de afiliación, 1.405
  preguntas y 6.197 búsquedas normativas, con un promedio de 6,5945 pasos por traza.

Las trazas documentan cómo se llegó a la ficha, pero no son una etiqueta independiente. Los errores se determinan
comparando con las normas.

**Método.**

1. Se cargan los parámetros y las secciones observables de las normas originales.
2. Se calculan las violaciones normativas fila por fila: registro, documento, orden del box, circuito, plazo,
   cobertura, verificación y entidad a notificar.
   - Para la terminación reservada (NOR-COB-04) se aplica el texto completo: la ficha registra `sin_cobertura` y se
     notifica a la entidad **esté activa o no la afiliación** (§3), salvo en accidentes de trabajo (§4).
3. Las filas con alguna violación se excluyen de la inferencia de costumbres.
4. Las costumbres se infieren de las filas limpias (1.2).
5. Toda fila limpia que se aparta de una costumbre confirmada es `ruido`, en cualquier motivo o cobertura. Una
   violación normativa siempre tiene prioridad sobre una costumbre: `error > ruido`.

```text
python scripts/clasificar_historial.py --entrada historial/historial.csv --normas normas/original --trazas historial/trazas.jsonl --salida historial_clasificado.csv
python -m unittest tests.test_clasificar_historial -v
```

**Pasadas.**

| Pasada | correcta | error | ruido | Qué cambió |
|---|---:|---:|---:|---|
| 2026-11-05, auditoría de #10 | 1.631 | 279 | 90 | Corrige la cobertura con dos credenciales: usa la credencial efectivamente verificada en vez de la primera. Cambiaron 115 filas respecto de `959838a`. |
| **2026-11-10, corrección vigente** | **1.600** | **240** | **160** | NOR-COB-04 con el texto completo y la costumbre del voluntariado completa, sin compuerta de estratos (ver abajo). |

La pasada vigente produjo estos cambios de etiqueta:

| causa | cambio | filas |
|---|---|---:|
| NOR-COB-04 con afiliación inactiva que notifica (antes se exigía no notificar) | `error` → `correcta` | 42 |
| ídem, con desvío de la costumbre del voluntariado | `error` → `ruido` | 3 |
| NOR-COB-04 con afiliación inactiva que **no** notifica | `correcta` → `error` | 6 |
| costumbre completa del voluntariado, sin compuerta de estratos | `correcta` → `ruido` | 67 |

Otras 3 filas con terminación reservada inactiva siguen siendo `error` por otros campos. La costumbre de
credenciales no cambió etiquetas: solo se completó su descripción.

**Límite.** El CSV no permite comprobar boxes disponibles, evaluación médica, retriage, camas, aislamiento ni
derivaciones. Esas normas quedaron documentadas como no verificables.

### 1.2 Costumbres

Detalle, IDs de respaldo de cada lado y frecuencias:

- [`costumbres.md`](costumbres.md)
- [`historial_costumbres.json`](historial_costumbres.json)
- [resumen para el equipo](docs/resumen-costumbres-issue-11.md)

**Cómo se encontraron.**

- **Voluntariado:** se prueban todas las reglas `edad >= umbral`, una por cada edad observada. Cada regla se prueba
  sola o combinada con un dato observable del triage (acompañante, si puede responder, forma de llegada). Se elige la
  que explica más filas limpias. Ni el umbral ni la condición están fijados en el código.
- **Credenciales:** entre las filas limpias con obra social y prepaga se mide qué credencial se verificó y qué se
  registró cuando estaba inactiva. Para confirmarla se exige una preferencia de al menos 90 %.

**1. Avisar al voluntariado solo a pacientes de 75 años o más sin acompañante.** La regla explica 1.609 de 1.760
filas limpias. La mejor regla solo por edad (también 75) explica 1.507, porque trata igual a los mayores
acompañados.

| situación | aviso | filas limpias que la siguen | todas las filas de los 17 operadores sin desvíos |
|---|---|---:|---:|
| 75+ sin acompañante | `sí` | 167/186 | 185/185 |
| 75+ con acompañante | `no` | 114/126 | 141/141 |
| menos de 75 | `no` | 1.328/1.448 | 1.348/1.348 |

Esta costumbre tuvo dos pasadas:

- **Primera pasada (2026-11-05):** solo encontró «no avisar a menores de 75» (1.298/1.415 filas limpias menores) y
  trataba al grupo de 75+ como mixto (175/306). Esa mezcla era la suma de mayores solos y acompañados. Además,
  marcaba `ruido` solo dentro de ocho estratos `motivo × cobertura`, por lo que desvíos idénticos quedaban
  `correcta` o `ruido` según el estrato.
- **Segunda pasada:** se quitó la compuerta de estratos. Para los operadores sin desvíos la costumbre se cumple al
  100 %.

La regla se cruzó con las 800 trazas de referencia de `entrenamiento/trazas.jsonl`: 74/74 `sí` en 75+ sin
acompañante, 67/67 `no` en 75+ con acompañante y 659/659 `no` en menores de 75.

**2. Con obra social y prepaga, verificar solo la prepaga.** Cuando el paciente presenta las dos credenciales, se
verifica solo la prepaga. No aplica a lesiones laborales, donde rige la ART. Si la prepaga está activa se registra y
notifica según las normas. Si está inactiva se registra `sin_cobertura` y no se verifica la obra social.

| evidencia | resultado |
|---|---:|
| filas limpias que verificaron la prepaga | 163/172 (94,8 %) |
| prepaga inactiva, filas limpias → `sin_cobertura` | 39/39 |
| prepaga inactiva, todas las filas → `sin_cobertura` | 45/45 |
| trazas con dos credenciales y una sola verificación | 193/193 |
| trazas de referencia que verificaron solo la prepaga | 65/65 |
| trazas de referencia con prepaga inactiva → `sin_cobertura` | 20/20 |

Las normas obligan a verificar la cobertura que se usa, pero no dicen cuál elegir: por eso es costumbre y no norma.
Las 9 filas limpias que verificaron la obra social son de `OP-04`, `OP-11` y `OP-17` y son `ruido`. La primera
pasada medía 162/171 (94,7 %).

**Desvíos por operador.** Todas las filas `ruido` vienen de tres operadores:

| operador | ruido por voluntariado | ruido por credencial | total |
|---|---:|---:|---:|
| OP-04 | 38 | 1 | 39 |
| OP-11 | 60 | 4 | 64 |
| OP-17 | 53 | 4 | 57 |

- Estos operadores se apartan en las dos direcciones: avisan a menores o a mayores acompañados, y omiten avisos a
  mayores solos.
- No siguen otra regla reconocible, así que no se confirma una tercera costumbre: es variación individual y se
  etiqueta como `ruido`.
- Los otros 17 operadores no tienen desvíos.

### 1.3 Conversaciones de entrenamiento

Las conversaciones se armaron en seis pasadas. Las cuatro primeras rehacen el dataset completo y las dos últimas son
subconjuntos para seguir entrenando el adapter.

| Pasada | Dataset | Reales | Sintéticas | Total | Uso |
|---|---|---:|---:|---:|---|
| 1 (#12, 2026-11-05) | `datos/entrenamiento` v1, primera versión | 1.595 | 500 | 2.095 | Reemplazada |
| 2 (#12, 2026-11-10) | `datos/entrenamiento` v1 corregido | 1.600 | 500 | 2.100 | Reemplazada por v2 |
| 3 (#8) | `datos/conversaciones/v2-baseline` | 2.400 (1.600 de historial + 800 de referencia) | 500 | 2.900 | Prompt `baseline`; no se entrenó |
| 4 (#13) | `datos/conversaciones/v3-procedimiento` | 2.400 | 500 | **2.900** | **SFT base** (`dc41674a`) |
| 5 (#15) | `v4-aviso` | 504 (335 + 169) | 129 | 633 | v36, no adoptada |
| 6 (#15) | `v5-aviso-peso` / **`v6-aviso-datos`** | 920 (625 + 295) | 213 | 1.133 | v38 (no adoptada) / **v39 (entrega)** |

Las cantidades de las pasadas 5 y 6 salen de contar `origen` en cada `conversaciones.jsonl`.

**Pasadas 1 y 2: datos v1 (issue #12).**

- **Reales:** se parte de las filas `correcta` del historial. La pasada 1 excluía además 36 avisos «ambiguos»
  (1.595 reales).
- **Corrección de la pasada 2:** con la costumbre completa, todo desvío ya es `ruido`, así que se quitó esa
  exclusión y quedaron 1.600 reales. Salen 31 por la costumbre completa y 6 por NOR-COB-04, y entran 42 que antes
  eran `error` por NOR-COB-04.
- **Guarda:** el selector falla si alguna fila `correcta` contradice la costumbre.
- **Sintéticas:** `admision.generador.generar` con semilla **12012** genera 5.000 candidatos y se eligen 500 de forma
  determinista:
  - 100 por circuito.
  - Al menos 60 por cada dificultad especial: ART 70, dos credenciales 60, acompañante 63, sin nadie 63 y
    general 244.
- **Fichas sintéticas:** se deducen de los parámetros de `normas/original/`, con la misma regla de NOR-COB-04.
- **Costumbres en los sintéticos:** se aplican completas y deterministas, leídas de `historial_costumbres.json`.
  - La pasada 1 sorteaba el aviso de los mayores con probabilidad 175/306.
  - En la pasada 2 hay 61 avisos (todos los 75+ sin acompañante) y ningún aviso en 50 mayores acompañados ni en
    389 menores.
- **Buscador:** las conversaciones se ejecutan sobre `admision.Entorno` con el buscador de la entrega
  (`entrega.construir_buscador`, `baseline`, k=2).
  - Antes se usaba un buscador sintético que, por ejemplo, devolvía NOR-COB-02 + NOR-TDE-01 para el código ACV.
  - Ahora las 400 conversaciones con circuito recuperan su sección rectora.
- **Validación:** 2.100 registros, 2.100 válidos.
  - Respecto de la pasada 1 cambiaron 53 fichas sintéticas: 45 por el aviso y 10 entidades por NOR-COB-04; dos
    fichas cambian en ambos campos.
- **Límite:** algunas secciones citadas en `fundamentos` no aparecían en ninguna búsqueda (NOR-COB-03 en 246,
  NOR-ADM-03 en 114 y NOR-COB-04 en 38 sintéticos). La pasada 3 lo corrige.

```text
python -m scripts.preparar_datos_entrenamiento --raiz . --salida datos/entrenamiento --semilla 12012 --candidatos 5000
python -m scripts.validar_conversaciones datos/entrenamiento/conversaciones.jsonl
```

**Pasada 3: dataset v2 (issue #8).** Lo genera `scripts/generar_conversaciones.py`:

- Prompt `baseline` (sha256 `944ae7f6…f474`).
- Buscador `baseline`, k=2.
- Semilla 12012.

Validación: 2.900 registros válidos, 0 errores (SHA-256 `3e6b8897…d2ee`). Tres generaciones independientes dieron
archivos idénticos byte a byte. Detalle en el
[contrato de conversaciones](docs/conversaciones-entrenamiento.md#dataset-v2-scriptsgenerar_conversaciones-issue-8).

```text
uv run --locked --no-sync python -m scripts.generar_conversaciones
uv run --locked --no-sync python -m scripts.validar_conversaciones datos/conversaciones/v2-baseline/conversaciones.jsonl
```

| origen | entradas | excluidas | conversaciones |
|---|---|---:|---:|
| reales: historial `correcta` | 1.600 (de 2.000; 240 `error` y 160 `ruido` no entran) | 0 | 1.600 |
| reales: referencia de la cátedra (`entrenamiento/trazas.jsonl`) | 800 | 0 | 800 |
| sintéticas (`admision.generador`, valores variados) | 500 | 0 | 500 |
| total | | | **2.900** |

- **Búsquedas re-ejecutadas.** Cada búsqueda se re-ejecutó sobre una copia de las normas con los valores de esa
  conversación.
  - Las 800 de referencia tienen 706 conjuntos de valores distintos. Con los valores de `normas/original`, 584 de
    sus fichas serían erróneas (469 por plazo, 152 por box).
  - Ningún valor mostrado cambió. Solo cambió el desempate de la segunda sección en dos consultas.
- **Revalidación.** Las 2.400 fichas reales cumplen las normas con sus valores y las dos costumbres. Las 1.600 de
  historial muestran solo valores originales.
- **Sintéticas variadas.** Tienen 500 conjuntos de valores, sorteados entre los observados en la referencia; con los
  valores originales, 365 fichas serían distintas. Cada una busca las secciones que la deciden y las que cita.
- **Curado para responder con la evidencia recuperada, no de memoria.**
  - Con la consulta combinada, 49 conversaciones de referencia y 93 de historial no recuperaban NOR-COB-02. En esas
    142 se agregó la búsqueda «accidente de trabajo ART».
  - Los `fundamentos` citaban una sección no recuperada en 783 de referencia y 1.543 de historial (NOR-COB-03 1.199,
    NOR-COB-02 659, NOR-COB-01 468). Se quitaron esas 2.326 citas y ninguna ficha quedó sin `fundamentos`.
  - Resultado: las 2.900 conversaciones recuperan todas sus secciones rectoras y ninguna cita fundamentos no
    recuperados.
- **Práctica.** Ninguna conversación comparte triage, documento, afiliado ni CUIL con `practica/casos.jsonl`.
- **Largo en caracteres.** Mediana de 10.831 en historial, 10.826 en referencia y 15.618 en sintéticas (máximo
  21.939).
- **Uso de la referencia.** Las 800 trazas de referencia de la cátedra entran como datos desde esta pasada. En el
  análisis de costumbres (1.2) solo se usaron para verificar.

**Pasada 4: v3-procedimiento.** Es el mismo dataset regenerado con el prompt elegido en la Parte 3; frente a v2 solo
cambia el mensaje de sistema.

- Tiene 2.900 conversaciones válidas, con `dirty=false` y regeneración idéntica.
- Tokenizado con el tokenizer real (`manifiesto-tokens.json`): 9.990.837 tokens (~3.445 por conversación),
  1.015.072 supervisados (10,2 %). La mediana es 3.331 y el máximo 6.322 por conversación. Con `max_seq_length` 8192
  ninguna se recortó.
- Tiene 302 avisos `true`, que son todos los 75+ sin acompañante, y 2.598 `false`.

**Pasadas 5 y 6: subconjuntos para el aviso al voluntariado (issue #15).** Los arma `scripts.subconjunto_aviso`
sobre v3 (Parte 4).

| Dataset | Composición |
|---|---|
| `v4-aviso` | 633: 302 positivos, 231 mayores con acompañante y 100 del resto; semilla 15 |
| `v5-aviso-peso` | 1.133: 302 con aviso, 231 mayores con acompañante y 600 del resto (27 % positivos) |
| `v6-aviso-datos` | v5 con un prefijo en el último mensaje (`registrar_decision`) |

- El prefijo de `v6-aviso-datos` es `Edad: N años. Acompañante: X.` y lo agrega `scripts.anteponer_datos_aviso`.
- Los valores se copian de la nota de triage y el umbral no se escribe.
- 1.133 registros válidos, 0 errores, manifiesto con `dirty=false`.

---

## Parte 2: buscador

### Enfoque

El buscador (`admision_tp/buscador.py`, conectado por `entrega.construir_buscador`) se arma siempre con la carpeta de
normas recibida.

- Indexa una sección completa por archivo `NOR-*.md` y le agrega su línea de valores del `parametros.yaml` de esa
  carpeta. Así los números de las normas b, c o del test llegan por búsqueda y no por los pesos.
- Es BM25 (k1=1,5, b=0,75) sobre texto normalizado (minúsculas, sin tildes). El texto devuelto no se modifica.
- No usa servicios externos, embeddings ni descargas.
- Las variantes se eligen con `BUSCADOR_VARIANTE`; el default de la entrega es `baseline`.

Detalle y reproducción en la [guía del buscador](docs/buscador-baseline.md).

### Evaluación por separado (`admision.evaluar_rag`)

Corre sobre las 60 consultas de desarrollo y normas originales, con el código de `4e3613a` (repetido sin cambios
sobre `d37f492`):

```text
BUSCADOR_VARIANTE=<variante> uv run --locked --no-sync python -m admision.evaluar_rag --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl
```

| Buscador | Secciones por búsqueda | Completas | Recall de evidencia | Precisión | Tokens/consulta |
|---|---|---:|---:|---:|---:|
| Referencia BM25 de la cátedra | 2 | 51/60 | 0,875 | 0,492 | 580 |
| **`baseline` (entrega)** | 2 | 48/60 | 0,817 | 0,467 | 618 |
| `sin_comunes` | 2 | 50/60 | 0,850 | 0,483 | 591 |
| `prefijos` | 1 a 3 (umbral 0,6) | **55/60** | **0,933** | **0,686** | **549** |
| BM25 con 4 secciones (comparación) | 4 | 57/60 | 0,950 | 0,271 | 1.270 |

- **`baseline`** queda por debajo de la referencia. Las consultas coloquiales tienen recall 0,25 porque dependen de
  compartir palabras.
- **`sin_comunes`** quita una lista fija de palabras comunes, pero conserva `no`, `sin`, `ni` y los números. Gana
  d022, d026 y d028 y pierde d011 contra `baseline`.
- **`prefijos`** recorta cada palabra a sus primeras 4 letras y trae hasta 3 secciones. Descarta las que puntúan
  menos de 0,6 × la primera.
  - Supera a la referencia en las cuatro métricas.
  - Gana 9 consultas y pierde 2 (d027 y d052, por colisiones de prefijo).
  - El recall de las consultas coloquiales es 0,75.

**Pasadas que no se adoptaron.** El script de exploración mide 68 configuraciones (las entradas de `INTENTOS` en
`scripts/explorar_buscadores.py`, incluidas `baseline` y `sin_comunes`). Se reproducen con
`python -m scripts.explorar_buscadores --salida <carpeta nueva>`; la tabla completa está en
[buscador baseline](docs/buscador-baseline.md#exploración-descartada).

| Familia | Mejor resultado | Por qué se descartó |
|---|---|---|
| Fragmentos por párrafo (top 2–5) | 39–46/60, 0,683–0,792, 176–389 tokens | Pierde el contexto de la sección |
| Fragmentos por párrafo con stop + prefijo 4 (top 4–5) | 49/60, 0,833, 291–341 tokens | Menos tokens, pero menos recall que la referencia |
| Sección recortada a los párrafos con coincidencia | 51/60, 0,867, 535 tokens | Pierde evidencia de párrafos sin palabras de la consulta |
| Otros stemmers (plurales, sufijos, prefijo 5 o 6) | 51/60, 0,867 | Peores que prefijo 4 |
| Título con más peso, ajustes de k1/b | 49–51/60 | Ganancia de 1 consulta como mucho |
| Peso 0,25 para palabras comunes | 49/60, 0,833, 605 tokens | Pierde d011 igual que `sin_comunes` y gana menos |
| Fusión BM25 + n-gramas de caracteres (RRF) | 48/60, 0,817, 612 tokens | No supera a `sin_comunes` |
| `prefijos` sin umbral (k=3) | 56/60, 0,942, 895 tokens | +45 % de tokens por 1 consulta |
| `prefijos` con umbral 0,4 / 0,5 / 0,7 / 0,8 | 56 / 55 / 53 / 52 de 60 | 0,6 es el codo entre recall y tokens |

**Límites de esta evaluación:**

- Prefijo, umbral y k se eligieron con estas mismas 60 consultas: son números de desarrollo.
- Con normas b y c cambian solo los valores, y las tres variantes dan lo mismo que con original.
- Densos, híbridos y rerankers quedaron fuera porque requieren descargas no autorizadas.
- Siguen sin resolverse d030 (pecho ↔ dolor torácico) y d023 (abuela ↔ edad), que necesitan sinónimos.

### Cuánto rinde el agente con el buscador

**Pasada 1: elegir la variante con el modelo sin entrenar (Kaggle v22).**

- Configuración: 50 casos de práctica, normas originales, Kaggle T4 x2, commit `4e3613a`.
- Modelo: Gemma 4 E4B sin entrenar, fp16, `MODELO_LOTE=4`, prompt `baseline` de `AgenteJSON`.

```text
BUSCADOR_VARIANTE=<variante> MODELO_LOTE=4 python infra/kaggle_baseline.py --variante original --cantidad 50 --salida <carpeta nueva>
python -m admision.correr puntuar --esperadas practica/esperadas-original.jsonl ...
```

| Métrica | `baseline` | `prefijos` |
|---|---:|---:|
| Casos resueltos | **10/50** | 6/50 |
| Parcial medio | **0,831** | 0,796 |
| registro / box / circuito / plazo | 0,76 / 0,72 / 0,98 / 0,92 | 0,62 / 0,68 / 0,94 / 0,84 |
| cobertura / entidad / aviso / verificación | 0,80 / 0,82 / 0,72 / 0,76 | 0,78 / 0,82 / 0,72 / 0,76 |
| Llamadas / búsquedas por caso | 4,9 / 1,34 | 5,0 / 1,40 |
| Caracteres de contexto por caso | 19.954 | 20.689 |
| Segundos por caso (T4, lote 4) | 15,2 | 15,7 |

- **No es concluyente:** `prefijos` gana 3 casos y pierde 7; con 10 casos discordantes, la prueba de signos da
  p≈0,34. Tampoco hay señal de mejora.
- **Por qué el mejor recall no ayudó.** `baseline` devuelve siempre 2 secciones. `prefijos` devolvió 1 sección en 30
  búsquedas, 3 en 26 y 2 en 14.
  - Con 3 secciones suele sumar secciones de otras áreas que no deciden la ficha.
  - El contexto por caso subió 3,7 %.
  - Es lo que muestran las trazas, no una causa probada.
- **Costo de GPU:** la comparación usó 2.968 s de sesión T4.

**Pasada 2: con el prompt elegido (Kaggle v26, dev de 40 casos).** `procedimiento-v1` con `prefijos` cae a 4/40,
contra 9/40 con `baseline` (Parte 3).

**Decisión.** El default sigue siendo `baseline` por tres motivos:

- Es el que mejor resolvió con el agente.
- Generó las conversaciones de entrenamiento.
- El adapter aprendió a usarlo.

`prefijos` queda seleccionable como resultado negativo: mejora la evaluación separada, pero no al agente.

**Con el modelo de la entrega** (`ambos`, Kaggle v40,
[`evaluaciones/practica/tabla-v39`](evaluaciones/practica/tabla-v39/tabla.md)):

- El agente hace 3,16–3,18 búsquedas por caso.
- Usa 52.089–52.449 caracteres de contexto por caso.
- `circuito` y `plazo_circuito_min` dan 1,00 en las tres versiones de normas, y `cobertura` 0,90–0,94.

**Límites del buscador con el agente:**

- La comparación entre variantes no se repitió con el modelo entrenado.
- Hubo una sola corrida por variante, todo en T4.
- Los artefactos de la pasada 1 (`corridas/kaggle/v22/`, `corridas/rag-*-20261110/`) no están versionados.

---

## Parte 3: prompt

Con Gemma 4 E4B **sin entrenar** se optimizaron el mensaje de sistema y, en una pasada aparte, el formato de los
mensajes. Detalle, comandos y corridas en [optimización de prompt](docs/optimizacion-prompt.md).

**Cómo se eligió sin tocar práctica ni test.**

- Los candidatos se diseñaron desde el perfil agregado de fallas del modelo sin entrenar: busca una vez, decide de
  memoria y nunca aplica la costumbre del voluntariado.
- Se eligieron en un conjunto de desarrollo propio, `datos/prompt-dev/v1`:
  - 40 casos nuevos del generador (semilla 20261110).
  - Sin pacientes de práctica ni de entrenamiento.
  - Dos carpetas de normas con valores distintos a los originales.
- El criterio se declaró antes de correr: más casos resueltos y, a igualdad, menos tokens.
- Se hizo búsqueda manual estructurada en lugar de DSPy/GEPA, por costo de GPU.

**Fase 1: mensaje de sistema (Kaggle v25–v26, dev).**

- Configuración: T4 x2, fp16, lote 4, buscador `baseline`, commit `f4d7300`.
- Tokens medidos con el tokenizer real de Gemma, con plantilla de chat y sin razonamiento.

Cada prompt corrió en Kaggle una vez por carpeta del dev (`n1`, `n2`; la celda completa está en
[optimización de prompt](docs/optimizacion-prompt.md)):

```bash
python -m scripts.evaluar_prompts --prompt configuraciones/prompts/<prompt>.json \
  --configuracion configuraciones/evaluacion-prompt-kaggle-t4-l4.json \
  --casos datos/prompt-dev/v1/entradas/<n1|n2>/casos.jsonl \
  --esperadas datos/prompt-dev/v1/esperadas/esperadas-<n1|n2>.jsonl \
  --normas datos/prompt-dev/v1/entradas/<n1|n2>/normas --salida <carpeta nueva>/<prompt>-<n1|n2>
```

El ranking local:

```bash
uv run --locked --no-sync python -m scripts.rankear_prompts evaluaciones/dev/v25-fase1/<prompt>-n{1,2}
```

| Prompt (mensaje de sistema) | Tokens | Dev (40 casos) | Contexto/caso en dev (caracteres) |
|---|---:|---:|---:|
| `baseline` (cátedra) | 615 | 8/40 | ~19.000 |
| A `corto-v1`: mismas reglas, descripciones compactas | 453 | 1/40 | ~14.000 |
| **B `procedimiento-v1`: + procedimiento (buscar cada tema por separado, verificar antes de decidir)** | **857** | **9/40** | ~53.000 |
| C: B + las dos costumbres del historial | 969 | no comparable (falla CUDA en T4) | ~53.000 |
| D: C compactado | 659 | 2/40 | ~22.000 |

- **Acortar empeora.** Sin las descripciones de las herramientas el modelo busca menos (0,95 veces por caso con A).
- **C no termina.** Falla con `CUDA error: unspecified launch failure` en 5 de 7 intentos, siempre en el mismo punto.
  No se puede elegir un prompt que no termina en el hardware de evaluación.
- **La ventaja de B es débil.** Gana +1 caso en el dev, dentro del ruido (prueba de signos p=1,0), pero es el único
  candidato que no empeora.
- **Con `prefijos`, B baja a 4/40** (v26).

**Fase 2: antes y después en práctica (Kaggle v25 y v27; solo para reportar, no para elegir).**

- Punto de control: modelo base sin entrenar, `google/gemma-4-E4B-it@ee0ef60`.
- Prompts: `baseline` (sha256 `944ae7f6…`) y `procedimiento-v1` (sha256 `8c692c2c…`).

```text
python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/<actual|solo-prompt>.json   # Kaggle, commits f4d7300 y a5745f0
python -m scripts.tabla_evaluacion --evaluacion evaluaciones/practica/v25-sin-entrenar --evaluacion evaluaciones/practica/v27-solo-prompt ...
```

| Normas | Sin optimizar (`baseline`, 615 tokens) | Optimizado (B, 857 tokens) | Referencia cátedra sin entrenar |
|---|---:|---:|---:|
| original | 10/50 | **14/50** | 12/50 |
| b | 10/50 | **16/50** | 15/50 |
| c | 7/50 | **12/50** | 12/50 |
| total | 27/150 | **42/150** | 39/150 |

**Qué cambió y a qué costo.**

- **Más búsquedas:** el procedimiento hace que el modelo busque ~3,5 veces por caso en vez de ~1,3, y las fallas
  atribuibles al buscador desaparecen.
- **Controles que mejoran:** `box_antes_del_registro` (0,72 → 0,90–0,96), `plazo` (0,92 → 0,96–1,00) y `registro`.
- **Controles que empeoran:** con la evidencia delante, el modelo sin entrenar decide peor la verificación
  (0,74–0,76 → 0,60–0,66) y la entidad a notificar.
- **Aviso al voluntariado:** queda en 0,72, porque B no menciona la costumbre.
- **Costo:** el prompt es 39 % más largo; el contexto por caso es ~2,6× y el tiempo por caso ~1,6× (Parte 5).
- **Caso por caso:** 23 ganados y 8 perdidos sumando versiones, que comparten pacientes.

**Fase 3: formato de los mensajes (Kaggle v37, commit `7b92583`, dev, sin adapter).**

- `procedimiento-formato-v1` (sha `47b6a23b…`) deja el mensaje de sistema de B y cambia la observación y el aviso de
  JSON inválido.
- Resultado: 10/40 contra 9/40 de B (2 ganados, 1 perdido), con −2 % de contexto y −8 % de tiempo.
- **No se adoptó:**
  - La diferencia está dentro del ruido.
  - El ahorro máximo del formato es 1,4 % del contexto.
  - Adoptarlo exigía regenerar las conversaciones y reentrenar.

```bash
uv run --locked --no-sync python -m scripts.rankear_prompts \
  evaluaciones/dev/v25-fase1/procedimiento-v1-n{1,2} evaluaciones/dev/v37-formato/procedimiento-formato-v1-n{1,2}
```

**Decisión.** Desde 2026-11-12, `entrega.py` carga `procedimiento-v1`, el prompt con el que se entrenó el adapter.

---

## Parte 4: entrenamiento

### Estrategia

1. **SFT con LoRA sobre las conversaciones curadas con el prompt optimizado** (`v3-procedimiento`, 2.900).
   - La consigna pide entrenar con el prompt optimizado.
   - La pérdida se calcula **solo sobre las respuestas del asistente**: el contexto y el padding van con
     `labels=-100`, y no hay recortes.
   - LoRA r=8, alpha 16, dropout 0, solo en q/v de la atención del modelo de lenguaje. El modelo se carga sin las
     torres de visión y audio.
   - `configuraciones/sft-ejemplo.json`: 1 época, lr 2e-4, semilla 42, `batch_size=1`, acumulación 8,
     `max_seq_length` 8192.
   - Kaggle 2× T4, fp16, `device_map=auto`.
   - Los logits se calculan solo en las posiciones supervisadas (`logits_to_keep`). La pérdida es idéntica a la
     completa y la memoria entra en T4.
2. **Diagnóstico con el dev:** el adapter nunca respondía `true` en el aviso al voluntariado.
3. **Seguir entrenando el mismo adapter** (`adapter_inicial`) con subconjuntos dirigidos a esa falla, en vez de
   reentrenar (~9,5 h, casi toda la cuota). Tampoco se escribió la regla en el prompt: la consigna pide que las
   costumbres las enseñe el entrenamiento. El adapter de la entrega es el de la v39.
4. **Selección solo con el dev** (`datos/prompt-dev/v1`, 40 casos), con barras declaradas antes de cada corrida. La
   práctica se corrió una vez al final, para reportar.

Los hiperparámetros son una configuración inicial, **no** el resultado de un barrido. No hubo DPO ni refuerzo
(Limitaciones). Contratos y comandos en la [guía del entrenador](docs/entrenamiento-sft.md).

### Tabla de la Parte 4 (práctica, 50 casos × normas original, b y c)

```text
python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/<actual|solo-prompt|entrenado-v39>.json --salida <carpeta>   # Kaggle v25, v27, v40
uv run python -m scripts.tabla_evaluacion --evaluacion evaluaciones/practica/v25-sin-entrenar \
    --evaluacion evaluaciones/practica/v27-solo-prompt --evaluacion evaluaciones/practica/v40-entrenado-v39 \
    --salida <carpeta nueva>   # la vigente es evaluaciones/practica/tabla-v39
```

| Configuración | original | b | c | Total | Punto de control | Prompt | Corrida |
|---|---:|---:|---:|---:|---|---|---|
| sin entrenar | 10/50 | 10/50 | 7/50 | 27/150 | `gemma-4-E4B-it@ee0ef60` | `baseline` (`944ae7f6`) | v25, `f4d7300` |
| solo prompt | 14/50 | 16/50 | 12/50 | 42/150 | `gemma-4-E4B-it@ee0ef60` | `procedimiento-v1` (`8c692c2c`) | v27, `a5745f0` |
| solo entrenamiento | 17/50 | 14/50 | 20/50 | 51/150 | `tp-admision@770c9fac/final` | `baseline` (`944ae7f6`) | v40, `1405aab` |
| **ambos (la entrega)** | **36/50** | **28/50** | **36/50** | **100/150** | `tp-admision@770c9fac/final` | `procedimiento-v1` (`8c692c2c`) | v40, `1405aab` |
| *Cátedra: sin entrenar* | 12/50 | 15/50 | 12/50 | 39/150 | (cátedra) | (cátedra) | `mission.md` |
| *Cátedra: SFT simple con datos perfectos* | 25/50 | 24/50 | 25/50 | 74/150 | (cátedra) | (cátedra) | `mission.md` |

Todas las filas corrieron en Kaggle 2× T4, fp16, lote 4, con buscador `baseline`. La procedencia por celda está en
[`tabla-v39/tabla.md`](evaluaciones/practica/tabla-v39/tabla.md): commit, trazas con sha256, comando de jugar y de
puntuar.

**Lectura.**

- **Contra la cátedra:** `ambos` supera la referencia SFT en las tres versiones de normas (+11 / +4 / +11). Las
  normas b y c cambian los valores, y la ventaja se mantiene: el modelo aplica lo que trae el buscador en vez de
  memorizar los valores originales.
- **Controles de `ambos`:** quedan entre 0,90 y 1,00, salvo `aviso_al_voluntariado` (0,82 / 0,70 / 0,78). El aviso
  sigue siendo la falla principal.
- **Prompt y adapter juntos:** el adapter rinde con el prompt con el que se entrenó. `solo entrenamiento` (51/150)
  usa `baseline`, que el adapter nunca vio, y tiene 3 / 3 / 1 casos sin decisión. Esa fila puede subestimar al
  adapter.
- **Contra el adapter anterior (`dc41674a`, v34):** `ambos` pasó de 30/29/33 = 92/150 a 36/28/36.
  - Ganó `pev-008`, `pev-025`, `pev-031`, `pev-042` y `pev-043` en las tres versiones.
  - Perdió `pev-007`, `pev-021`, `pev-022` y `pev-045` en las tres.
  - **En la versión b perdió un caso neto** (29 → 28).
- **Una sola corrida por versión:** no hay intervalos de confianza.
- **Tabla histórica:** la del adapter `dc41674a` (10/10/7 · 14/16/12 · 11/13/12 · 30/29/33) se conserva en
  [`evaluaciones/practica/tabla`](evaluaciones/practica/tabla/tabla.md).

### Pasadas de entrenamiento, incluidas las que no funcionaron

Todas corrieron en Kaggle 2× T4 fp16. Los recibos están en [`evaluaciones/entrenamiento/`](evaluaciones/README.md)
(`runner.json`, `runner.log`, `estado.json`). Cada checkpoint está en HF con su revisión y cada corrida en W&B.

| Kaggle | Qué | Resultado | Decisión |
|---|---|---|---|
| v4–v7 | Humo: 8 pasos con datos sintéticos cortos, checkpoints en HF y reanudación desde `checkpoint-2` | Pipeline validado en GPU (v6 y v7). Fallaron antes v4 (un `ValueError` que el CLI no mostraba) y v5 (LoRA alcanzaba las torres de visión y audio, que PEFT no soporta) | Infraestructura lista; no mide calidad |
| v28 | Primer SFT real, `v3-procedimiento`, commit `8843004` | **OOM en el paso 0**: logits completos en fp32 sobre 262k de vocabulario (`ForCausalLMLoss`) | Arreglo: pérdida solo en posiciones supervisadas |
| v29, v31 | Ídem con la pérdida nueva | Error de dispositivo (`cuda:1`): accelerate mueve la máscara a la primera GPU | Arreglo: posiciones como lista de `int` |
| v30 | Ídem | Timeout de W&B en `wandb.init` | `WANDB_INIT_TIMEOUT=300` |
| v32 + v33 | SFT completo, commit `6169241`, W&B `3add2742ddfd4b42` | v32 cortada por el límite de 8 h en el paso 315/363; v33 reanudó desde checkpoint-300 y terminó. `final` = `dc41674a` | Primer adapter de la entrega |
| v34 | Práctica de `dc41674a` (plan `entrenado.json`, `3adf12e`) | solo entrenamiento 11/13/12; ambos 30/29/33 | Solo reporte |
| v35 | Dev, `dc41674a` + B (`bea206b`) | **26/40**; el aviso nunca da `true`: los 8 mayores sin acompañante fallan solo por eso | Hipótesis confirmada en el dev |
| v36 | Continuación sobre `v4-aviso` (302/633 = 47,7 % positivos), lr 1e-4, `fb353c0`, W&B `6d8edb168beb4e9b`, `final` `22ee5596` | **23/40**: 6/8 positivos, pero 12 falsos `true` | **No adoptada** |
| v38 | Continuación sobre `v5-aviso-peso` con `peso_aviso` 20, `984fc18`, W&B `f9326b10342b4be4`, `final` `17e9fe3d` | **26/40**: 1/8 positivos, 3 falsos `true` y nuevas fallas en cobertura, registro y box | **No adoptada** (la barra era superar 26) |
| **v39** | Continuación sobre `v6-aviso-datos` (prefijo `Edad: N años. Acompañante: X.`, `peso_aviso` 20), `5a36ecf`, W&B `41ec01f292e64306`, `final` **`770c9fac`** | **31/40**: 5/8 positivos, 2 falsos `true`; ganó 7 casos y perdió 2 | **Adoptada** (barra ≥30/40) |
| v40 | Práctica de `770c9fac` (plan `entrenado-v39.json`, `1405aab`) | solo entrenamiento 17/14/20; ambos 36/28/36 | Solo reporte; no cambió la decisión |

El análisis de cada pasada está en el [análisis de errores](docs/analisis-errores.md), pasadas 7 a 10.

**Por qué funcionó v39 y no v36 ni v38.**

- El aviso es **un solo token** entre los ~350 supervisados de cada conversación (1.015.072 / 2.900 en el
  `manifiesto-tokens.json` de `v3-procedimiento`), y la pérdida media lo diluye.
- Subir la proporción de positivos (v36) movió el sesgo hacia `true` sin enseñar la condición.
- Multiplicar su pérdida (v38) casi no lo movió.
- Hacer que el modelo **copie la edad y el acompañante antes de decidir** (v39) sí movió la decisión: de 0/8 a 5/8
  positivos en el dev.
  - El prefijo apareció en 38 de 40 mensajes finales.
  - `extraer_json` de la cátedra lo lee sin cambios.

**Límite declarado.** La falla del aviso se **vio** primero en un análisis de práctica (pasada 7). Para no usar la
práctica como selector, la hipótesis se confirmó en el dev, los remedios se armaron solo con datos de entrenamiento
y la adopción se decidió solo con el dev. Saber que la falla existía vino de mirar práctica: lo declaramos en vez de
ocultarlo.

---

## Parte 5: recursos

Todo se midió en **Kaggle 2× T4 (15.360 MiB cada una), fp16, `device_map=auto`**.

- **No hay mediciones en A100-40GB.** Los tiempos de T4 no se transfieren a A100 y acá no se inventan factores.
- La VRAM es el máximo por GPU sumado entre las dos GPU. Es una cota superior, en GB decimales.

El borrador completo, con el origen de cada número, está en la [Parte 5 detallada](docs/informe-parte5.md).

### 5.1 Entrenamiento: tiempo, VRAM y casos a la vez

| Corrida | Datos | Pasos | Tiempo | VRAM pico (suma de 2 T4) | Fuente |
|---|---|---:|---|---|---|
| SFT base (v32 + v33) | `v3-procedimiento`, 2.900 conv., ~10 M tokens | 363 | ~80–85 s/paso, **~8,1 h de cómputo** (~9,5 h de sesión en 2 tandas) | **19,97 GB** (12,1 + 7,87; reservada 20,58) | `evaluaciones/entrenamiento/v32-v33-sft-v3/` |
| v36 (no adoptada) | `v4-aviso`, 633 | 80 | 7.029 s (~88 s/paso) | 20,29 GB | `evaluaciones/entrenamiento/v36-aviso/` |
| v38 (no adoptada) | `v5-aviso-peso`, 1.133 | 142 | 11.750 s (~83 s/paso) | 20,29 GB (12,30 + 7,99) | `evaluaciones/entrenamiento/v38-peso/` |
| **v39 (entrega)** | `v6-aviso-datos`, 1.133 | 142 | **11.735 s (3,26 h, ~83 s/paso)** | **20,36 GB** (12,37 + 7,99; reservada 20,96) | `evaluaciones/entrenamiento/v39-datos/` |

**Origen de cada fila.** Todas corrieron con el runner de Kaggle, después de `scripts.preparar_entrenamiento` y
`scripts.tokenizar_entrenamiento` con la configuración de la fila:

```text
python infra/kaggle_entrenamiento.py --preparacion <preparada> --tokens <tokens> \
  --salida /kaggle/working/corridas/<salida> --guardar-cada <N>
```

| Corrida | Configuración (igual a `configuracion` de `runner.json`) | `--salida` | `--guardar-cada` | Checkpoint de partida → `final` |
|---|---|---|---:|---|
| v32 | `configuraciones/sft-ejemplo.json` | `sft-v3-1` | 50 | nuevo; cortada en el paso 315 |
| v33 | `configuraciones/sft-ejemplo.json` | `sft-v3-2` | 50 | `--reanudar-checkpoint checkpoint-300 --reanudar-revision 83dd1d20…` → `dc41674a` |
| v36 | `configuraciones/sft-aviso.json` | `sft-aviso` | 20 | `adapter_inicial` `dc41674a` → `22ee5596` |
| v38 | `configuraciones/sft-aviso-peso.json` | `sft-peso` | 20 | `adapter_inicial` `dc41674a` → `17e9fe3d` |
| v39 | `configuraciones/sft-aviso-datos.json` | `sft-datos` | 20 | `adapter_inicial` `dc41674a` → `770c9fac` |

Precisión y dispositivos son los defaults del runner (`fp16`, `auto`). La salida y `guardar_cada` salen de
`runner.log` y `runner.json`. Las rutas exactas de `--preparacion` y `--tokens` **no quedaron registradas**.

- **Casos a la vez al entrenar:** 1 conversación por pasada hacia adelante (`batch_size=1`) y 8 por paso de
  optimización (acumulación 8).
- **Costo total del adapter de la entrega:** SFT base más continuación v39, ~11,4 h de cómputo en T4.
- **Pérdida:** la del SFT bajó de 0,44 a 0,0044 de media final. Una pérdida baja no implicó un agente bueno: el aviso
  no se había aprendido. La pérdida de v38 y v39 es ponderada y no se compara con la de v36.
- **El humo** (2,18 s/paso con secuencias cortas) **no** servía para extrapolar: el SFT real tardó ~80 s/paso.

### 5.2 Inferencia: tiempo por caso, VRAM y contexto

Práctica, 50 casos × 3 versiones, lote 4.

- **s/caso:** `jugar_lote / casos` de [`tabla-v39`](evaluaciones/practica/tabla-v39/tabla.md), sin cargar el modelo.
- **Tokens, VRAM y ancho:** cálculo propio sobre `metricas.jsonl` de cada corrida versionada.
  - Tokens por caso = suma sobre los micro-lotes / 50.
  - Ancho = máximo de `ancho_entrada`.

| Configuración (corrida) | s/caso | Tokens de entrada por caso | Tokens generados por caso | VRAM pico | Ancho máx. (tokens) |
|---|---:|---:|---:|---:|---:|
| sin entrenar (v25) | 15,4–16,1 | 5.692–5.753 | 270–271 | 18,12 GB | 3.072 |
| solo prompt (v27) | 24,4–26,6 | 13.655–14.636 | 322–332 | 19,25–19,64 GB | 4.436 |
| solo entrenamiento, v39 (v40) | 27,7–30,2 | 12.146–14.204 | 333–338 | 20,25–21,71 GB | 7.636 |
| **ambos, v39 = entrega (v40)** | **27,8–28,8** | **14.101–14.196** | **353–355** | **19,94 GB** (11,59 + 8,35) | **4.395** |

- **El costo lo pone el prompt, no el adapter.** El adapter se fusiona al cargar y no agrega capas.
  - `procedimiento-v1` hace ~3,2–3,6 búsquedas por caso contra ~1,3, y eso da ~2,5× tokens de entrada.
  - La entrega tarda ~+80 % por caso frente a sin entrenar (15,4–16,1 s), y a cambio resuelve 100/150 contra 27/150.
- **Contra el adapter anterior:** la entrega tarda más que con `dc41674a` (24,7–24,9 s/caso). El mensaje final lleva
  la edad y el acompañante, y hay ~6,7 llamadas por caso.
- **Sin recortes:** la entrada más ancha (7.636 tokens, `solo entrenamiento` en b) queda por debajo del tope de
  8.192. La entrega llegó a 4.395.
- **Carga del modelo** (`construir_agente` en `evaluaciones/practica/*/salida/tiempos.json`). Es un costo fijo por
  proceso y no separa descarga de carga.
  - Modelo base: 20,5–21,6 s (v25) y 39,4–43,2 s (v27, normas b y c).
  - Base + adapter: 22,6–48,8 s (v34 con `dc41674a` y v40 con el v39).
  - La primera corrida de algunas sesiones tardó mucho más: 147,9 s (v27, base) y 141,8–155,0 s (v34 y v40, con
    adapter).

### 5.3 Casos que se procesan a la vez al evaluar

El lote se barrió con el modelo sin entrenar, normas originales y 50 casos (sección 2 de la
[Parte 5 detallada](docs/informe-parte5.md#2-casos-a-la-vez-al-evaluar-lote)).

| Lote | Estado | s/caso | VRAM pico |
|---|---|---:|---:|
| 1 | ok, 10/50 | 33,3 | 16,7 GB |
| **4 (default de la entrega)** | ok, 10/50 | **15,6** | 19,1 GB |
| 8 | **cayó** (`CUDA error: unspecified launch failure`) | inválido | 20,2 GB |
| 16 | **cayó** (mismo error) | inválido | 22,7 GB |

Comando (bitácora, 2026-11-04): `for l in 1 8 16; do MODELO_LOTE=$l python infra/kaggle_baseline.py --variante
original --cantidad 50 --salida ...; done` (commit `11bf3d6`, salidas `lote-original-l{1,8,16}`); el lote 4 es `base-original-l4`
(commit `a89f5b9`, mismo runner con `MODELO_LOTE=4`). Puntuado con `admision.correr puntuar` contra
`practica/esperadas-original.jsonl`. Los artefactos están versionados en `evaluaciones/recursos/lote-original-l{1,8,16}`,
`base-original-l4` y `diag-l8` (lote 8 con `CUDA_LAUNCH_BLOCKING=1`).

**Tres tiempos distintos para el modelo sin entrenar con lote 4** (15,2 s en v22, 15,6 s en `a89f5b9` y 15,4–16,1 s
en v25) son corridas distintas: otra sesión de Kaggle, otro commit y, en v22, la comparación de buscadores. No hay
contradicción, sino la variación entre sesiones de T4.

- **Lote 1 contra lote 4:** el lote 4 es 2,1× más rápido. Resolvió los mismos 10 casos y dio la conversación idéntica
  en 49 de 50.
- **Latencia:** cada micro-lote pasa de 6,8 a 12,0 s, así que la latencia de cada llamada casi se duplica.
- **Lote 8 con `CUDA_LAUNCH_BLOCKING=1`:** completó (10/50, 17,4 s/caso), pero es más lento que el lote 4. La causa
  no se identificó.
- **Default:** la entrega usa `LOTE_ENTREGA = 4`, porque así se midió todo.
- **PENDIENTE:** barrer 1/4/8/16 en A100-40GB y fijar el lote con esa evidencia.

### 5.4 Costo

**Desarrollo: medido en Kaggle, 0 USD.**

- La cuota es de ~30 h/semana de GPU.
- Según la API de Kaggle, el 2026-11-12 la semana del SFT llevaba 48.546 s (13,5 h), incluidos los intentos
  fallidos v28–v31.
- Después se sumaron estas corridas, que no entran en esa cifra:
  - v36: 2 h de runner más ~0,4 h de dev.
  - v37: ~20 min.
  - v38: 3,3 h más el dev.
  - v39: 3,3 h más ~0,4 h de dev.
  - v40: ~2,5 h.
- **El total real de cuota usado es mayor y no está auditado.**

**A100-40GB: ilustrativo, no medido.**

- La consigna da ~2,10 USD/h. Modal se descartó el 2026-11-05 porque habilitar su crédito exige cargar una tarjeta,
  y no hay otra vía de A100.
- Fórmula: `costo por caso = (s/caso / 3600) × 2,10 + carga amortizada`.
- Solo como orden de magnitud, con tiempos **de T4**: la entrega a 28,8 s/caso daría 0,0168 USD por caso y 0,84 USD
  por 50 casos. El adapter de la entrega (~9,5 h de sesión del SFT + 3,26 h de v39) daría ~27 USD.
- **PENDIENTE (A100):** s/caso, carga, costo por caso y costo del entrenamiento medidos en A100-40GB.

### 5.5 Producción: simultáneos, proyección y escalado (hipótesis)

**Rendimiento medido en T4.** Un proceso de la entrega con lote 4 procesa ~125–129 casos/h (3.600 / 28,8–27,8 s),
con 19,94 GB de VRAM pico. En el kit, «casos a la vez» es el tamaño del lote: los casos entran y salen por tandas,
sin cola ni batching continuo.

**Cuántos simultáneos tolera: PENDIENTE (A100).**

- En T4 la VRAM pico crece de 16,7 a 22,7 GB al pasar de lote 1 a 16. Eso *sugiere* margen en 40 GB.
- La topología cambia: 2 GPU de 15 GB contra 1 GPU de 40 GB. Tampoco medimos la KV cache por conversación ni el techo
  de lote.
- Hace falta barrer 1/4/8/16/32 en A100 con `MODELO_METRICAS` antes de fijar un número.

**Proyección de consumo.** Usamos un modelo de variables, no una predicción:

- Capacidad de una instancia: `Q = 3600 × N / T_lote` casos/h.
- Réplicas para una demanda `λ` y una utilización objetivo `ρ`: `réplicas = ceil(λ / (ρ × Q))`.
- Costo mensual: `C_mes = r × H × p + C_otros`.
- Costo por caso: `C_caso = C_mes / V`.

Plataforma, tarifa, demanda, utilización y SLO no están medidos, así que **no se publica un presupuesto de
producción**.

**Cómo escalaríamos** (diseño, no implementado):

1. **Réplicas:** una A100-40GB por contenedor, con el modelo y el adapter cargados una vez. Conviene tener
   contenedores tibios por el arranque en frío de 20,5–155,0 s medido en T4 (sección 5.2).
2. **Servidor con batching continuo (vLLM):** es la mayor ganancia esperable, pero no se probó. Falta verificar
   el soporte de Gemma 4 y de LoRA.
3. **Menos tokens por caso:** en nuestras corridas con el prompt `baseline` de la cátedra, el mensaje de sistema
   ocupa ~35 % de los caracteres de la conversación mediana (2.232 de 6.409). Como se repite en cada llamada,
   recortarlo baja el cómputo y la VRAM. La consigna habla de «la quinta parte» porque mide sobre las
   conversaciones de referencia de la cátedra, no sobre las nuestras.

---

## Limitaciones

- **A100-40GB sin validar** ([KI-A100-001](README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb)).
  - El entrenamiento, la evaluación y los recursos se midieron solo en Kaggle 2× T4 fp16.
  - No hay prueba de que la entrega cargue y corra en una A100 con `cuda:0` y bf16, ni con lote 4.
  - Es una limitación declarada, no un número estimado.
- **Una sola corrida por configuración.** No hay intervalos. En el dev, diferencias de ±1–3 casos están dentro del
  ruido.
- **No se hizo barrido de hiperparámetros** (lr, r de LoRA, épocas). La configuración del SFT es la inicial.
- **No se probaron DPO ni refuerzo.** Quedan como trabajo futuro (issue #15, cerrado).
- **El aviso al voluntariado sigue fallando:** 0,70–0,82 en práctica, y en el dev quedan 3/8 positivos sin atrapar y
  2 falsos `true`.
- **La falla del aviso se descubrió mirando práctica** (Parte 4, límite declarado).
- **El buscador no se recomparó con el modelo entrenado.**
- **Acceso de la cátedra:** el repo HF y el proyecto W&B son privados. Hace falta invitar a la cátedra (#16).

## Trabajos futuros

| Idea | Beneficio esperado | Estado |
|---|---|---|
| Medir todo en A100-40GB: s/caso, VRAM, lote máximo estable, costo | Los números finales de la entrega | Sin vía de acceso; Modal descartado |
| DPO con pares que difieran solo en el aviso | Enseñar la condición edad **y** sin acompañante | Sin probar |
| Barrido de hiperparámetros (lr, r, épocas, módulos LoRA) | Más casos resueltos y menos fallas fuera del aviso | Sin probar |
| Servidor con batching continuo (vLLM) | Más rendimiento y mejor latencia en producción | Sin probar |
| Corregir el fallo CUDA en lotes ≥ 8 | Más casos a la vez | Lote 8 con bloqueo completa, pero más lento; causa pendiente |
| Recortar el mensaje de sistema y el contexto de búsquedas | Menos cómputo y VRAM por llamada | Sin probar; en nuestras corridas con el prompt `baseline`, el mensaje de sistema es ~35 % de los caracteres de la conversación mediana |
| Sinónimos en el buscador | Resolver d030 y d023 | Sin probar |
| Repetir mediciones | Intervalos en vez de una sola corrida | Sin hacer |
