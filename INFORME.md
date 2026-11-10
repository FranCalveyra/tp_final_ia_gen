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
