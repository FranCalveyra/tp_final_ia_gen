# Informe: costumbres del hospital

## Resumen ejecutivo

El análisis amplía el trabajo del issue #10 con `historial/historial.csv`, las 2.000 trazas correspondientes de `historial/trazas.jsonl` y las normas de `normas/original/`. Se confirmaron dos costumbres y se investigó un patrón adicional de operadores sin convertirlo en una regla.

La clasificación compatible con el issue #10 se conserva:

| tipo | cantidad |
|---|---:|
| correcta | 1.631 |
| error | 279 |
| ruido | 90 |
| total | 2.000 |

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
2. Se calculan las violaciones normativas fila por fila, incluyendo registro, documento, orden del box, circuito, plazo, cobertura, verificación y entidad a notificar.
3. Se excluyen de la inferencia de costumbres todas las filas con alguna violación.
4. Para la costumbre etaria se comparan situaciones equivalentes por `motivo × cobertura`, con al menos 20 filas limpias menores y 20 mayores en cada estrato.
5. Sólo se confirma una señal cuando tiene muestra suficiente, una frecuencia claramente dominante y una explicación compatible con una práctica no escrita. Una minoría aislada no se convierte automáticamente en costumbre.
6. Se usan las trazas para validar la correspondencia ID-operador-ficha y para investigar variaciones por operador.

El análisis completo, con frecuencias e IDs, queda en `historial_costumbres.json`. `costumbres.md` es la versión legible para la presentación y para compartir con Martu y Eze.

## Costumbres confirmadas

### 1. No avisar al voluntariado a menores de 75 años

El umbral observado es 75 años. En filas limpias menores de 75, el aviso es `no` en 1.298 de 1.415 casos (91,7%). En filas limpias de 75 años o más, el aviso es `sí` en 175 de 306 (57,2%), por lo que el grupo mayor no se trata como una regla uniforme.

La comparación no mezcla motivos ni coberturas: se conservaron ocho estratos con muestras de ambos grupos. Los IDs de respaldo de los `no` y los IDs de las excepciones `sí` están en `costumbres.md` y en cada estrato de `historial_costumbres.json`.

Esta costumbre explica el ruido que ya clasificaba el issue #10, pero no reemplaza las normas ni justifica avisar o no avisar en cualquier situación sin considerar el contexto.

### 2. Priorizar la prepaga con dos credenciales

En 171 filas limpias que presentan dos credenciales, la credencial que figura como verificada es:

| selección observada | cantidad | frecuencia |
|---|---:|---:|
| prepaga | 162 | 94,7% |
| obra social | 9 | 5,3% |

La norma no indica que haya que elegir la primera ni que haya que preferir una clase. Por eso esta tendencia se conserva como costumbre observada: no se transforma en una norma inventada. Las 9 excepciones y los 162 IDs de respaldo están documentados.

## Patrón adicional investigado: operadores

Las trazas permiten estudiar si la costumbre etaria se concentra en personas concretas. En las filas normativamente limpias de menores de 75 años:

| operadores | avisos | muestra | frecuencia |
|---|---:|---:|---:|
| OP-04 | 28 | 60 | 46,7% |
| OP-11 | 49 | 82 | 59,8% |
| OP-17 | 40 | 83 | 48,2% |
| otros 17 operadores | 0 | 1.190 | 0,0% |

La señal es fuerte y reproducible, y `costumbres.md` conserva los 117 IDs involucrados. Sin embargo, no se confirma como una costumbre del hospital: está concentrada en un subgrupo de operadores y no representa la práctica mayoritaria. Se documenta como variación histórica para el prompt y los datos, pero no agrega una regla ni reclasifica automáticamente filas fuera de los estratos ya comparables.

No se encontraron otros patrones con evidencia equivalente en nivel, motivo, llegada, documento, capacidad de responder, lesión laboral, sospecha de sepsis o cobertura. Las diferencias observadas en esos campos no superan el criterio de confirmación o están explicadas por normas escritas.

## Compatibilidad con el issue #10

El clasificador conserva su interfaz y su precedencia. Con las trazas habilitadas, el CSV generado sigue teniendo exactamente 2.000 filas y los mismos conteos `1.631/279/90`. La validación adicional de trazas sólo rechaza datos inconsistentes; no usa una decisión de la traza para sobreescribir las reglas normativas.

La ejecución reproducible es:

```text
python scripts/clasificar_historial.py --entrada historial/historial.csv --normas normas/original --trazas historial/trazas.jsonl --salida historial_clasificado.csv
```

## Archivos y resumen para el equipo

- [`costumbres.md`](costumbres.md): costumbres confirmadas, patrón adicional e IDs.
- [`historial_costumbres.json`](historial_costumbres.json): frecuencias, estratos, operadores e IDs completos.
- [`docs/resumen-costumbres-issue-11.md`](docs/resumen-costumbres-issue-11.md): versión breve para Martu y Eze.
- [`scripts/clasificar_historial.py`](scripts/clasificar_historial.py): análisis reproducible y compatible.
- [`tests/test_clasificar_historial.py`](tests/test_clasificar_historial.py): pruebas de salida, trazas y determinismo.

## Verificación

```text
python -m unittest tests.test_clasificar_historial -v
```

## Issue #12: datos de entrenamiento

### Seleccion de historial

Se parte de las 2.000 filas clasificadas en el issue #10. Se excluyen las 279 filas `error` y las 90 `ruido`. Ademas, se excluyen 36 filas `correcta` con aviso a menores de 75 anos que quedaron fuera de los estratos comparables y estan concentradas en la variacion de `OP-04`, `OP-11` y `OP-17` documentada en el issue #11. Esto evita ensenar como respuesta esperada una conducta ambigua.

Quedan **1.595 conversaciones reales**. Cada una se obtiene de `historial/trazas.jsonl`, conserva el sistema, el mensaje inicial, todas las acciones y observaciones, y termina en `registrar_decision`. No se usan filas de `practica/` ni sus fichas esperadas.

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

Cada ficha sintetica se deduce desde los parametros cargados de `normas/original/`: documento, box, circuito, plazo, fuente de cobertura, afiliacion, digito reservado y notificacion. La seleccion de prepaga en dos credenciales y el aviso etario se toman como costumbres observadas del issue #11; no se fijan numeros de parametros en el prompt ni en los datos.

Las conversaciones se ejecutan sobre `admision.Entorno` con un buscador determinista de las normas originales. Las 500 fichas pasan la comparacion normativa y el validador de conversaciones v1. Los fundamentos de cada ficha se derivan segun el circuito y la situacion, en vez de usar una lista generica.

### Dataset final y reproducibilidad

El dataset combinado tiene **2.095 conversaciones**:

| origen | cantidad | proporcion |
|---|---:|---:|
| historial | 1.595 | 76,1% |
| sintetico | 500 | 23,9% |
| total | 2.095 | 100% |

El comando reproducible es:

```text
python -m scripts.preparar_datos_entrenamiento --raiz . --salida datos/entrenamiento --semilla 12012 --candidatos 5000
```

La salida se valida con:

```text
python -m scripts.validar_conversaciones datos/entrenamiento/conversaciones.jsonl
```

Resultado: 2.095 registros, 2.095 validos, sin errores. `datos/entrenamiento/manifiesto.json` conserva procedencia, semilla, balance, cantidades y SHA-256; `auditoria_sinteticos.jsonl` conserva la ficha deducida y la categoria de cada sintetico. El ensamblador no sobrescribe una salida existente.

### Archivos del issue

- [`datos/entrenamiento/conversaciones.jsonl`](datos/entrenamiento/conversaciones.jsonl): dataset combinado v1.
- [`datos/entrenamiento/auditoria_sinteticos.jsonl`](datos/entrenamiento/auditoria_sinteticos.jsonl): auditoria de fichas sinteticas.
- [`datos/entrenamiento/manifiesto.json`](datos/entrenamiento/manifiesto.json): procedencia, balance y hash.
- [`scripts/seleccionar_entrenamiento.py`](scripts/seleccionar_entrenamiento.py): curacion y conversion de trazas reales.
- [`scripts/generar_sinteticos_entrenamiento.py`](scripts/generar_sinteticos_entrenamiento.py): generacion, deduccion y balance sintetico.
- [`scripts/preparar_datos_entrenamiento.py`](scripts/preparar_datos_entrenamiento.py): ensamblado y validacion final.
- [`tests/test_seleccionar_entrenamiento.py`](tests/test_seleccionar_entrenamiento.py), [`tests/test_generar_sinteticos_entrenamiento.py`](tests/test_generar_sinteticos_entrenamiento.py) y [`tests/test_preparar_datos_entrenamiento.py`](tests/test_preparar_datos_entrenamiento.py): pruebas reproducibles.
La suite específica verifica cinco casos: 2.000 filas y conteos, reglas normativas, las dos costumbres, el patrón adicional por operador, validación de 2.000 trazas y reproducibilidad byte a byte.
