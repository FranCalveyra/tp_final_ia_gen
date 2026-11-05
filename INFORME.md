# Informe: clasificación del historial

## Resultado actualizado

Se genera con:

```text
python scripts/clasificar_historial.py --entrada historial/historial.csv --normas normas/original --salida historial_clasificado.csv
```

| tipo | cantidad |
|---|---:|
| correcta | 1.631 |
| error | 279 |
| ruido | 90 |
| total | 2.000 |

Una violación de una norma escrita siempre tiene prioridad sobre una costumbre: `error > ruido`.

## Reglas normativas verificables

El script lee los parámetros de `normas/original/parametros.yaml` y verifica las secciones observables:

- `NOR-ADM-01`, `NOR-ADM-02` y `NOR-ADM-03`: registro, documento, número de documento y box antes del registro. El umbral urgente usa `P-ADM-3`; la edad usa `P-ADM-2`.
- `NOR-COB-01` a `NOR-COB-04`: cobertura, lesión laboral, afiliación verificada, respuesta, dígito reservado `P-COB-2` y entidad a notificar.
- `NOR-TDE-01` a `NOR-TDE-04`: circuito y plazo, incluyendo el descuento `P-TDE-5` para ambulancia y las dos categorías de sepsis.

Cuando hay dos credenciales, la norma no dice que deba elegirse la primera. La auditoría usa como fuente la credencial que el propio historial registra como verificada. La selección observada se analiza aparte como costumbre, sin convertirla en una norma inventada.

Las razones normativas contabilizadas pueden solaparse:

| campo en conflicto | filas |
|---|---:|
| entidad a notificar | 48 |
| plazo del circuito | 97 |
| verificación | 0; las verificaciones de dos credenciales se resuelven por la credencial efectivamente verificada |
| cobertura | 12 |
| registro | 64 |
| box antes del registro | 61 |
| filas con al menos una violación | 279 |

No se clasifican reglas que el CSV no permite comprobar: disponibilidad y tipo físico de boxes, evaluación médica, retriage, camas, aislamiento y derivaciones.

## Costumbres y situaciones equivalentes

Las frecuencias se calculan después de excluir las filas con alguna violación normativa. Para la costumbre etaria se comparan situaciones con el mismo `motivo` y `cobertura`, con al menos 20 filas menores y 20 mayores en cada estrato.

Se detectaron ocho estratos comparables para la costumbre etaria. El umbral observado sigue siendo 75 años. En el conjunto limpio menor de 75, `no` domina; el grupo de 75 o más sigue siendo mixto y sus minorías no se marcan automáticamente.

También se detectó una costumbre en filas con dos credenciales: se selecciona `prepaga` en 162 de 171 filas limpias (94,7%). Las 9 selecciones limpias de obra social se clasifican como `ruido`; las filas con otros errores conservan `error`.

El JSON conserva los estratos, frecuencias e IDs completos para reutilizarlos en el issue #11. La clasificación no usa como evidencia de costumbre las filas que ya violan normas.

## Impacto respecto de la versión anterior

Se comparó la salida actual con la versión del commit `959838a`:

| transición | cantidad |
|---|---:|
| `error` -> `correcta` | 71 |
| `ruido` -> `correcta` | 34 |
| `error` -> `ruido` | 5 |
| `correcta` -> `ruido` | 5 |
| total de filas cambiadas | 115 |

La mayor corrección corresponde a filas con dos credenciales que antes se marcaban como error solo porque el clasificador asumía incorrectamente que siempre debía usarse la primera.

## Archivos

- [`historial_clasificado.csv`](historial_clasificado.csv)
- [`costumbres.md`](costumbres.md)
- [`historial_costumbres.json`](historial_costumbres.json)
- [`scripts/clasificar_historial.py`](scripts/clasificar_historial.py)
- [`tests/test_clasificar_historial.py`](tests/test_clasificar_historial.py)

## Verificación

```text
python -m unittest tests.test_clasificar_historial -v
```

Resultado actualizado: 4 pruebas, todas pasan. Se verifica cantidad, columnas, IDs, conteos, parámetros, credenciales múltiples, estratos de costumbre y reproducibilidad byte a byte en ejecuciones temporales separadas.
