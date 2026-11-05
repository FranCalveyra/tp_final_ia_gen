# Informe: clasificación del historial

## Resultado

La clasificación se genera con:

```text
python scripts/clasificar_historial.py --entrada historial/historial.csv --normas normas/original --salida historial_clasificado.csv
```

El resultado contiene exactamente 2.000 filas, una por cada `id` del historial, y solo las columnas `id,tipo`.

| tipo | cantidad |
|---|---:|
| correcta | 1.531 |
| error | 355 |
| ruido | 114 |
| total | 2.000 |

La precedencia es deliberada: una violación de una norma escrita se clasifica como `error`, aunque también se aparte de una costumbre. Solo las filas sin violaciones normativas pueden ser `ruido`.

## Reglas normativas verificables

El script carga los parámetros de `normas/original/parametros.yaml` y verifica que estén presentes las secciones usadas. Se pueden observar en las columnas del CSV:

- `NOR-ADM-01`, `NOR-ADM-02` y `NOR-ADM-03`: tipo de registro, documento presentado y nivel/edad para decidir si el box precede al registro.
- `NOR-COB-01` a `NOR-COB-04`: cobertura declarada o credencial, lesión laboral, entidad y número verificados, respuesta de la entidad, cobertura resultante y entidad notificada. También se aplica el dígito reservado de `P-COB-2`.
- `NOR-TDE-01` a `NOR-TDE-04`: circuito según motivo/sospecha de sepsis y plazo, descontando `P-TDE-5` cuando la llegada es en ambulancia.

Las razones normativas encontradas son conteos de evidencias y no suman 355 porque una fila puede tener más de una:

| campo observado en conflicto | filas |
|---|---:|
| entidad a notificar | 114 |
| plazo del circuito | 97 |
| verificación | 88 |
| cobertura | 78 |
| registro | 64 |
| box antes del registro | 61 |
| filas con al menos una violación | 355 |

No se marcaron como error reglas que el historial no permite comprobar: disponibilidad y tipo físico de boxes, tiempo hasta evaluación médica, retriage, camas y sectores de internación, aislamiento, red/destino de derivación, altas previstas y plazos de internación/derivación. Esos datos no tienen columnas observables en `historial/historial.csv`.

## Detección de costumbres

`aviso_al_voluntariado` no está fijado por las normas originales. Para no convertir una minoría aislada en ruido, se compararon frecuencias por edad y se buscó un salto local reproducible: al menos diez filas por edad, frecuencia de `sí` mayor al 50% en la edad candidata y menor al 20% en las tres edades anteriores, con muestras de al menos 100 filas a cada lado.

El único quiebre fuerte encontrado es **75 años**:

- menores de 75: `no` en 1.494 de 1.614 filas (92,6%);
- 75 o más: `sí` en 219 de 386 filas (56,7%).

Por eso se conserva como costumbre fuerte “no avisar al voluntariado a pacientes menores de 75 años”. Las 114 filas menores de 75 con `sí` y sin error normativo son `ruido`. El grupo de 75 o más es mixto y no se reclasifican sus minorías automáticamente.

El detalle completo de frecuencias e IDs queda en [`historial_costumbres.json`](historial_costumbres.json) y en [`costumbres.md`](costumbres.md), para reutilizarlo en el issue #11.

## Archivos generados

- [`historial_clasificado.csv`](historial_clasificado.csv): salida contractual `id,tipo`.
- [`costumbres.md`](costumbres.md): explicación legible e IDs de respaldo y ruido.
- [`historial_costumbres.json`](historial_costumbres.json): análisis estructurado para reutilización.
- [`scripts/clasificar_historial.py`](scripts/clasificar_historial.py): generador determinista.

## Verificación

Se ejecutó:

```text
python -m unittest tests.test_clasificar_historial -v
```

Resultado: 3 pruebas, todas pasan. Cubren el contrato de cantidad/columnas/IDs/tipos, casos representativos de reglas normativas, detección de la costumbre y reproducibilidad de la salida y del análisis JSON en dos ejecuciones separadas.

La verificación es local y determinista; no es una validación de calidad de un modelo ni requiere dependencias externas, GPU, build o servicios remotos.
