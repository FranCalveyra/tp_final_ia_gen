# Contradicciones, faltantes y límites

Este registro conserva las inconsistencias sin resolverlas por conveniencia. Cuando dos fuentes difieren, se
mantienen ambas y se indica qué prueba faltaría para decidir.

## Identidad de las versiones

### La metadata no acredita `vN`

Los 41 `kernel-metadata.json` identifican el mismo kernel y archivo, pero no contienen un número de versión. La
asociación entre cada carpeta externa `vN` y la versión histórica N depende del mecanismo con el que se hizo la
extracción. Para resolverlo con independencia de esa procedencia harían falta manifiestos de descarga o una API de
Kaggle que incluyera explícitamente `versionNumber` por archivo.

Las fechas de modificación de los archivos externos son del 15/11/2026 y corresponden a la descarga, no a la
ejecución. Las fechas históricas de esta auditoría salen de runners, commits y documentación.

### Metadata casi idéntica

- v1–v10 y v17–v41 comparten el mismo hash de metadata, GPU T4 activada.
- v11, v12, v13, v15 y v16 comparten metadata con GPU desactivada.
- v14 también tiene GPU desactivada, pero usa otra imagen Docker.

Esto acredita configuración del kernel, no ejecución efectiva ni asignación real de hardware. El hardware real se
afirma solo cuando aparece en `runner.json` o en documentación basada en ese runner.

## Notebooks duplicados y ramas móviles

Hay cuatro grupos idénticos byte a byte:

- v4/v5/v6;
- v10/v17/v19;
- v12/v15/v16;
- v20/v21.

v4–v6 produjeron resultados diferentes porque el notebook clonaba una rama sin fijar commit. Los runners registran
`6f9abc0`, `5e1837e` y `acc95b2`. Por lo tanto:

- el hash del notebook prueba que la receta de lanzamiento era la misma;
- el runner prueba qué árbol Git usó cada ejecución;
- el notebook, por sí solo, no permite reproducirla.

v13→v14 y v17→v18→v19 cambian solamente el orden de claves JSON del notebook. v14 además cambia la imagen del
kernel. No se considera un cambio de código.

## Tramo v10–v21

Es el tramo con menos evidencia:

- no quedan resultados para v10–v18;
- de v19 solo queda una duración documentada de 642 s;
- v20 y v21 contienen el plan de comparación completa, pero no sus salidas;
- no hay documentación que explique por qué se guardó individualmente cada versión de diagnóstico v11–v16.

El objetivo observable de esas celdas se registra como tal. No se inventa una justificación ni se presupone que una
celda haya terminado porque figure en el notebook.

## v22 perdida y v41 rehecha

La documentación histórica atribuye a v22 baseline 10/50 y prefijos 6/50. Los artefactos originales bajo
`corridas/kaggle/v22` se perdieron y ya no están en las fuentes locales. La evidencia actual es:

1. documentación y análisis de errores producidos cuando los artefactos existían;
2. v25, que conserva el baseline 10/50;
3. v41, que rehace prefijos y reproduce 6/50, controles, contexto y casos ganados/perdidos.

v41 confirma la reproducibilidad del resultado, pero no convierte el artefacto rehecho en el artefacto original.
Por eso v22 conserva confianza media y v41 alta.

Fuentes: [informe del buscador](../informe/buscador-baseline.md#comparación-con-el-agente-2026-11-10) y
[tabla rehecha](../../evaluaciones/practica/tabla-buscador-prefijos/tabla.md).

## v24 sin procedencia completa

La corrida histórica de v24 no conservó `runner.json`. El notebook fija `2c16014`, pero el artefacto que se comparó
entonces no podía acreditar por sí mismo código, prompt y modelo. La conclusión aceptable es limitada: para los diez
casos disponibles, las conversaciones coinciden con v22 y la nueva entrega no introdujo una diferencia observable.
No es una evaluación completa ni una medición con procedencia íntegra.

## v25: resumen parcial de una corrida fallida

La sesión de v25 registra `returncode=1` para `procedimiento-costumbres-v1-n1`, pero existe un `resumen.json`
parcial con 1/20 y parcial 0,511. El kit alcanzó a producir datos antes del fallo CUDA.

Ambas fuentes son compatibles: el resumen describe un prefijo incompleto y la sesión describe el estado terminal.
Ese 1/20 no se suma ni se compara; la corrida se registra como fallida. Haría falta una corrida completa con la misma
configuración para obtener una métrica comparable.

## Configuración histórica incorrecta en v2

Los `runner.json` originales de v2 informaban lote 1, mientras que:

- la celda exportaba lote 4;
- la cantidad de micro-lotes es consistente con lote 4;
- la bitácora documenta el bug del runner y su corrección posterior.

La auditoría registra lote efectivo 4 y conserva la contradicción. No se modifica el runner histórico.

## v9: notebook en error, ejecuciones terminadas

La versión de Kaggle quedó marcada en error por la celda final de resumen. Los runners independientes de
`solo-texto-l4` y `humo-solo-texto` terminaron con returncode 0. El estado de una celda auxiliar no invalida esas
ejecuciones, pero tampoco debe reetiquetarse el notebook completo como exitoso.

## v32 y v32b

`v32` y `v32b` son extracciones relacionadas de una única ejecución iniciada el
`2026-11-11T23:38:36+00:00` con commit `6169241`:

- ambos tienen el mismo `runner.json`, `publicaciones.json` y log general;
- `v32` contiene el `runner.log` que llega al paso 315/363;
- `v32b` agrega recibos HF de checkpoints 50–300;
- el runner quedó `en_curso`, sin `fin` ni returncode, porque Kaggle cortó la sesión.

No deben sumarse como dos entrenamientos ni asignarse dos consumos. La prueba que cerraría completamente la
extracción sería un runner final escrito antes del corte; no existe. v33 sí es otra ejecución: reanuda desde 300 y
termina los pasos restantes.

## Adapter de v33 frente a adapter de v39

No son versiones del mismo archivo ni dos nombres equivalentes:

- v33 termina el SFT principal y publica `dc41674a`;
- v36 y v38 continúan desde ese adapter, pero se descartan;
- v39 continúa desde `dc41674a`, publica `770c9fac` y lo reemplaza como adapter de la entrega.

La evaluación v34 pertenece a `dc41674a`. La evaluación v40 pertenece a `770c9fac`. Mezclar sus resultados o
atribuir 100/150 al adapter v33 sería incorrecto.

## Desarrollo, práctica y test secreto

- El conjunto de desarrollo `datos/prompt-dev/v1` seleccionó prompt y variantes de adapter.
- La práctica se usó solo para reportar después de las decisiones.
- v40 ocurrió después de adoptar v39 con 31/40 en desarrollo.
- No se inspeccionó ni ejecutó el test secreto.

Los resultados 10/10/7, 14/16/12, 30/29/33 y 36/28/36 son práctica. Los valores 9/40, 26/40, 23/40,
10/40, 26/40 y 31/40 son desarrollo. Ninguno prueba generalización al test secreto.

## Tiempos y consumo

`train_runtime`, intervalo `inicio`–`fin` del runner y duración de sesión miden cosas distintas. Ejemplos:

- v6: 17,4 s de entrenamiento, 163 s de runner, ~7 min de sesión;
- v33: 5.072 s de entrenamiento y 5.445 s de runner;
- v39: 11.735 s de entrenamiento, 11.997 s de runner y 1.367 s adicionales de dev.

El campo `gpu_horas_fisicas` de [`recursos.csv`](recursos.csv), cuando se completa, es una derivación matemática de
reloj de pared × GPU visibles. No representa consumo facturado ni cuota de Kaggle.

Hay un snapshot agregado documentado de 48.546 s (13,5 h de una referencia semanal de 30 h) al 12/11. No permite
asignar consumo real por versión y no cubre las versiones posteriores. El total real histórico es mayor por
instalaciones y sesiones sin runner. Por eso `cuota_real` queda vacío en los registros por ejecución.

## Hardware y generalización

Todo lo validado en GPU usó 2× Tesla T4, fp16 y reparto automático. Las mediciones de VRAM son sumas sobre GPU
visibles y no prueban que la misma configuración entre o funcione en una A100-40GB. Tampoco los tests CPU con
dobles acreditan compatibilidad CUDA.

A100-40GB sigue sin validar. No se debe convertir una proyección de tiempo o costo en una medición.

## Información que permanece pendiente

- Resultado y estado terminal de v10–v18, salvo duración de v19.
- Resultados de v20/v21.
- Artefactos originales de v22.
- Procedencia completa del artefacto v24.
- Duración por ejecución de v26; solo hay duración de sesión documentada.
- Cierre formal del runner v32.
- Consumo real de cuota por versión.
- Toda medición en A100-40GB.
- Resultado del test secreto.
