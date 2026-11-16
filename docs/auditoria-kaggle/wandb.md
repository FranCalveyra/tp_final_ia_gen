# Auditoría de corridas en Weights & Biases

Este documento inventaría las corridas del proyecto `francalveyra-ia-gen/tp-admision` registradas por los
runners de Kaggle. El proyecto está alojado en Forge/CoreWeave. La aplicación respondió HTTP 200, pero el
contenido de las corridas requiere sesión y no fue accesible directamente.

## Resultado del inventario

Se recuperaron **12 Run IDs distintos de los archivos locales**. La captura autenticada de la tabla
de W&B también muestra 12 entradas, **pero no son el mismo conjunto**: 11 se identifican con
corridas locales, el ID de v30 (`3466b67b56f94861`) no figura y hay una entrada adicional de
verificación (`verifica...ntacion`) cuyo ID completo no se extrajo. No se supone equivalencia por
coincidir las cantidades:

| Run ID | Kaggle | Estado técnico local | Resultado del entrenamiento |
|---|---|---|---|
| `cd5bab60479c4038` | v4 | Fallida | Primer humo: `ValueError`; sin pasos ni publicación. |
| `5d92dd61d93a4ad0` | v5 | Fallida | LoRA alcanzó módulos multimodales no soportados. |
| `b9088ec869814b35` | v6 y v7 | Finalizada | Humo de 8 pasos y reanudación desde checkpoint-2; mismo Run ID. |
| `c2b7df5c7f3b4b12` | v9 | Finalizada | Humo sólo texto, 8 pasos. |
| `16344974702649b1` | v28 | Fallida | OOM en paso 0; W&B local `_runtime=143 s`. |
| `27c3598ace44438b` | v29 | Fallida | Tensores en dispositivos distintos, paso 0; `_runtime=134 s`. |
| `3466b67b56f94861` | v30 | Fallida | Timeout al inicializar W&B; no cargó el modelo. |
| `868c27bb01974179` | v31 | Fallida | Reaparece el error de dispositivo en paso 0. |
| `3add2742ddfd4b42` | v32 y v33 | Finalizada localmente en v33 | SFT principal: corte en 315 y reanudación 300→363. |
| `6d8edb168beb4e9b` | v36 | Finalizada y sincronizada | 80 pasos; adapter publicado y luego descartado por evaluación. |
| `f9326b10342b4be4` | v38 | Finalizada y sincronizada | 142 pasos; adapter publicado y luego descartado. |
| `41ec01f292e64306` | v39 | `Finished` en captura; finalizada localmente | 142 pasos; adapter adoptado. |

El detalle normalizado está en [`wandb-corridas.csv`](wandb-corridas.csv). Allí se distinguen
`estado_tecnico_local`, acceso de API no verificado y estado observado en captura. En la interfaz
las once corridas coincidentes muestran `Finished` o `Failed`; **v30 queda sin estado remoto
comprobado**. Esas observaciones corresponden a las capturas compartidas el 15/11/2026,
no a una consulta automatizada a la API. La corrida adicional de verificación no se agrega
al CSV de IDs locales porque no conocemos su identificador completo.

## Contraste de v39

| Campo | Dato observado | Contraste local | Conclusión |
|---|---:|---|---|
| Run ID | `41ec01f292e64306` | `estado.json` y URL del runner | Coincide. |
| Estado W&B | Finished | El log termina con resumen y sincronización; runner returncode 0 | Compatible; el estado remoto sólo está respaldado por la captura informada. |
| Commit | `5a36ecf720080d2061d69a4af86273104ec4bc82` | `runner.json` y snapshot de máquina | Coincide. |
| Runtime W&B | 3 h 17 min 51 s = 11.871 s | No quedó `wandb-summary.json` local | Se conserva como observación de interfaz, no se sustituye por tiempo de runner. |
| Runtime de runner | — | 09:00:33–12:20:30 = 11.997 s | Medición distinta de W&B. |
| `train_runtime` | 11.735,1174 s | Resumen del log | Coincide exactamente. |
| Paso | 142 | Progreso 142/142 y checkpoint-142 | Coincide. |
| `train_loss` | ≈0,039797 | Log local redondea a 0,0398 | Compatible por redondeo. |
| VRAM pico | 20,356 GB | `runner.json` y `estado.json` | Coincide; suma sobre dos GPU visibles. |
| GPU | 2 Tesla T4 | Snapshot de máquina | Coincide. |
| Precisión | FP16 | Configuración del runner | Coincide. |
| HF final | `770c9fac84d5718b0b2cf4c40c3451cad322ac9e` | Recibo `final` | Coincide. |

Que el run esté Finished y que la pérdida sea baja sólo acredita terminación técnica. La adopción de v39 se
decidió por 31/40 en desarrollo frente a 26/40 de v35, no por esos dos campos.

## Tres relojes diferentes

- **Runtime W&B:** intervalo atribuido por W&B a la corrida. El CSV mantiene por separado
  `wandb_runtime_s` (resumen local, cuando existe) y `wandb_runtime_interfaz_s` (captura).
- **Runtime del runner:** diferencia entre `inicio` y `fin` del proceso envolvente; incluye carga,
  publicación y cierre. Para v39 son 11.997 s.
- **`train_runtime`:** tiempo informado por Trainer para entrenamiento efectivo; para v39 son 11.735,1174 s.

Duraciones **observadas en la tabla de W&B** (no son cuotas Kaggle):

| Ejecución | Runtime W&B en captura | Segundos exactos visibles | Observación |
|---|---|---:|---|
| v32–v33 | 12 h 31 min 23 s | 45.083 | Un Run ID compartido; no equivale a un solo entrenamiento ininterrumpido. |
| v36 | 1 h 59 min 13 s | 7.153 | El resumen local registra 7.151 s; se conservan ambos. |
| v38 | 3 h 17 min 57 s | 11.877 | Corrida terminada, adapter no adoptado. |
| v39 | 3 h 17 min 51 s | 11.871 | Corrida terminada, adapter adoptado. |

La captura también muestra 2 min 25 s para v28, 2 min 16 s para v29 y v31, y tiempos
redondeados a minutos para otras corridas. Los minutos redondeados no se convierten a segundos
exactos. Ninguno de estos relojes se suma ni se convierte en consumo real de cuota de Kaggle.
Tampoco se presentan GPU-horas de W&B como cuota: W&B registra telemetría.

## Métricas y configuración

Los entrenamientos reales v33, v36, v38 y v39 usaron FP16, dos Tesla T4, batch 1, acumulación 8 y longitud
máxima 8192. v33 usó tasa inicial `2e-4`; v36, v38 y v39 usaron `1e-4`. Los runs v38 y v39 además fijaron peso
20 para el aviso. La revisión del modelo base fue `ee0ef6023621cff504d758262d4e04895a5af4a2`.

| Run | Pasos | `train_runtime` | `train_loss` | VRAM pico | HF final |
|---|---:|---:|---:|---:|---|
| v33 / `3add…` | 363; el segmento v33 reanuda en 300 | 5.072,188 s para v33 | 0,004363 para el segmento v33 | 19,966 GB | `dc41674a…` |
| v36 / `6d8e…` | 80 | 7.029,4381 s | 0,005596998 | 20,289 GB | `22ee5596…` |
| v38 / `f932…` | 142 | 11.750,0218 s | 0,03002 | 20,289 GB | `17e9fe3d…` |
| v39 / `41ec…` | 142 | 11.735,1174 s | 0,0398 local | 20,356 GB | `770c9fac…` |

Los valores de v33 no representan toda la época v32+v33: son el resumen del segmento reanudado. v32 llegó al
paso 315 antes del corte y v33 repitió 301–315 desde el checkpoint durable 300.

## Limitaciones

- No se consultó una API autenticada para nombres, tags, estados o runtime. Los estados y algunas
  duraciones de once corridas se observaron en capturas autenticadas; v30 quedó sin respaldo remoto.
- Los logs locales prueban sincronización de varios runs, pero “Synced” no equivale por sí solo a `Finished`.
- No se recuperaron historiales completos desde la API ni se usaron credenciales para acceso automatizado.
- En intentos fallidos antes de entrenar, `train_runtime` permanece vacío en el CSV: el valor 0
  no era una medición del Trainer.
- La pérdida de entrenamiento no se interpreta como métrica de calidad o generalización.
