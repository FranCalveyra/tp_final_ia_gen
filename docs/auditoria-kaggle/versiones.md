# Cronología de versiones v1–v41

Esta cronología separa el estado técnico del resultado experimental. Los hashes completos de los notebooks, los
campos normalizados y las referencias de evidencia están en [`versiones.csv`](versiones.csv) y
[`evidencia.csv`](evidencia.csv).

Convenciones:

- **Continuó:** el intento falló o fue parcial y motivó una versión posterior.
- **Adoptado:** el cambio pasó a la línea de entrega en ese momento; puede haber sido reemplazado después.
- **Descartado:** se decidió no usar la alternativa.
- **Solo reporte:** la medición no intervino en la selección.
- Confianza A/M/B: alta, media o baja, según [la metodología](README.md#clasificación-de-evidencia).

## v1–v9: infraestructura, lotes y humo SFT

| V | Fecha | Commit acreditado | Objetivo y justificación documentada | Estado técnico | Resultado experimental | Decisión | Conf. |
|---:|---|---|---|---|---|---|:---:|
| v1 | 2026-11-03 | Sin pin; la corrida conservada registra `2dc0545` | Lograr una primera ejecución real después de incompatibilidades de paquetes, processor multimodal y atención SDPA. El notebook también conserva celdas de experimentos posteriores, por lo que no representa una sola corrida. | `t4-original-09` terminó; otros intentos se cortaron o perdieron. | 0/3, parcial 0,852; 30,1 s/caso; 16,3 GB pico. No alcanza para comparar calidad. | Continuó hacia baseline completo. | M |
| v2 | 2026-11-04 | Runner `a89f5b9` | Obtener una línea base persistente de 50 casos para original, b y c; el runner nuevo debía conservar evidencia ante cortes. | Tres runners terminados, código limpio, 2×T4. | 10/50, 10/50 y 7/50; parcial 0,831/0,820/0,816. | Baseline histórico adoptado. | A |
| v3 | 2026-11-04 | Runner `11bf3d6` | Medir efecto del lote y comprobar que el runner registrara la configuración efectiva. | Lote 1 terminó; lotes 8/16 cayeron dentro de `jugar_lote` aunque el proceso histórico devolvió 0. | L1: 10/50 y 33,3 s/caso. L8/L16: corridas incompletas, resultados inválidos. | Se mantiene lote 4 de v2: mayor lote estable y 2,1× más rápido que L1. | A |
| v4 | 2026-11-04 | Runner `6f9abc0` | Primer humo real del pipeline SFT con publicación durable. | Falló con `ValueError`; el CLI no mostraba traceback útil. | Sin checkpoint ni evidencia de calidad. | Continuó con mejor diagnóstico. | A |
| v5 | 2026-11-04 | Runner `5e1837e` | Repetir v4 con traceback visible. | Falló: LoRA alcanzaba capas no soportadas de visión/audio. | Sin checkpoint. | Continuó limitando LoRA a la atención del language model. | A |
| v6 | 2026-11-04 | Runner `acc95b2`; notebook no fijaba commit | Validar el humo después de restringir los módulos LoRA. | Terminó 8 pasos y publicó checkpoints 2/4/6/8/final. | Pérdida 2,679→2,178; `train_runtime` 17,4 s. Datos sintéticos: prueba pipeline, no calidad. | Pipeline y publicación adoptados. | A |
| v7 | 2026-11-04 | `acc95b2`; HF `f11b081…` | Verificar reanudación desde un checkpoint publicado, incluyendo optimizador, scheduler, RNG y scaler. | Reanudó checkpoint-2 y completó pasos 3–8. | Curva cercana a v6 hasta cuatro decimales; final `391c3eb…`. | Reanudación validada. | A |
| v8 | 2026-11-04 | Runner `522d540`; notebook sin pin | Diagnosticar los fallos de lote 8 con kernels sincronizados y stacktrace CUDA. | Terminó 50 casos con bloqueo CUDA. | 10/50, mismos casos que lote 1; 17,4 s/caso, peor que lote 4; 21,7 GB pico. | Descartado: útil como diagnóstico, no como configuración. | A |
| v9 | 2026-11-05 | Runner `a828467`; notebook sin pin | Evitar cargar torres de visión/audio que el agente de texto no usa y medir el ahorro. | Inferencia y humo terminaron; la celda final de resumen dejó el notebook en error. | Inferencia 10/50, trazas idénticas; VRAM 19,1→18,1 GB. Humo 15,675 GB. | Carga solo texto adoptada. | A |

Detalle y contexto: [bitácora de las primeras corridas](../progreso.md#2026-11-03-primera-corrida-real-del-agente-kaggle-t4-no-comparable-con-a100),
[lotes](../progreso.md#2026-11-04-efecto-del-tamaño-de-lote-en-t4-issue-4-no-comparable-con-a100) y
[humo SFT](../progreso.md#2026-11-04-entrenamiento-sft-en-kaggle-t4-x2-humo-issue-3-implementado-probado-con-dobles-sin-gpu).

## v10–v24: buscador, accesos y ciclo del agente

| V | Fecha | Rama/commit del notebook | Objetivo verificable | Estado técnico y resultado | Decisión | Conf. |
|---:|---|---|---|---|---|:---:|
| v10 | 2026-11-10 | `feat/6-buscador-eze@5208810` | Humo `prefijos`, original, 3 casos, lote 4. | No quedan logs ni resultados. | Pendiente; no se atribuye una razón adicional. | B |
| v11 | 2026-11-10 inferida | Sin pin; GPU off | Diagnóstico de disponibilidad y lectura de integraciones. | No queda salida. No fue una medición del agente. | Pendiente. | B |
| v12 | 2026-11-10 inferida | Sin pin; GPU off | Ampliar el diagnóstico anterior por distintos mecanismos de acceso. | No queda salida. | Pendiente. | B |
| v13 | 2026-11-10 inferida | Sin pin; GPU off | Diagnóstico simplificado del entorno. | No queda salida. | Pendiente. | B |
| v14 | 2026-11-10 inferida | Sin pin; GPU off | Mismo código semántico que v13 bajo otra imagen del kernel. | No queda salida; solo cambia metadata/orden JSON. | Pendiente. | B |
| v15 | 2026-11-10 inferida | Sin pin; GPU off | Repite el diagnóstico de v12. | Notebook idéntico a v12. Sin salida. | Pendiente. | B |
| v16 | 2026-11-10 inferida | Sin pin; GPU off | Segunda repetición del diagnóstico de v12. | Notebook idéntico a v12/v15. Sin salida. | Pendiente. | B |
| v17 | 2026-11-10 inferida | `feat/6-buscador-eze@5208810` | Nuevo intento del humo `prefijos` de v10. | Notebook idéntico a v10. Sin resultado. | Pendiente. | B |
| v18 | 2026-11-10 inferida | Igual a v17 | Mismo intento; cambia solo orden de claves JSON. | Sin resultado. | Pendiente. | B |
| v19 | 2026-11-10 | Igual a v10/v17 | Humo `prefijos` de 3 casos anterior a la comparación completa. | Sesión documentada de 642 s; no se conserva métrica de calidad. | Continuó hacia v22. | M |
| v20 | 2026-11-10 inferida | `feat/6-buscador-eze@5208810` | Ampliar a baseline y prefijos, 50 casos cada uno. | Sin artefactos. | Pendiente. | B |
| v21 | 2026-11-10 inferida | Igual a v20 | Repetición del plan de v20. | Notebook idéntico; sin artefactos. | Pendiente. | B |
| v22 | 2026-11-10 | `feat/6-buscador-eze@4e3613a` | Comparar el buscador con el agente real; el RAG aislado no bastaba para elegir. | Ambas corridas terminaron. Los artefactos originales se perdieron después. | Baseline 10/50; prefijos 6/50, gana 3 y pierde 7. Se mantiene `baseline`. | M |
| v23 | 2026-11-10 | `main@5d4b657` | Comprobar que `AgentePrompt` con prompt baseline reprodujera `AgenteJSON`. | 50/50 conversaciones idénticas a v22; 10/50, parcial 0,831. | Equivalencia validada; continúa `AgentePrompt`. | M |
| v24 | 2026-11-10 | `feat/7-agente-eze@2c16014` | Probar `entrega.py` con tope de 40 llamadas y guarda de contexto. | 10 casos, sin error; falta `runner.json`. 10/10 conversaciones idénticas a v22; 1/10 resuelto. | Validación parcial; la guarda y el tope no se activaron. | M |

La razón individual de guardar v11–v18 no está documentada. La tabla registra el propósito observable de las
celdas, no inventa una justificación histórica. La evidencia de v22 fue rehecha en v41; véase el
[informe del buscador](../informe/buscador-baseline.md#comparación-con-el-agente-2026-11-10) y el
[ciclo del agente](../progreso.md#2026-11-10-ciclo-del-agente-y-contexto-issue-7).

## v25–v35: prompt y primer SFT real

| V | Fecha | Rama/commit | Objetivo y problema | Estado técnico | Resultado | Decisión | Conf. |
|---:|---|---|---|---|---|---|:---:|
| v25 | 2026-11-10/11 | `feat/13-prompt-eze@f4d7300` | Fase 1: comparar cinco prompts en desarrollo y producir la fila práctica `sin_entrenar`. | Nueve corridas dev completas y C-n1 fallida; tres corridas de práctica terminadas. | Dev: B 9/40, baseline 8/40, D 2/40, A 1/40; C no comparable. Práctica baseline 10/10/7. | B queda finalista; práctica solo reporte. | A |
| v26 | 2026-11-11 | `feat/13-prompt-eze@275c15b` | Cerrar el intento C y comprobar B/C con `prefijos`. | C vuelve a fallar de forma determinista; faltan artefactos locales de esta versión. | B+prefijos 4/40 frente a B+baseline 9/40; C+prefijos incompleto. | Se eligen B `procedimiento-v1` y buscador baseline. | M |
| v27 | 2026-11-11 | `feat/13-prompt-eze@a5745f0` | Reportar `solo_prompt` en práctica después de elegir en desarrollo. | Tres runners terminados. | 14/16/12 frente a 10/10/7; 23 casos ganados y 8 perdidos. | B continúa y genera los datos `v3-procedimiento`. | A |
| v28 | 2026-11-11 | `main@8843004` | Primer SFT real sobre 2.900 conversaciones y ~10 M tokens. | Preparación/tokenización OK; OOM en paso 0 al convertir todos los logits a fp32. | Sin checkpoint ni publicación. | Continuó con pérdida solo en posiciones supervisadas. | A |
| v29 | 2026-11-11 | `fix/3-perdida-supervisada@10e117c` | Probar la pérdida selectiva. | Falló en paso 0: máscara y tensor indexado en GPU distintas. | Sin entrenamiento efectivo. | Continuó corrigiendo colocación. | A |
| v30 | 2026-11-11 | `fix/3-perdida-supervisada@f5a7196` | Repetir con corrección de máscara. | Falló antes de cargar el modelo por timeout de W&B. | Sin entrenamiento. | Reintento con timeout mayor. | A |
| v31 | 2026-11-11 | Mismo commit que v30 | Separar el timeout del fallo de dispositivo. | Con timeout mayor reapareció el error de dispositivo. | Sin entrenamiento efectivo. | Continuó pasando posiciones como lista de enteros. | A |
| v32 | 2026-11-11/12 | `fix/3-perdida-supervisada@6169241` | Validar en GPU el arreglo de posiciones y completar la época. | Entrenó hasta 315/363; Kaggle cortó la sesión a 8 h. Publicó hasta checkpoint-300. | Pérdida ~0,44 inicial, ~0,03 paso 100 y ~0,005 paso 268. | Continuó desde checkpoint-300. | A, extracción parcial |
| v33 | 2026-11-12 | `6169241`; HF `83dd1d…` | Reanudar la ejecución de v32 sin repetir toda la época. | Terminó 363 pasos; returncode 0; 19,966 GB pico. | Final `dc41674a`; pérdida media final 0,0044. | Primer adapter completo adoptado; luego reemplazado por v39. | A |
| v34 | 2026-11-12 | `fix/3-perdida-supervisada@3adf12e` | Medir `solo_entrenamiento` y `ambos` con `dc41674a`. | Seis corridas de práctica terminadas. | Solo entrenamiento 11/13/12; ambos 30/29/33 = 92/150. | Adapter+prompt adoptados provisionalmente. | A |
| v35 | 2026-11-12 | `fix/3-perdida-supervisada@bea206b` | Verificar en desarrollo la falla principal del adapter. | Dos corridas dev terminadas. | 26/40; aviso nunca `true`, 0/8 positivos. | Se usa como baseline para v36/v38/v39. | A |

La selección de prompt está detallada en [optimización de prompt](../informe/optimizacion-prompt.md#resultados-kaggle-v25v27).
El entrenamiento, sus fallos y la reanudación están en [entrenamiento SFT](../informe/entrenamiento-sft.md) y
[Parte 5](../informe/informe-parte5.md#31-entrenamiento-real-kaggle-v32--v33-2026-11-1112).

## v36–v41: aviso, adapter final y cierre de evidencia

| V | Fecha | Rama/commit | Hipótesis u objetivo | Estado técnico | Resultado | Decisión | Conf. |
|---:|---|---|---|---|---|---|:---:|
| v36 | 2026-11-12 | `fix/3-perdida-supervisada@fb353c0` | Reforzar el aviso continuando `dc41674a` con 47,7 % de ejemplos positivos. | 80 pasos + dev terminados; final `22ee5596`; 20,289 GB. | 23/40; 6/8 positivos, pero 12 falsos `true`. | No adoptado: movió el sesgo, no aprendió la condición. | A |
| v37 | 2026-11-12 | `fix/3-perdida-supervisada@7b92583` | Cambiar el formato de observaciones/errores sin adapter. | Dos corridas dev terminadas. | 10/40 frente a 9/40; 2 ganados y 1 perdido, contexto −2 %. | No adoptado: dentro del ruido y requería reentrenar. | A |
| v38 | 2026-11-12/13 | `feat/15-peso-aviso@984fc18` | Dar peso 20 al token del aviso sin alterar tanto la proporción. | 142 pasos + dev terminados; final `17e9fe3d`; 20,289 GB. | 26/40, empate con v35; 1/8 positivos, 3 falsos `true`, otros controles rotos. | No adoptado. | A |
| v39 | 2026-11-13 | `feat/15-aviso-datos@5a36ecf` | Anteponer edad y acompañante a la decisión para facilitar la regla compuesta. | 142 pasos + dev terminados; final `770c9fac`; 20,356 GB. | 31/40 frente a 26/40; 5/8 positivos y 2 falsos `true`. | Adoptado como adapter definitivo; reemplaza a v33. | A |
| v40 | 2026-11-13 | `feat/15-aviso-datos@1405aab` | Reportar en práctica el adapter ya elegido en desarrollo. | Seis corridas terminadas. | Solo entrenamiento 17/14/20; ambos 36/28/36 = 100/150. | Solo reporte; no selecciona modelo ni prompt. | A |
| v41 | 2026-11-13 | `docs/informe-completo@6f6a4f5` | Rehacer con evidencia versionada la comparación de buscadores de v22. | Una corrida de 50 casos terminada. | Prefijos 6/50 frente a baseline v25 10/50; reproduce v22 en controles y casos discordantes. | Confirma mantener baseline. | A |

Las comparaciones v35/v36/v38/v39 están en el [análisis de errores](../informe/analisis-errores.md) y los
resultados finales en la [tabla v39](../../evaluaciones/practica/tabla-v39/tabla.md). La práctica v40 ocurrió
después de adoptar v39 y no puede reinterpretarse como selección.

## Lectura final

- **Implementado y validado en T4:** adapter v39 `770c9fac` + `procedimiento-v1` + buscador baseline, lote 4.
- **Resultado de desarrollo usado para elegir v39:** 31/40.
- **Resultado de práctica usado solo para reportar:** 100/150 en la fila `ambos` de v40.
- **Sin evidencia:** generalización al test secreto y funcionamiento/rendimiento en A100-40GB.
