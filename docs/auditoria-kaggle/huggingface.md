# Auditoría de publicaciones en Hugging Face

Este documento extiende la [auditoría de Kaggle](README.md) con el inventario de adapters y checkpoints
publicados. No vuelve a interpretar las 41 versiones: enlaza cada recibo de publicación con su entrenamiento,
su corrida W&B y, cuando existe, su evaluación posterior.

## Alcance y acceso

- Repositorio registrado por todos los runners históricos: `FranCalveyra/tp-admision`.
- Repositorio actual informado: `TP-Final-IA-Gen/tp-admision`.
- Ambos endpoints de metadata respondieron HTTP 401 al acceso anónimo durante esta auditoría.
- No se usaron tokens ni se descargaron pesos, archivos `.safetensors` o checkpoints.
- La interfaz autenticada mostrada en las capturas contiene 47 commits, incluido un
  `initial commit` con prefijo `3371bb7`. Los recibos locales conservan 46 revisiones de publicación distintas.
- El acceso anónimo automatizado (HTTP 401) y la inspección en una sesión autenticada del navegador
  son dos fuentes distintas. La columna `acceso_directo` del CSV expresa el resultado de la consulta
  anónima de Codex, **no** la ausencia de los commits en la interfaz autenticada.

Por lo tanto, las revisiones de este inventario están verificadas contra recibos locales, no contra el historial
remoto actual. Tampoco puede probarse directamente que ambos namespaces sean el mismo repositorio transferido:
los artefactos históricos sólo nombran el namespace personal y no existe un recibo local de transferencia.

## Evidencia y criterio

Una revisión se considera **verificada localmente** cuando aparece como `hf_revision` en un
`publicaciones.json`, `publicacion-hf.json`, `estado.json` o `runner.json`. Eso prueba que el proceso recibió ese
identificador de Hugging Face; no prueba que la revisión siga accesible hoy ni permite inspeccionar sus archivos.

Las fechas corresponden a las sesiones de Kaggle. Los recibos HF no traen timestamp propio. El inventario
completo está en [`huggingface-revisiones.csv`](huggingface-revisiones.csv).

## Publicaciones por entrenamiento

| Kaggle | Publicaciones locales | Entrada o reanudación | Revisión final | Uso posterior |
|---|---:|---|---|---|
| v6 | 5 | Entrenamiento de humo nuevo | `2fa71b9945a39accb8cff212f2e8cdb265ea1d19` | Valida publicación durable; datos sintéticos. |
| v7 | 4 | `checkpoint-2@f11b0814564f09e2ff788937db3fba111317f149` de v6 | `391c3eba7c89fd91054d902afbf8b7e7100f82e8` | Valida reanudación; no mide calidad. |
| v9 | 5 | Humo nuevo con carga sólo texto | `69570fc94e663bf68b70765f431e1b99f93e7774` | Valida el pipeline optimizado; datos sintéticos. |
| v32 | 6 | SFT principal nuevo | `checkpoint-300@83dd1d205763e4885b66ac14b177799785b3fd69` | Último checkpoint durable antes del corte de Kaggle. |
| v33 | 3 | Reanuda exactamente `83dd1d…/checkpoint-300` | `dc41674a8a2561a724884be2170ff6808efa6e96` | Primer adapter completo; adoptado provisionalmente. |
| v36 | 5 | Continúa `dc41674a…/final` | `22ee5596c8d3014bd86caaca6a87034d06e1d4e8` | Publicado, pero descartado tras 23/40 en desarrollo. |
| v38 | 9 | Continúa `dc41674a…/final` | `17e9fe3d8a422284d6337a9372f697377d43319e` | Publicado, pero descartado: 26/40 y regresiones por control. |
| v39 | 9 | Continúa `dc41674a…/final` | `770c9fac84d5718b0b2cf4c40c3451cad322ac9e` | Adapter definitivo: 31/40 en desarrollo; práctica posterior en v40. |

Total verificable mediante recibos locales: **46 revisiones distintas**. Las capturas muestran además un
`initial commit` (`3371bb7`), por lo que **46 publicaciones + 1 inicial = 47** es una explicación
plausible de la cantidad visible. Todavía falta cotejar exhaustivamente las 47 entradas con los 46
recibos para demostrar una correspondencia exacta; ese dato no prueba una migración o transferencia.

## Linaje acreditado

```text
v32 checkpoint-300 83dd1d…
            │
            └─ v33 final dc41674a…
                    ├─ v36 final 22ee5596… → descartado por evaluación
                    ├─ v38 final 17e9fe3d… → descartado por evaluación
                    └─ v39 final 770c9fac… → adoptado
```

Los runners v36, v38 y v39 registran `dc41674a…` como `adapter_inicial`; no son reetiquetados del mismo adapter,
sino tres continuaciones distintas. La selección depende de las evaluaciones de desarrollo, no de que la
publicación exista ni de su pérdida de entrenamiento.

## Reanudación v32–v33

El `runner.json` de v33 fija simultáneamente:

- modo `reanudacion`;
- repositorio `FranCalveyra/tp-admision`;
- checkpoint `checkpoint-300`;
- revisión `83dd1d205763e4885b66ac14b177799785b3fd69`.

Esa revisión coincide con el sexto recibo de v32. v32 y v32b siguen siendo dos extracciones de la misma sesión;
v33 sí es una ejecución posterior que repite los pasos 301–315 y completa hasta 363.

## Adopción y descartes

- v36 y v38 publicaron correctamente. “Publicado” describe el estado técnico, no una decisión de calidad.
- Sus descartes están respaldados por [`analisis-errores.md`](../informe/analisis-errores.md): v36 introdujo
  demasiados falsos positivos y v38 no mejoró el total de v35 y rompió otros controles.
- v39 mejoró de 26/40 a 31/40 en desarrollo y fue adoptado antes de la práctica de v40. La tabla final está en
  [`evaluaciones/practica/tabla-v39/`](../../evaluaciones/practica/tabla-v39/).

## Pendientes

- Cotejo exhaustivo (autenticado y de sólo lectura) de las 47 entradas remotas, sus identificadores completos y timestamps; las capturas sólo acreditan una inspección parcial.
- Prueba documental de transferencia o equivalencia entre los dos namespaces.
- Confirmación de que las 46 revisiones históricas siguen resolviendo en el repositorio actual.
