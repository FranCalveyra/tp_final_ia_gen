# Evaluaciones reportadas

Cada número del informe sale de una de estas carpetas. Son las salidas tal cual las dejó la corrida en Kaggle
(T4 x2, fp16, lote 4), sin editar: conversaciones completas (`salida/trazas.jsonl`: cada mensaje que entró y salió
del modelo, cada llamada a herramienta con su observación y la ficha), tiempos, métricas por micro-lote,
`runner.json` con commit, modelo, adapter, prompt (con sha256), buscador y comando, y el resultado caso por caso.

## Práctica (50 casos × normas original, b y c; solo para reportar)

| Carpeta | Kaggle | Plan | Filas |
|---|---|---|---|
| `practica/v25-sin-entrenar/` | v25 | `configuraciones/evaluacion/actual.json` | `sin_entrenar` |
| `practica/v27-solo-prompt/` | v27 | `configuraciones/evaluacion/solo-prompt.json` | `solo_prompt` |
| `practica/v34-entrenado/` | v34 | `configuraciones/evaluacion/entrenado.json` | `solo_entrenamiento`, `ambos` |
| `practica/v40-entrenado-v39/` | v40 | `configuraciones/evaluacion/entrenado-v39.json` | `solo_entrenamiento`, `ambos` con el adapter de la entrega v39 (`770c9fac`) |
| `practica/tabla-v39/` | — | — | **Tabla vigente** de la Parte 4 (v25 + v27 + v40) con la procedencia de cada celda y `resultados.jsonl` por corrida |
| `practica/tabla/` | — | — | Tabla histórica (v25 + v27 + v34, adapter `dc41674a`); se conserva como historial |
| `practica/v41-buscador-prefijos/` | v41 | `configuraciones/evaluacion/buscador-prefijos.json` | Parte 2: sin entrenar con el buscador `prefijos`, solo normas originales (6/50). Rehace v22, que se perdió |
| `practica/tabla-buscador-prefijos/` | — | — | Parte 2: `baseline` (v25) contra `prefijos` (v41) con el agente sin entrenar |

La tabla se regenera, sin GPU, con:

```bash
uv run python -m scripts.tabla_evaluacion --evaluacion evaluaciones/practica/v25-sin-entrenar \
    --evaluacion evaluaciones/practica/v27-solo-prompt --evaluacion evaluaciones/practica/v40-entrenado-v39 \
    --salida <carpeta nueva>   # la vigente es practica/tabla-v39
```

## Conjunto de desarrollo (`datos/prompt-dev/v1`, 40 casos; con esto se eligió)

| Carpeta | Kaggle | Qué mide |
|---|---|---|
| `dev/v25-fase1/` | v25 | Fase 1 de la optimización de prompt: cinco candidatos sin entrenar (`scripts.rankear_prompts`) |
| `dev/v35-entrenado/` | v35 | Adapter `dc41674a` + `procedimiento-v1`: confirma que el aviso al voluntariado nunca da `true` |
| `dev/v37-formato/` | v37 | Fase 3, formato de los mensajes sin adapter: `procedimiento-formato-v1` 10/40 contra 9/40 de B (v25); no adoptado |
| `dev/v36-aviso/` | v36 | Adapter reforzado en el aviso (`22ee5596`) + `procedimiento-v1`: 23/40, no adoptado (configuración usada en `evaluacion-prompt-aviso.json`) |
| `dev/v38-peso/` | v38 | Adapter con peso 20 en el aviso (`17e9fe3d`) + `procedimiento-v1`: 26/40, igual que v35 y con 3 controles rotos; no adoptado |
| `dev/v39-datos/` | v39 | Adapter con edad y acompañante antes de la decisión (`770c9fac`) + `procedimiento-v1`: **31/40** contra 26/40 de v35; **adoptado como adapter de la entrega** |

Cada corrida del conjunto de desarrollo tiene `manifiesto.json` (snapshots y hashes de prompt, configuración,
casos y esperadas; métricas) y `entradas/`. Detalle e interpretación en [optimización de prompt](../docs/optimizacion-prompt.md),
[evaluación](../docs/evaluacion.md) y [análisis de errores](../docs/analisis-errores.md).

## Entrenamientos (recibos)

Los pesos y cada checkpoint están en HF (`FranCalveyra/tp-admision`, con su revisión) y cada corrida en W&B; acá
quedan los recibos para auditarlos sin acceso a Kaggle: `runner.json` (commit, entradas, publicaciones HF con
revisión), `runner.log` (pérdida por paso), `estado.json` (VRAM, run de W&B).

| Carpeta | Kaggle | Qué es |
|---|---|---|
| `entrenamiento/v32-v33-sft-v3/` | v32 + v33 | SFT de la entrega sobre `v3-procedimiento` (v32 cortada a las 8 h en el paso 315; v33 reanuda desde checkpoint-300). `manifiesto-tokens.json`: 2.900 conversaciones, ~10 M tokens |
| `entrenamiento/v36-aviso/` | v36 | Continuación sobre `v4-aviso` (no adoptada) |
| `entrenamiento/v38-peso/` | v38 | Continuación sobre `v5-aviso-peso` con `peso_aviso` 20 (no adoptada) |
| `entrenamiento/v39-datos/` | v39 | Continuación sobre `v6-aviso-datos` (edad y acompañante antes de la decisión): adapter de la entrega, 20,36 GB de VRAM pico, W&B `41ec01f292e64306` |

## Recursos (Parte 5: lote, VRAM, carga y humos de entrenamiento)

Corridas de Kaggle T4 x2 con el modelo sin entrenar y normas originales, salvo donde se indica. Son la evidencia de
las secciones 5.1–5.3 del informe y del [anexo de la Parte 5](../docs/informe-parte5.md).

| Carpeta | Commit | Qué mide |
|---|---|---|
| `recursos/base-original-l4/`, `base-b-l4/`, `base-c-l4/` | `a89f5b9` | Lote 4 con las normas original, b y c (10/10/7), con las torres de visión y audio |
| `recursos/lote-original-l1/`, `lote-original-l8/`, `lote-original-l16/` | `11bf3d6` | Barrido de lote: el lote 1 da 10/50; los lotes 8 y 16 caen con `CUDA error: unspecified launch failure` |
| `recursos/diag-l8/` | `522d540` | Lote 8 con `CUDA_LAUNCH_BLOCKING=1`: completa, pero es más lento que el lote 4 |
| `recursos/solo-texto-l4/` | `a828467` | Lote 4 sin las torres de visión y audio: la VRAM baja de 19,1 a 18,1 GB, con las mismas trazas |
| `recursos/humo-solo-texto/`, `humo-entrena-1-v6/`, `humo-entrena-2-v7/` | `a828467`, `acc95b2` | Humos de entrenamiento: pipeline, checkpoints y reanudación. Los checkpoints traen solo `trainer_state.json`, sin pesos |

## Buscador por separado (Parte 2, CPU)

Salidas de `evaluaciones/rag/`, regeneradas en CPU el 2026-11-13 con las mismas métricas que el informe:

| Archivo | Comando |
|---|---|
| `rag/evaluar_rag-{baseline,sin_comunes,prefijos}-original.txt` | `BUSCADOR_VARIANTE=<variante> python -m admision.evaluar_rag --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl` |
| `rag/comparar-original.json` | `python -m scripts.comparar_buscadores --salida <archivo nuevo>` |
| `rag/comparar-b-evidencia-ajustada.json` | `python -m scripts.comparar_buscadores --normas normas/b --evidencia-ajustada --salida <archivo nuevo>` |
| `rag/exploracion-original.json` | `python -m scripts.explorar_buscadores --salida <archivo nuevo>`: las 68 configuraciones |
| `rag/comparar-variantes-20261101.json` | Primera comparación de variantes (2026-11-01), copiada sin editar |
