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
| `practica/tabla/` | — | — | Tabla de la Parte 4 con la procedencia de cada celda y `resultados.jsonl` por corrida |

La tabla se regenera, sin GPU, con:

```bash
uv run python -m scripts.tabla_evaluacion --evaluacion evaluaciones/practica/v25-sin-entrenar \
    --evaluacion evaluaciones/practica/v27-solo-prompt --evaluacion evaluaciones/practica/v34-entrenado \
    --salida <carpeta nueva>
```

## Conjunto de desarrollo (`datos/prompt-dev/v1`, 40 casos; con esto se eligió)

| Carpeta | Kaggle | Qué mide |
|---|---|---|
| `dev/v25-fase1/` | v25 | Fase 1 de la optimización de prompt: cinco candidatos sin entrenar (`scripts.rankear_prompts`) |
| `dev/v35-entrenado/` | v35 | Adapter `dc41674a` + `procedimiento-v1`: confirma que el aviso al voluntariado nunca da `true` |
| `dev/v37-formato/` | v37 | Fase 3, formato de los mensajes sin adapter: `procedimiento-formato-v1` 10/40 contra 9/40 de B (v25); no adoptado |
| `dev/v36-aviso/` | v36 | Adapter reforzado en el aviso (`22ee5596`) + `procedimiento-v1`: 23/40, no adoptado (configuración usada en `evaluacion-prompt-aviso.json`) |
| `dev/v38-peso/` | v38 | Adapter con peso 20 en el aviso (`17e9fe3d`) + `procedimiento-v1`: 26/40, igual que v35 y con 3 controles rotos; no adoptado |

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
