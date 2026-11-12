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

Cada corrida del conjunto de desarrollo tiene `manifiesto.json` (snapshots y hashes de prompt, configuración,
casos y esperadas; métricas) y `entradas/`. Detalle e interpretación en [optimización de prompt](../docs/optimizacion-prompt.md),
[evaluación](../docs/evaluacion.md) y [análisis de errores](../docs/analisis-errores.md).
