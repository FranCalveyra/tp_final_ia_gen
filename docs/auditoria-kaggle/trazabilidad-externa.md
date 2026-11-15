# Trazabilidad externa: Git → Kaggle → W&B → Hugging Face → evaluación

Esta vista une los identificadores ya auditados sin duplicar la cronología v1–v41. Cada fila distingue ejecución,
telemetría, publicación y decisión experimental. Una corrida terminada o una revisión publicada no implica que
el modelo haya sido adoptado.

| Git | Kaggle | W&B | Hugging Face | Evaluación y decisión |
|---|---|---|---|---|
| `6f9abc0` | v4 | `cd5bab60479c4038` | Sin publicación | Fallo técnico; continúa. |
| `5e1837e` | v5 | `5d92dd61d93a4ad0` | Sin publicación | Fallo técnico; continúa. |
| `acc95b2` | v6/v7 | `b9088ec869814b35` | v6 `2fa71b…`; v7 `391c3e…` | Humo y reanudación validados; sin afirmación de calidad. |
| `a828467` | v9 | `c2b7df5c7f3b4b12` | `69570f…` | Pipeline sólo texto validado; datos sintéticos. |
| `8843004` | v28 | `16344974702649b1` | Sin publicación | OOM en paso 0. |
| `10e117c` | v29 | `27c3598ace44438b` | Sin publicación | Error de dispositivos en paso 0. |
| `f5a7196` | v30 | `3466b67b56f94861` (solo recibo local) | Sin publicación | Timeout de W&B antes de cargar el modelo; el ID no figura entre las 12 entradas visibles en la captura. |
| `f5a7196` | v31 | `868c27bb01974179` | Sin publicación | Error de dispositivos en paso 0. |
| `6169241` | v32 | `3add2742ddfd4b42` | `83dd1d…/checkpoint-300` | Sesión cortada; checkpoint durable usado por v33. |
| `6169241` | v33 | `3add2742ddfd4b42` | `dc4167…/final` | Primer adapter completo; 26/40 en v35; adopción provisional. |
| `fb353c0` | v36 | `6d8edb168beb4e9b` | `22ee55…/final` | 23/40; descartado por falsos positivos. |
| `984fc18` | v38 | `f9326b10342b4be4` | `17e9fe…/final` | 26/40; descartado por empate y regresiones. |
| `5a36ecf` | v39 | `41ec01f292e64306` | `770c9f…/final` | 31/40; adoptado como adapter definitivo. |
| `1405aab` | v40 | Sin entrenamiento W&B nuevo | Usa `770c9f…/final` | Práctica 100/150, sólo reporte posterior a selección. |

## Cadena principal

1. v32 publica `checkpoint-300@83dd1d…` antes del corte.
2. v33 reanuda ese checkpoint, completa el SFT y publica `dc41674a…/final`.
3. v36, v38 y v39 parten de `dc41674a…`, pero producen revisiones independientes.
4. Las evaluaciones de desarrollo descartan v36 y v38 y seleccionan v39.
5. v40 carga `770c9fac…` para práctica; no produce ni selecciona otro adapter.

Las relaciones exactas de checkpoints están en [`huggingface-revisiones.csv`](huggingface-revisiones.csv), y
los tiempos y estados en [`wandb-corridas.csv`](wandb-corridas.csv). Los resultados por caso permanecen en
[`evaluaciones/`](../../evaluaciones/) y en el [análisis de errores](../informe/analisis-errores.md).

## Clases de evidencia

- **Directa local:** runner, estado, recibo HF, summary W&B o log conservado.
- **Versionada:** artefacto directo local incluido en `evaluaciones/`.
- **Documental:** bitácora o informe que preserva un hecho cuyo artefacto completo ya no está.
- **Observación de interfaz:** dato comunicado desde una captura privada; se contrasta, pero no se presenta como
  acceso directo de esta auditoría.
- **Inferencia:** relación limitada entre fuentes concordantes, identificada expresamente.

## Límites de acceso

- Los dos endpoints HF respondieron 401 y Forge no expuso el proyecto sin sesión.
- Los 46 recibos históricos nombran `FranCalveyra/tp-admision`; ninguno documenta el namespace actual.
- No puede acreditarse la equivalencia o transferencia a `TP-Final-IA-Gen/tp-admision`.
- Hay 12 Run IDs identificados en registros locales. En la captura autenticada de W&B se reconocen
  **11** de ellos mediante los identificadores abreviados; sus estados aparecen como `Finished` o `Failed`.
  El ID de v30 (`3466b67b56f94861`) no figura en esa vista; existe una fila adicional de
  verificación cuyo identificador completo no fue recuperado.
- Los estados de esas once corridas provienen de **observación de interfaz del 15/11/2026**, no de
  acceso de esta auditoría a una API autenticada. No se debe extrapolar su estado a v30.
- No se descargaron pesos ni se usaron secretos.
