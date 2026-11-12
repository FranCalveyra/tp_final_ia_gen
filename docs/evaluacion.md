# Evaluación completa: 50 casos × normas original, b y c (issue #9)

Estado (2026-11-10): **implementado y probado con dobles**; la tabla local está **validada con artefactos reales**
(v22). El runner de planes **todavía no corrió en GPU**: hace falta autorización para gastar cuota.
Esta guía es la que va a citar el ítem 4 del README de la entrega («La evaluación completa»); el README final
es del issue #16.

Son dos fases, como en el kit: **jugar** en Kaggle (GPU, sin fichas esperadas) y **puntuar + tabla** en local.

## 1. Jugar en Kaggle: un plan

El plan es un JSON versionado en [`configuraciones/evaluacion/`](../configuraciones/evaluacion/):

- [`actual.json`](../configuraciones/evaluacion/actual.json): lo que existe hoy, solo `sin_entrenar`
  (modelo base, prompt baseline) × original/b/c × 50 casos.
- [`plantilla-cuatro-filas.json`](../configuraciones/evaluacion/plantilla-cuatro-filas.json): **plantilla** de la
  tabla completa (`sin_entrenar`, `solo_prompt`, `solo_entrenamiento`, `ambos`). Tiene `"plantilla": true` y
  `<...>`: el runner la rechaza hasta que se copie con un prompt y un adapter reales.

| Campo | Nivel | Significado |
|---|---|---|
| `version` | plan | Formato, `1` |
| `cantidad`, `versiones` | plan o configuración | Casos (1–50, default 50) y normas (default `original`, `b`, `c`) |
| `lote`, `buscador` | plan | `MODELO_LOTE` y `BUSCADOR_VARIANTE` de todas las corridas |
| `etiqueta` | configuración | Fila de la tabla; única, minúsculas/dígitos/`_` |
| `prompt`, `prompt_sha256` | configuración | `PromptSpec` relativo a la raíz y su sha256 fijado |
| `adapter` | configuración | `null` (sin entrenar) o `{repo, revision, subcarpeta}`; `revision` = commit HF de 40 hex |

Antes de crear la salida o pedir el token, `infra/kaggle_evaluacion.py` valida **todo** el plan: cada prompt
carga y su sha256 interno coincide (y además coincide con `prompt_sha256`), cada adapter pasa las reglas de
`Configuracion` (revisión inmutable), etiquetas únicas, versiones y cantidades válidas, buscador conocido.
Con `--validar` solo valida (sirve en CPU local). Cada configuración × versión corre con
`infra/kaggle_baseline.correr` (el mismo runner de siempre) en `<salida>/<etiqueta>-<version>/`, con un entorno
armado desde el plan (sin adapter, `ADAPTER_*` vacías para no tomar el de la entrega): borra del entorno heredado `PROMPT_SPEC`, `MODELO_LOTE`, `BUSCADOR_VARIANTE`,
`ADAPTER_*`, `MODELO_ID` y `MODELO_REVISION` y pone los del plan. Si una corrida falla (excepción, código
distinto de 0 o error atrapado por el kit), se anota y siguen las demás. `<salida>/evaluacion.json` registra
el plan, su sha256, el comando, la máquina y el estado de cada corrida, y se reescribe después de cada una.

Celdas (notebook con **GPU T4 x2**, Internet, Secrets `HF_TOKEN` y `GITHUB_TOKEN`): las celdas 1 y 2 de
[kaggle.md](kaggle.md#cómo-correr) (clonar e instalar). Si un plan usa un adapter del repo privado, `HF_TOKEN`
debe poder leerlo. Celda 3, una sola vez y sin tocar el notebook hasta que termine:

```
!cd /kaggle/working/repo && python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/actual.json --salida /kaggle/working/corridas/evaluacion-actual
```

Celda 4: `!cd /kaggle/working/corridas && zip -qr evaluacion-actual.zip evaluacion-actual`. Conviene lanzarlo
como versión guardada (SaveAndRunAll) para que el output persista. Descargar desde el panel de archivos o con
el MCP de Kaggle (`download_notebook_output`) a una carpeta **nueva** `corridas/kaggle/vNN/`.

## 2. Puntuar y armar la tabla (local, sin GPU)

```
uv run --locked --no-sync python -m scripts.tabla_evaluacion \
    --evaluacion corridas/kaggle/vNN/evaluacion-actual --salida corridas/tablas/<fecha>-<nombre>
```

`--evaluacion` es repetible; `--corrida ETIQUETA=CARPETA` agrega una corrida suelta de `kaggle_baseline.py`
(con `runner.json`). Puntúa cada corrida con `admision.correr.puntuar` contra
`practica/esperadas-<version>.jsonl` y escribe en una carpeta nueva (si existe, falla):

- `tabla.md`: casos resueltos por configuración y versión, con las filas de referencia de la cátedra
  rotuladas como tales (12/15/12 sin entrenar; 25/24/25 SFT simple), columnas secundarias de costo y tiempo
  (parcial medio, caracteres de contexto por caso, segundos por caso, llamadas, búsquedas), desglose por
  control, procedencia de cada número y faltantes.
- `tabla.json`: lo mismo con la procedencia completa de cada celda: commit y `sucio`, modelo, checkpoint
  (`repo@revision/subcarpeta` o «sin entrenar»), prompt (nombre, versión, sha256, archivo), buscador,
  configuración, GPU/torch, inicio/fin, sha256 de trazas y esperadas, comando de jugar (reconstruido de
  `runner.json`), del plan y de puntuar.
- `corridas/<etiqueta>-<version>/`: `resultados.jsonl` y `resumen.json` del kit, caso por caso.

Una combinación sin corrida, fallida, cortada o con trazas incompletas aparece como «—» con su motivo; nunca se
completa. Una carpeta sin `runner.json` se rechaza (sin procedencia no hay número). Dos corridas para la misma
celda también se rechazan: hay que elegir una explícitamente.

**La práctica se puntúa solo para reportar**, nunca para elegir prompt, checkpoint ni hiperparámetros.

## Comprobación con artefactos reales (2026-11-10)

```
uv run --locked --no-sync python -m scripts.tabla_evaluacion \
    --corrida sin_entrenar=corridas/kaggle/v22/baseline-original-n50-l4 \
    --salida corridas/tablas/2026-11-10-v22-sin-entrenar
```

Reproduce v22 (Kaggle T4 x2, commit `4e3613a` limpio, fp16, lote 4): `sin_entrenar` × original **10/50**,
parcial 0,831, 19.954 caracteres de contexto por caso, 15,16 s/caso; `resultados.jsonl` es byte a byte igual
al puntuado antes a mano. b y c salen «— sin corrida». El runner de v22 es anterior al registro del prompt en
`runner.json`, así que la tabla dice «prompt no registrado» y lo marca como aviso. `corridas/kaggle/v24/` se
rechaza: solo tiene `trazas.jsonl`, sin `runner.json`.

## Corridas en GPU (Kaggle T4, 2026-11-10/11)

| Plan | Sesión | Fila | original | b | c | Contexto/caso |
|---|---|---|---:|---:|---:|---:|
| `actual.json` | v25 | `sin_entrenar` | 10/50 | 10/50 | 7/50 | ~19.800 |
| `solo-prompt.json` | v27 | `solo_prompt` (`procedimiento-v1`) | 14/50 | 16/50 | 12/50 | ~52.500 |

`sin_entrenar` × original reproduce v22 exacto (10/50, parcial 0,831, 19.954 caracteres). Tabla completa:
`python -m scripts.tabla_evaluacion --evaluacion corridas/kaggle/v25/evaluacion-actual --evaluacion
corridas/kaggle/v27/evaluacion-solo-prompt --salida corridas/tablas/<nueva>`. Cada sesión de una fila tardó
~40 min (baseline) y ~65 min (B) de jugar, más instalación. El prompt de `solo_prompt` se eligió en el conjunto
de desarrollo ([optimización de prompt](optimizacion-prompt.md#decisión)), no en práctica.

## Límites y costo

- **T4 ≠ A100.** Todo lo que salga de Kaggle se reporta aparte de A100 (KI-A100-001); la tabla avisa por celda.
- Evaluar en lote en GPU puede mover algún caso: si se repite una corrida y cambia, se documenta.
- `segundos/caso` es `jugar_lote / casos` del kit: no incluye cargar el modelo; con lotes los casos corren a la vez.
- **Presupuesto estimado (T4 x2, lote 4, sin entrenar):** v22 midió ~15,2 s/caso → ~13 min por versión, más
  35–150 s de carga del modelo por corrida (cada corrida es un subproceso; la primera descarga). Una fila
  (3 versiones) ≈ 40–45 min; la sesión suma ~7–20 min de instalación. Plan actual ≈ **50–65 min**; las
  cuatro filas ≈ **3 h** en una sesión. Con adapter se suma la descarga y fusión (no medida) y el modelo
  entrenado puede hacer más o menos llamadas: es una estimación, no una medición.
