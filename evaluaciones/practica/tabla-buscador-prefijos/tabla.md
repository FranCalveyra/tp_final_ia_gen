# Evaluación completa: casos resueltos (Parte 4)

> Puntuado sobre los 50 casos de práctica solo para reportar: la práctica no se usó para seleccionar prompt, checkpoint ni hiperparámetros.

| Configuración | original | b | c | Checkpoint | Prompt |
|---|---|---|---|---|---|
| sin_entrenar | 10/50 | — | — | sin entrenar | baseline v1 (944ae7f6f52d) |
| solo_prompt | — | — | — | — | — |
| solo_entrenamiento | — | — | — | — | — |
| ambos | — | — | — | — | — |
| sin_entrenar_prefijos | 6/50 | — | — | sin entrenar | baseline v1 (944ae7f6f52d) |
| *Referencia de la cátedra: sin entrenar* | 12/50 | 15/50 | 12/50 | (cátedra) | (cátedra) |
| *Referencia de la cátedra: fine-tuning supervisado simple (datos perfectos de la cátedra)* | 25/50 | 24/50 | 25/50 | (cátedra) | (cátedra) |

Las filas de referencia son de la cátedra (Gemma 4 E4B; mission.md, «Referencia de la cátedra»), no corridas nuestras.
«—» = no hay número: ver «Faltantes». Ningún valor se completa a mano.

## Costo y tiempo por caso

| Configuración | Normas | Parcial medio | Caracteres de contexto/caso | Segundos/caso | Llamadas/caso | Búsquedas/caso | Sin decisión | Hardware |
|---|---|---|---|---|---|---|---|---|
| sin_entrenar | original | 0,831 | 19.954 | 16,07 | 4,9 | 1,34 | 0 | 2× Tesla T4 |
| sin_entrenar_prefijos | original | 0,796 | 20.689 | 16,09 | 5,0 | 1,40 | 0 | 2× Tesla T4 |

Segundos/caso = `jugar_lote / casos` del kit (sin cargar el modelo); con lotes los casos corren a la vez. Resultados en T4 no son comparables con A100.

## Por control

| Configuración | Normas | registro | numero_documento | box_antes_del_registro | circuito | plazo_circuito_min | cobertura | entidad_a_notificar | aviso_al_voluntariado | verificacion |
|---|---|---|---|---|---|---|---|---|---|---|
| sin_entrenar | original | 0,76 | 1,00 | 0,72 | 0,98 | 0,92 | 0,80 | 0,82 | 0,72 | 0,76 |
| sin_entrenar_prefijos | original | 0,62 | 1,00 | 0,68 | 0,94 | 0,84 | 0,78 | 0,82 | 0,72 | 0,76 |

## Procedencia

- **sin_entrenar × original**: commit `f4d73001f7e74407d42ecee6c22c201dd14cf24e` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-10T22:07:00+0000 → 2026-11-10T22:20:46+0000.
  - Trazas `evaluaciones/practica/v25-sin-entrenar/sin_entrenar-original/salida/trazas.jsonl` (sha256 `05c52625a43c6ee0404e8c59a25aa379bab781450e80556f33cf136a06611177`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-actual/sin_entrenar-original --dtype float16 --device-map auto`
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v25-sin-entrenar/sin_entrenar-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla-buscador-prefijos/corridas/sin_entrenar-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **sin_entrenar_prefijos × original**: commit `6f6a4f5da8950101abec01a5fc322fcff9d29a4d` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `prefijos`; lote 4, float16; 2026-11-13T19:37:32+0000 → 2026-11-13T19:53:16+0000.
  - Trazas `evaluaciones/practica/v41-buscador-prefijos/sin_entrenar_prefijos-original/salida/trazas.jsonl` (sha256 `ede2f6f77996ed222c7257d57f3770d1f182aa738b3217168bf02922305e9105`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=prefijos PROMPT_SPEC=configuraciones/prompts/baseline.json python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-prefijos/sin_entrenar_prefijos-original --dtype float16 --device-map auto`
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v41-buscador-prefijos/sin_entrenar_prefijos-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla-buscador-prefijos/corridas/sin_entrenar_prefijos-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega

## Faltantes

- sin_entrenar × b: sin corrida
- sin_entrenar × c: sin corrida
- solo_prompt × original: sin corrida
- solo_prompt × b: sin corrida
- solo_prompt × c: sin corrida
- solo_entrenamiento × original: sin corrida
- solo_entrenamiento × b: sin corrida
- solo_entrenamiento × c: sin corrida
- ambos × original: sin corrida
- ambos × b: sin corrida
- ambos × c: sin corrida
- sin_entrenar_prefijos × b: sin corrida
- sin_entrenar_prefijos × c: sin corrida

Tabla generada 2026-11-13T16:56:36-0300 con `python -m scripts.tabla_evaluacion --corrida sin_entrenar=evaluaciones/practica/v25-sin-entrenar/sin_entrenar-original --corrida sin_entrenar_prefijos=evaluaciones/practica/v41-buscador-prefijos/sin_entrenar_prefijos-original --salida evaluaciones/practica/tabla-buscador-prefijos` (commit `269718bf05b5d2279dcde18fbe1c7d637f953afd`, sucio=False).
