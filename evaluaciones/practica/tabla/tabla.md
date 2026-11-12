# Evaluación completa: casos resueltos (Parte 4)

> Puntuado sobre los 50 casos de práctica solo para reportar: la práctica no se usó para seleccionar prompt, checkpoint ni hiperparámetros.

| Configuración | original | b | c | Checkpoint | Prompt |
|---|---|---|---|---|---|
| sin_entrenar | 10/50 | 10/50 | 7/50 | sin entrenar | baseline v1 (944ae7f6f52d) |
| solo_prompt | 14/50 | 16/50 | 12/50 | sin entrenar | procedimiento-v1 v1 (8c692c2c6785) |
| solo_entrenamiento | 11/50 | 13/50 | 12/50 | FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final | baseline v1 (944ae7f6f52d) |
| ambos | 30/50 | 29/50 | 33/50 | FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final | procedimiento-v1 v1 (8c692c2c6785) |
| *Referencia de la cátedra: sin entrenar* | 12/50 | 15/50 | 12/50 | (cátedra) | (cátedra) |
| *Referencia de la cátedra: fine-tuning supervisado simple (datos perfectos de la cátedra)* | 25/50 | 24/50 | 25/50 | (cátedra) | (cátedra) |

Las filas de referencia son de la cátedra (Gemma 4 E4B; mission.md, «Referencia de la cátedra»), no corridas nuestras.
«—» = no hay número: ver «Faltantes». Ningún valor se completa a mano.

## Costo y tiempo por caso

| Configuración | Normas | Parcial medio | Caracteres de contexto/caso | Segundos/caso | Llamadas/caso | Búsquedas/caso | Sin decisión | Hardware |
|---|---|---|---|---|---|---|---|---|
| sin_entrenar | original | 0,831 | 19.954 | 16,07 | 4,9 | 1,34 | 0 | 2× Tesla T4 |
| sin_entrenar | b | 0,820 | 19.838 | 15,44 | 4,9 | 1,34 | 0 | 2× Tesla T4 |
| sin_entrenar | c | 0,816 | 19.733 | 15,57 | 4,9 | 1,32 | 0 | 2× Tesla T4 |
| solo_prompt | original | 0,836 | 51.982 | 24,86 | 6,6 | 3,50 | 0 | 2× Tesla T4 |
| solo_prompt | b | 0,856 | 54.577 | 26,61 | 6,8 | 3,62 | 0 | 2× Tesla T4 |
| solo_prompt | c | 0,822 | 50.928 | 24,43 | 6,6 | 3,46 | 0 | 2× Tesla T4 |
| solo_entrenamiento | original | 0,818 | 28.144 | 18,75 | 5,4 | 1,88 | 0 | 2× Tesla T4 |
| solo_entrenamiento | b | 0,825 | 28.921 | 18,91 | 5,5 | 1,94 | 0 | 2× Tesla T4 |
| solo_entrenamiento | c | 0,820 | 30.226 | 19,73 | 5,6 | 2,04 | 0 | 2× Tesla T4 |
| ambos | original | 0,951 | 51.088 | 24,94 | 6,6 | 3,10 | 0 | 2× Tesla T4 |
| ambos | b | 0,947 | 51.088 | 24,68 | 6,6 | 3,10 | 0 | 2× Tesla T4 |
| ambos | c | 0,956 | 51.088 | 24,68 | 6,6 | 3,10 | 0 | 2× Tesla T4 |

Segundos/caso = `jugar_lote / casos` del kit (sin cargar el modelo); con lotes los casos corren a la vez. Resultados en T4 no son comparables con A100.

## Por control

| Configuración | Normas | registro | numero_documento | box_antes_del_registro | circuito | plazo_circuito_min | cobertura | entidad_a_notificar | aviso_al_voluntariado | verificacion |
|---|---|---|---|---|---|---|---|---|---|---|
| sin_entrenar | original | 0,76 | 1,00 | 0,72 | 0,98 | 0,92 | 0,80 | 0,82 | 0,72 | 0,76 |
| sin_entrenar | b | 0,78 | 1,00 | 0,72 | 0,98 | 0,92 | 0,80 | 0,72 | 0,72 | 0,74 |
| sin_entrenar | c | 0,68 | 1,00 | 0,72 | 0,98 | 0,92 | 0,76 | 0,80 | 0,72 | 0,76 |
| solo_prompt | original | 0,82 | 1,00 | 0,90 | 1,00 | 0,98 | 0,78 | 0,72 | 0,72 | 0,60 |
| solo_prompt | b | 0,82 | 1,00 | 0,96 | 1,00 | 1,00 | 0,82 | 0,72 | 0,72 | 0,66 |
| solo_prompt | c | 0,72 | 1,00 | 0,94 | 1,00 | 0,96 | 0,74 | 0,72 | 0,72 | 0,60 |
| solo_entrenamiento | original | 0,84 | 1,00 | 0,80 | 0,78 | 0,66 | 0,90 | 0,78 | 0,70 | 0,90 |
| solo_entrenamiento | b | 0,90 | 1,00 | 0,74 | 0,78 | 0,68 | 0,90 | 0,86 | 0,68 | 0,88 |
| solo_entrenamiento | c | 0,78 | 1,00 | 0,90 | 0,80 | 0,70 | 0,80 | 0,82 | 0,70 | 0,88 |
| ambos | original | 1,00 | 1,00 | 0,98 | 1,00 | 1,00 | 0,92 | 0,94 | 0,72 | 1,00 |
| ambos | b | 0,98 | 1,00 | 0,96 | 1,00 | 1,00 | 0,94 | 0,92 | 0,72 | 1,00 |
| ambos | c | 0,98 | 1,00 | 1,00 | 1,00 | 1,00 | 0,90 | 1,00 | 0,72 | 1,00 |

## Procedencia

- **sin_entrenar × original**: commit `f4d73001f7e74407d42ecee6c22c201dd14cf24e` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-10T22:07:00+0000 → 2026-11-10T22:20:46+0000.
  - Trazas `evaluaciones/practica/v25-sin-entrenar/sin_entrenar-original/salida/trazas.jsonl` (sha256 `05c52625a43c6ee0404e8c59a25aa379bab781450e80556f33cf136a06611177`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-actual/sin_entrenar-original --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/actual.json --salida /kaggle/working/corridas/evaluacion-actual` (plan sha256 `1bb43a4122f143d84c449e2b8a22c03d996e0b61ba7b36af0c39e7851873cd6d`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v25-sin-entrenar/sin_entrenar-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla/corridas/sin_entrenar-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **sin_entrenar × b**: commit `f4d73001f7e74407d42ecee6c22c201dd14cf24e` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-10T22:20:49+0000 → 2026-11-10T22:34:04+0000.
  - Trazas `evaluaciones/practica/v25-sin-entrenar/sin_entrenar-b/salida/trazas.jsonl` (sha256 `626ce4e95e0a7437dfef741d22749e3557297f443431181f960e4aeadb6203a4`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json python infra/kaggle_baseline.py --variante b --cantidad 50 --salida /kaggle/working/corridas/evaluacion-actual/sin_entrenar-b --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/actual.json --salida /kaggle/working/corridas/evaluacion-actual` (plan sha256 `1bb43a4122f143d84c449e2b8a22c03d996e0b61ba7b36af0c39e7851873cd6d`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v25-sin-entrenar/sin_entrenar-b/salida/trazas.jsonl --esperadas practica/esperadas-b.jsonl --salida evaluaciones/practica/tabla/corridas/sin_entrenar-b`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **sin_entrenar × c**: commit `f4d73001f7e74407d42ecee6c22c201dd14cf24e` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-10T22:34:06+0000 → 2026-11-10T22:47:27+0000.
  - Trazas `evaluaciones/practica/v25-sin-entrenar/sin_entrenar-c/salida/trazas.jsonl` (sha256 `53383fbdc4bfe13ee542d6682fc7e81654f612705a0851c8a5e70bdfca4c5d03`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json python infra/kaggle_baseline.py --variante c --cantidad 50 --salida /kaggle/working/corridas/evaluacion-actual/sin_entrenar-c --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/actual.json --salida /kaggle/working/corridas/evaluacion-actual` (plan sha256 `1bb43a4122f143d84c449e2b8a22c03d996e0b61ba7b36af0c39e7851873cd6d`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v25-sin-entrenar/sin_entrenar-c/salida/trazas.jsonl --esperadas practica/esperadas-c.jsonl --salida evaluaciones/practica/tabla/corridas/sin_entrenar-c`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_prompt × original**: commit `a5745f034bf2d025fee8c36477ea5b0dd3564367` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-11T02:35:54+0000 → 2026-11-11T02:59:07+0000.
  - Trazas `evaluaciones/practica/v27-solo-prompt/solo_prompt-original/salida/trazas.jsonl` (sha256 `9af8ce2630e6c03a79f1b24f1c34320f7dfb6c50915aee358c54da587679c7ae`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-solo-prompt/solo_prompt-original --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/solo-prompt.json --salida /kaggle/working/corridas/evaluacion-solo-prompt` (plan sha256 `84d6f2be1d9d0f20ab35a60db5b469fdaac443659c7d0840536e9db31d58e938`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v27-solo-prompt/solo_prompt-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla/corridas/solo_prompt-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_prompt × b**: commit `a5745f034bf2d025fee8c36477ea5b0dd3564367` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-11T02:59:10+0000 → 2026-11-11T03:22:08+0000.
  - Trazas `evaluaciones/practica/v27-solo-prompt/solo_prompt-b/salida/trazas.jsonl` (sha256 `0a20271e851cea88623f847f0c351fdb0730adcc4185bb7ad3f4a80755443bbf`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json python infra/kaggle_baseline.py --variante b --cantidad 50 --salida /kaggle/working/corridas/evaluacion-solo-prompt/solo_prompt-b --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/solo-prompt.json --salida /kaggle/working/corridas/evaluacion-solo-prompt` (plan sha256 `84d6f2be1d9d0f20ab35a60db5b469fdaac443659c7d0840536e9db31d58e938`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v27-solo-prompt/solo_prompt-b/salida/trazas.jsonl --esperadas practica/esperadas-b.jsonl --salida evaluaciones/practica/tabla/corridas/solo_prompt-b`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_prompt × c**: commit `a5745f034bf2d025fee8c36477ea5b0dd3564367` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint sin entrenar; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-11T03:22:13+0000 → 2026-11-11T03:43:16+0000.
  - Trazas `evaluaciones/practica/v27-solo-prompt/solo_prompt-c/salida/trazas.jsonl` (sha256 `b9b4e5e413503a6336ac64e889a5ead9c6569c0e59e8c4434cc2b67745c84fdb`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json python infra/kaggle_baseline.py --variante c --cantidad 50 --salida /kaggle/working/corridas/evaluacion-solo-prompt/solo_prompt-c --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/solo-prompt.json --salida /kaggle/working/corridas/evaluacion-solo-prompt` (plan sha256 `84d6f2be1d9d0f20ab35a60db5b469fdaac443659c7d0840536e9db31d58e938`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v27-solo-prompt/solo_prompt-c/salida/trazas.jsonl --esperadas practica/esperadas-c.jsonl --salida evaluaciones/practica/tabla/corridas/solo_prompt-c`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_entrenamiento × original**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-12T12:34:51+0000 → 2026-11-12T12:52:52+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/solo_entrenamiento-original/salida/trazas.jsonl` (sha256 `5c24a9d4f03c2376e5c72c2c2bcb78407e31517eb74a5295e18c6a78b9e69f44`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/solo_entrenamiento-original --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/solo_entrenamiento-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla/corridas/solo_entrenamiento-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_entrenamiento × b**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-12T12:52:55+0000 → 2026-11-12T13:09:33+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/solo_entrenamiento-b/salida/trazas.jsonl` (sha256 `f279e93842dcc354cc690c4751e105414d7759c55f0601121605899ec3eb9615`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante b --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/solo_entrenamiento-b --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/solo_entrenamiento-b/salida/trazas.jsonl --esperadas practica/esperadas-b.jsonl --salida evaluaciones/practica/tabla/corridas/solo_entrenamiento-b`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **solo_entrenamiento × c**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt baseline v1, `configuraciones/prompts/baseline.json`, sha256 `944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474`; buscador `baseline`; lote 4, float16; 2026-11-12T13:09:38+0000 → 2026-11-12T13:26:46+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/solo_entrenamiento-c/salida/trazas.jsonl` (sha256 `cd353d6e9267d038ef8e126f3727f0fbf6227007cb1d1fe556faf2448cb3042b`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/baseline.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante c --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/solo_entrenamiento-c --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/solo_entrenamiento-c/salida/trazas.jsonl --esperadas practica/esperadas-c.jsonl --salida evaluaciones/practica/tabla/corridas/solo_entrenamiento-c`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **ambos × original**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-12T13:26:49+0000 → 2026-11-12T13:48:02+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/ambos-original/salida/trazas.jsonl` (sha256 `b947a977935802448d87649a455a7d7ed748d9ce1bb7b88f6545c517912affd3`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante original --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/ambos-original --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/ambos-original/salida/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida evaluaciones/practica/tabla/corridas/ambos-original`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **ambos × b**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-12T13:48:05+0000 → 2026-11-12T14:09:04+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/ambos-b/salida/trazas.jsonl` (sha256 `37def773ce2497a191c7fed94633e4417dbe0afdd7d55c96e72cd0b14496a8d8`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante b --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/ambos-b --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/ambos-b/salida/trazas.jsonl --esperadas practica/esperadas-b.jsonl --salida evaluaciones/practica/tabla/corridas/ambos-b`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega
- **ambos × c**: commit `3adf12e117dee7a1a0c270e59a8d97e1499aaf83` (sucio=False); modelo `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`; checkpoint FranCalveyra/tp-admision@dc41674a8a2561a724884be2170ff6808efa6e96/final; prompt procedimiento-v1 v1, `configuraciones/prompts/procedimiento-v1.json`, sha256 `8c692c2c67850f1448b5f4ecbda6037713110983c41b0e9da933612d2291ecdd`; buscador `baseline`; lote 4, float16; 2026-11-12T14:09:07+0000 → 2026-11-12T14:30:06+0000.
  - Trazas `evaluaciones/practica/v34-entrenado/ambos-c/salida/trazas.jsonl` (sha256 `96b031d53933d3fb5301744fcf1c4ce220704913944b9e1419fd999b66750da4`).
  - Jugar (reconstruido de runner.json): `MODELO_LOTE=4 BUSCADOR_VARIANTE=baseline PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json ADAPTER_REPO=FranCalveyra/tp-admision ADAPTER_REVISION=dc41674a8a2561a724884be2170ff6808efa6e96 ADAPTER_SUBCARPETA=final python infra/kaggle_baseline.py --variante c --cantidad 50 --salida /kaggle/working/corridas/evaluacion-entrenado/ambos-c --dtype float16 --device-map auto`
  - Plan: `python infra/kaggle_evaluacion.py --plan configuraciones/evaluacion/entrenado.json --salida /kaggle/working/corridas/evaluacion-entrenado` (plan sha256 `d2f31ff5529b3e8094abc285df1a7fd52eea5f5c773826035580a5901a5c3d40`)
  - Puntuar (equivalente): `python -m admision.correr puntuar --trazas evaluaciones/practica/v34-entrenado/ambos-c/salida/trazas.jsonl --esperadas practica/esperadas-c.jsonl --salida evaluaciones/practica/tabla/corridas/ambos-c`
  - Aviso: no corrió en A100: tiempos y resultados no comparables con la entrega

## Faltantes

- Ninguno.

Tabla generada 2026-11-12T16:49:44-0300 con `python -m scripts.tabla_evaluacion --evaluacion evaluaciones/practica/v25-sin-entrenar --evaluacion evaluaciones/practica/v27-solo-prompt --evaluacion evaluaciones/practica/v34-entrenado --salida evaluaciones/practica/tabla` (commit `f24960622867861aee741d76ffad792b9767be54`, sucio=False).
