# Entrenador SFT LoRA: implementado y validado con humo en T4

**La infraestructura de #3 está implementada. No está validada en A100-40GB/Modal ni con datos reales.**
El humo sintético en Kaggle probó backward, checkpoints HF, W&B y reanudación desde un checkpoint publicado.
El siguiente paso propio es validar la ruta A100/Modal con autorización; datos/curación/prompt corresponden
al equipo. No hace falta agregar otro entrenador para hacer el handoff a Martu.

## Criterios de #3 y evidencia

| Criterio | Implementado / CPU con dobles | Dependencia real / GPU |
|---|---|---|
| SFT LoRA para A100-40GB | `entrenamiento.py`, defaults bf16/una GPU, runner Modal A100-40GB | LoRA/backward en 2 T4 fp16; **A100 sin validar** |
| Conversaciones, configuración y semilla | snapshots/hash en preparación, tokenización y `corrida.json`; validación antes de cargar pesos | tokenizador real CPU validado; humo T4 con semilla/config fijadas |
| Cada checkpoint HF con revisión | repo privado obligatorio, recibo `publicacion-hf.json`, índice `publicaciones.json`; fallar si upload falla | v6 publicó 2/4/6/8/final con commits inmutables |
| W&B con linaje | `wandb.init(config=auditoria)` incluye hashes de entradas, config/semilla, LoRA/política/código; `run.log` agrega checkpoint/revisión | v6/v7 online, mismo run_id; comprobado en artefactos locales, sin consultar W&B en esta auditoría |
| Retomar tras corte | exige optimizer/scheduler/RNG/Trainer/adapter/manifiestos; fp16 exige scaler; Modal confirma Volume por checkpoint | v7 descargó checkpoint-2 HF y continuó pasos 3–8; **no equivale a corte abrupto real en Modal** |
| Martu puede operarlo | comandos y contratos debajo, validación CPU disponible | falta que lo use con datos/prompt acordados; no se ejecutó ese experimento |

**Artefactos reales locales, no versionados:** `corridas/kaggle/humo-entrena-1-v6` y `humo-entrena-2-v7`:
ambos `runner.json` dicen `terminada`, `returncode=0`; `estado.json` dice `finalizada`, mismo
`run_id=b9088ec869814b35`. Reanudación desde `checkpoint-2`, revisión
`f11b0814564f09e2ff788937db3fba111317f149`; v7 publica nuevos 4/6/8/final. La curva es cercana, no idéntica bit a bit
(ver [bitácora](progreso.md)). `humo-solo-texto` terminó y midió 15,675 GB pico sumados en 2 T4 con datos cortos:
**no demuestra** ajuste de conversaciones reales en A100. Sin acceso a estos artefactos, consultar la bitácora;
reproducirlos consume GPU y requiere autorización.

## Handoff: entradas, salidas y permisos

- **Datos:** conversaciones v1 curadas, con prompt acordado. Las trazas crudas necesitan el
  [conversor](conversaciones-entrenamiento.md#conversor-cpu-de-trazas-implementado); convertir no cura ni certifica procedencia.
- **Config:** copiar y revisar `configuraciones/sft-ejemplo.json`: modelo/revisión inmutable, semilla,
  learning rate, epochs, batch/acumulación, límite de secuencia, repo HF y entidad/proyecto W&B.
  Son valores ilustrativos, no hiperparámetros elegidos para el modelo final. No contiene secretos.
- **Preparación:** `conversaciones.jsonl`, `configuracion.json`, `manifiesto.json`.
  **Tokenización:** `tokens.jsonl`, `manifiesto-tokens.json` (sin truncar; assistant-only, contexto/padding `-100`).
- **Entrenamiento:** `entradas/`, `corrida.json`, `estado.json`, `checkpoint-N/`, `final/`, logs W&B locales;
  con publicación, índice/recibos HF. `final/` es adapter, **no checkpoint reanudable completo**.
- **Accesos:** HF permite lectura del Gemma gateado si la cuenta aceptó licencia; publicar requiere escritura
  en `hf_repo` privado existente. W&B requiere acceso a `wandb_entity/wandb_project`.
  Variables: `HF_TOKEN`, `WANDB_API_KEY`. Configuralas por entorno/Secrets, nunca por argumentos o chat.
  No se carga `.env` automáticamente. El token Read no habilita uploads.

### 1. CPU offline, sin GPU ni escrituras cloud

En el entorno ya instalado (no ejecutar sync para estos checks):

```bash
rtk proxy .venv/bin/python -m scripts.validar_conversaciones datos/conversaciones-curadas.jsonl
rtk proxy mkdir -p corridas
rtk proxy .venv/bin/python -m scripts.preparar_entrenamiento \
  --datos datos/conversaciones-curadas.jsonl --configuracion configuraciones/sft-ejemplo.json \
  --salida corridas/sft-preparacion-001
```

Las rutas `datos/...` son entradas que debe proveer el equipo, no archivos finales ya disponibles.
Las salidas deben ser nuevas. Guardan bytes exactos, hashes, config/semilla y estado Git. Con `dirty=true`,
el commit no reconstruye los cambios: no reportar esa corrida como reproducible solo por su commit.
Versionar código y volver a preparar requiere autorización de commit, no la da este documento.

### 2. CPU con tokenizador, sin pesos

```bash
rtk proxy .venv/bin/python -m scripts.tokenizar_entrenamiento \
  --preparacion corridas/sft-preparacion-001 --salida corridas/sft-tokens-001
```

Requiere entorno con grupo `tokenizacion` y tokenizer fijado disponible; puede descargar archivos de tokenizer
si no están en caché y pedir HF. No se ejecutó aquí ni se autoriza una descarga por mostrar el comando.
Ver [instalación/tokenización y prueba real CPU](tokenizacion-entrenamiento.md). No recorta secuencias largas.

### 3. GPU Linux ya provisionada: ejecutar SOLO tras autorización

```bash
rtk proxy .venv/bin/python -m scripts.entrenar \
  --preparacion corridas/sft-preparacion-001 --tokens corridas/sft-tokens-001 \
  --salida corridas/sft-entrenamiento-001 --online --publicar
```

**Carga pesos, usa GPU y escribe W&B/HF.** El entorno Linux necesita dependencias fijadas de entrenamiento
(ver `uv.lock` y `requirements-entrenamiento.txt`); no instalar ni construir como paso implícito.
Defaults: `--precision bf16 --dispositivos unico --guardar-cada 50`, objetivo A100-40GB.
Sin `--online --publicar`, W&B queda offline y HF no recibe checkpoints: útil para ensayos, no cumple el
registro cloud de la entrega. Una falla de subida falla la corrida, no se oculta.

Para T4 x2 **de desarrollo**, agregar `--precision fp16 --dispositivos auto`; el humo usó `--guardar-cada 2`.
LoRA apunta solo a atención del modelo de lenguaje (`LORA_OBJETIVO`), r=8/alpha16/dropout0; carga sin torres
visión/audio. Eso evita módulos no soportados y reduce VRAM, pero no garantiza el ajuste final en A100.

### 4. Reanudar: mismo código, entradas, modo y política

```bash
rtk proxy .venv/bin/python -m scripts.entrenar \
  --preparacion corridas/sft-preparacion-001 --tokens corridas/sft-tokens-001 \
  --salida corridas/sft-entrenamiento-001 \
  --reanudar corridas/sft-entrenamiento-001/checkpoint-50 --online --publicar
```

Elegir un checkpoint **completo confirmado**; el nombre es un ejemplo, no una garantía de existencia.
Repetir precisión/dispositivos/cadencia originales si no fueron defaults. Conserva run_id; online usa W&B
`resume="must"`. Offline produce segmentos locales que hay que conciliar. No promete recuperar pasos
posteriores al último checkpoint íntegro, ni salvar una escritura parcial. La CLI directa no descarga por sí sola
un checkpoint HF: para recuperación desde HF usar [runner Kaggle](kaggle.md#entrenamiento-sft-humo-en-kaggle),
que fija revisión y valida integridad, o la ruta Volume de [Modal](entrenamiento-modal.md).

## Lo propio que falta validar para cierre completo

1. Humo autorizado en **A100-40GB/Modal**: cargar/backward, confirmar checkpoint/Volume y publicación HF/W&B.
2. Corte y recuperación reales en esa ruta: registrar último checkpoint completo, reanudar y conservar evidencia.
3. Entregar evidencia a Martu; después el entrenamiento real depende de datos/prompt del equipo, no de otro trainer.

Modal sigue con bloqueo de facturación según la bitácora. `hf-write` es un Secret propuesto **solo para Modal**,
no creado/verificado; no confundir con `HF_WRITE_TOKEN` ya usado en Kaggle. Cargar un adapter en inferencia
**sí está implementado**, pero el adapter final/revisión aún debe fijarse cuando exista; ver `configuracion.py`.
Accesos de cátedra pertenecen a #16, no invalidan el humo de publicación ya registrado.

**Gotcha del humo Modal:** su entrypoint usa cadencia50, no expone `--guardar-cada`. El humo habitual de 8 pasos
no garantiza un checkpoint-50. Para el ensayo futuro se puede preparar humo de 4 conversaciones×20épocas=80 pasos
(`scripts.datos_humo --registros 4 --epochs 20`) y elegir un checkpoint-50 confirmado; no es un dataset real ni se
ejecutó. Antes hay que acordar recursos/timeout/costo. Alternativamente la CLI directa sí permite cadencia2.

## Verificación de esta auditoría CPU

```bash
rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest test_entrenamiento test_persistencia test_kaggle_entrenamiento test_cli_entrenar test_preparacion test_tokenizacion test_conversion_trazas -q
```

2026-11-05: **52 ejecutados, 51 pasan, 1 opcional omitido**. También se verificaron `--help` de entrenar,
preparar y tokenizar sin cargar dependencias pesadas. No se agregaron tests en seams nuevas ni se hizo GPU,
instalación, cloud o build. Los tests con dobles no demuestran CUDA/PEFT/Modal reales.
