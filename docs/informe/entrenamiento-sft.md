# Entrenador SFT LoRA: implementado y validado con humo en T4

**La infraestructura genérica de #3 está implementada y tiene humo/reanudación HF reales en T4.**
Desde la decisión del 2026-11-05, **Modal está descartado definitivamente**: no se incorporará tarjeta.
Kaggle es la ruta operativa de desarrollo. La entrega sigue exigiendo A100-40GB, aún sin validar y por una
vía alternativa **a determinar**. Datos/curación/prompt finales corresponden al equipo.

## Criterios de #3 y evidencia

| Criterio | Implementado / CPU con dobles | Dependencia real / GPU |
|---|---|---|
| SFT LoRA para A100-40GB | `entrenamiento.py`, defaults bf16/una GPU para hardware final; Modal conservado como legacy | LoRA/backward en 2 T4 fp16; **A100 sin validar** |
| Conversaciones, configuración y semilla | snapshots/hash en preparación, tokenización y `corrida.json`; validación antes de cargar pesos | tokenizador real CPU validado; humo T4 con semilla/config fijadas |
| Cada checkpoint HF con revisión | repo privado obligatorio, recibo `publicacion-hf.json`, índice `publicaciones.json`; fallar si upload falla | v6 publicó 2/4/6/8/final con commits inmutables |
| W&B con linaje | `wandb.init(config=auditoria)` incluye hashes de entradas, config/semilla, LoRA/política/código; `run.log` agrega checkpoint/revisión | v6/v7 online, mismo run_id; comprobado en artefactos locales, sin consultar W&B en esta auditoría |
| Retomar tras corte | exige optimizer/scheduler/RNG/Trainer/adapter/manifiestos; fp16 exige scaler; recuperación HF con revisión fijada en Kaggle | v7 descargó checkpoint-2 HF y continuó pasos 3–8; valida recuperar el último checkpoint completo, no pasos posteriores o escrituras parciales |
| Martu puede operarlo | comandos y contratos debajo, validación CPU disponible | falta que lo use con datos/prompt acordados; no se ejecutó ese experimento |

**Artefactos reales, versionados:** `evaluaciones/recursos/humo-entrena-1-v6` y `humo-entrena-2-v7`:
ambos `runner.json` dicen `terminada`, `returncode=0`; `estado.json` dice `finalizada`, mismo
`run_id=b9088ec869814b35`. Reanudación desde `checkpoint-2`, revisión
`f11b0814564f09e2ff788937db3fba111317f149`; v7 publica nuevos 4/6/8/final. La curva es cercana, no idéntica bit a bit
(ver [bitácora](../progreso.md)). `humo-solo-texto` terminó y midió 15,675 GB pico sumados en 2 T4 con datos cortos:
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
.venv/bin/python -m scripts.validar_conversaciones datos/conversaciones-curadas.jsonl
mkdir -p corridas
.venv/bin/python -m scripts.preparar_entrenamiento \
  --datos datos/conversaciones-curadas.jsonl --configuracion configuraciones/sft-ejemplo.json \
  --salida corridas/sft-preparacion-001
```

Las rutas `datos/...` son entradas que debe proveer el equipo, no archivos finales ya disponibles.
Las salidas deben ser nuevas. Guardan bytes exactos, hashes, config/semilla y estado Git. Con `dirty=true`,
el commit no reconstruye los cambios: no reportar esa corrida como reproducible solo por su commit.
Versionar código y volver a preparar requiere autorización de commit, no la da este documento.

### 2. CPU con tokenizador, sin pesos

```bash
.venv/bin/python -m scripts.tokenizar_entrenamiento \
  --preparacion corridas/sft-preparacion-001 --salida corridas/sft-tokens-001
```

Requiere entorno con grupo `tokenizacion` y tokenizer fijado disponible; puede descargar archivos de tokenizer
si no están en caché y pedir HF. No se ejecutó aquí ni se autoriza una descarga por mostrar el comando.
Ver [instalación/tokenización y prueba real CPU](tokenizacion-entrenamiento.md). No recorta secuencias largas.

### 3. GPU Linux ya provisionada: ejecutar SOLO tras autorización

```bash
.venv/bin/python -m scripts.entrenar \
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

**Pérdida solo en posiciones supervisadas.** `perdida_supervisada` pide al modelo logits únicamente donde la
posición predice un token del asistente (`logits_to_keep` de Transformers, como lista de `int`: accelerate
mueve los tensores del forward a la primera GPU y Gemma corta en la última, Kaggle v29/v31). Con el vocabulario de Gemma
(262k) y ~3.300 tokens por conversación, los logits completos en fp32 no entran en una T4: el primer SFT real
(Kaggle v28) cayó por OOM en el paso 0 dentro de `ForCausalLMLoss`. Solo ~10 % de los tokens está supervisado.
La pérdida y los gradientes son idénticos a la pérdida completa: verificado con torch real en CPU sobre un modelo
de juguete con softcapping, con y sin `num_items_in_batch`. Los tests del repo cubren el desplazamiento con dobles.

### 4. Reanudar: mismo código, entradas, modo y política

```bash
.venv/bin/python -m scripts.entrenar \
  --preparacion corridas/sft-preparacion-001 --tokens corridas/sft-tokens-001 \
  --salida corridas/sft-entrenamiento-001 \
  --reanudar corridas/sft-entrenamiento-001/checkpoint-50 --online --publicar
```

Elegir un checkpoint **completo confirmado**; el nombre es un ejemplo, no una garantía de existencia.
Repetir precisión/dispositivos/cadencia originales si no fueron defaults. Conserva run_id; online usa W&B
`resume="must"`. Offline produce segmentos locales que hay que conciliar. No promete recuperar pasos
posteriores al último checkpoint íntegro, ni salvar una escritura parcial. La CLI directa no descarga por sí sola
un checkpoint HF: para recuperación desde HF usar [runner Kaggle](../guias/kaggle.md#entrenamiento-sft-humo-en-kaggle),
que fija revisión y valida integridad. [Modal](../historico/entrenamiento-modal.md) queda solo como referencia legacy, no usar.

### 5. Seguir entrenando un adapter publicado (`adapter_inicial`)

Con `adapter_inicial` (`repo`, `revision` de 40 hex, `subcarpeta`) en la configuración, el entrenamiento carga
ese adapter entrenable (mismo r y módulos con los que se publicó) en vez de crear un LoRA nuevo, y falla si fue
entrenado sobre otra base o revisión. Se usa para el aviso al voluntariado (#15): el adapter de `v3-procedimiento`
nunca responde `true` ([análisis de errores](analisis-errores.md), pasada 7).
`configuraciones/sft-aviso.json` parte del adapter de la entrega con learning rate 1e-4 y 1 época sobre
`datos/conversaciones/v4-aviso`, armado con `scripts.subconjunto_aviso`: todos los positivos, todos los negativos
del mismo rango de edad (umbral = la menor edad entre los positivos, sale de los datos) y una muestra fija del
resto. Publica en el mismo repo HF: `checkpoint-N` y `final` quedan con revisiones nuevas, y las anteriores
siguen accesibles por su revisión (la entrega fija la suya en `configuracion.py`).

**Resultado (Kaggle v36, 2026-11-12): no se adoptó.** 80 pasos, ~88 s/paso (7.029 s), 20,29 GB de VRAM pico,
W&B `6d8edb168beb4e9b`, `final` `22ee5596c8d3014bd86caaca6a87034d06e1d4e8`. En el conjunto de desarrollo bajó de 26 a
23 resueltos: aprendió a decir `true` sin la condición ([pasada 8](analisis-errores.md)). Validó en GPU que
`adapter_inicial` carga el adapter publicado (pérdida inicial 0,0069, no ~0,44).

### 6. Peso en el token del aviso (`peso_aviso`)

Con `peso_aviso` (> 1) en la configuración, la pérdida multiplica por ese valor la del token del valor
(`true`/`false`) de `aviso_al_voluntariado` que registra el asistente; el resto pesa 1 y se normaliza igual que
sin pesos. `pesos_del_aviso` ubica ese token decodificando el contexto anterior (`"aviso_al_voluntariado":`), sin
suponer cómo tokeniza la clave, y **falla antes de cargar los pesos** si no lo encuentra exactamente una vez en una
conversación. Motivo ([pasada 8](analisis-errores.md)): el aviso es un token entre ~350 supervisados (1.015.072 / 2.900 en la tokenización de `v3-procedimiento`) y se diluye en
la pérdida media; subir la proporción de positivos (v36) movió el sesgo sin enseñar la regla.

`configuraciones/sft-aviso-peso.json`: parte del adapter de la entrega, `peso_aviso` 20, lr 1e-4, 1 época sobre
`datos/conversaciones/v5-aviso-peso` (1.133: 302 con aviso, 231 mayores con acompañante, 600 del resto; 27 % de
positivos contra 47,7 % de v4-aviso (302/633)). **Probado con dobles** (ubicación del token, alineación con los objetivos,
padding); la suma ponderada en torch se valida en GPU.

**Resultado (Kaggle v38, 2026-11-12/13): no se adoptó.** 142 pasos, ~83 s/paso (11.750 s), 20,29 GB de VRAM pico,
W&B `f9326b10342b4be4`, `final` `17e9fe3d8a422284d6337a9372f697377d43319e`. En el conjunto de desarrollo dio 26
resueltos, igual que v35, con aviso 30/40 (v35: 32), 1/8 positivos atrapados y 3 falsos `true`, más fallas en
cobertura, registro y orden del box ([pasada 9](analisis-errores.md)). Validó en GPU que la pérdida ponderada
entrena sin errores (la pérdida se reporta con el peso: no es comparable con la de v36). Recibos en
`evaluaciones/entrenamiento/v38-peso`.

### 7. Hechos antes de la decisión (datos `v6-aviso-datos`)

`scripts.anteponer_datos_aviso` antepone a la decisión final `Edad: N años. Acompañante: X.` y un salto de línea;
los valores se copian de la nota, el umbral no se escribe. El validador acepta ese prefijo solo en el último mensaje
(`registrar_decision`) y `extraer_json` de la cátedra lo lee. `configuraciones/sft-aviso-datos.json` es copia exacta
de `sft-aviso-peso.json` (adapter inicial `dc41674a`, lr 1e-4, `peso_aviso` 20, 1 época sobre 1.133 conversaciones).

**Resultado (Kaggle v39, 2026-11-13 09:00 a 12:20 UTC): se adoptó.** 142 pasos, 11.735 s de `train_runtime`
(3 h 15 min 35 s, ~83 s/paso en 2 T4), **20,36 GB de VRAM pico** (12,37 + 7,99 GB; reservada 20,96 GB), pérdida final
ponderada 0,038 y media 0,0398, W&B `41ec01f292e64306`, código `5a36ecf`, `final`
`770c9fac84d5718b0b2cf4c40c3451cad322ac9e`. En el dev dio 31/40 contra 26/40 de v35 ([pasada 10](analisis-errores.md)).
La pérdida es ponderada: no es comparable con v36. Recibos en `evaluaciones/entrenamiento/v39-datos`; práctica
(Kaggle v40, solo para reportar): `ambos` 36/28/36 = 100/150, en [`evaluaciones/practica/tabla-v39`](../../evaluaciones/practica/tabla-v39/tabla.md).

## Estado de cierre de infraestructura y límites restantes

Los cuatro criterios técnicos del pipeline genérico están respaldados: conversaciones/config/semilla,
checkpoints HF versionados, linaje W&B y recuperación real desde HF tienen evidencia de humo T4.
**#3 sigue pendiente de aceptación/accesos y uso efectivo por Martu**: confirmar `--help` sin pesos es
preflight, no un experimento. Ver [pasos de aceptación](../historico/handoff-equipo.md#pendientes-por-persona-y-aceptación-operativa).
No se exige una nueva prueba de corte en Modal: esa plataforma fue descartada. Esta afirmación no cierra
el issue automáticamente ni certifica entrenamiento final, calidad o ajuste en A100-40GB.

- **Fran / hardware final:** determinar una vía alternativa de A100-40GB; validar carga/backward/memoria y
  medir recursos/lotes allí antes de afirmar compatibilidad final. No hay plataforma, tarifa ni plazo confirmados.
- **Equipo:** acordar procedencia/curación y prompt antes del entrenamiento final; Martu puede usar el pipeline
  en Kaggle con sus accesos y autorización de consumo, sin esperar otro entrenador.
- **Inferencia:** la carga del adapter está implementada; fijar adapter/revisión final cuando exista.
  Accesos de cátedra quedan en #16.

Los nombres de Secrets y comandos de [Modal](../historico/entrenamiento-modal.md) son históricos: no crearlos ni ejecutarlos.
El código/dependencias legacy se conservan sin migraciones ni borrados en esta decisión.

## Verificación de esta auditoría CPU

```bash
env PYTHONPATH=tests .venv/bin/python -m unittest test_entrenamiento test_persistencia test_kaggle_entrenamiento test_cli_entrenar test_preparacion test_tokenizacion test_conversion_trazas -q
```

2026-11-05: **52 ejecutados, 51 pasan, 1 opcional omitido**. También se verificaron `--help` de entrenar,
preparar y tokenizar sin cargar dependencias pesadas. No se agregaron tests en seams nuevas ni se hizo GPU,
instalación, cloud o build. Los tests con dobles no demuestran CUDA/PEFT/Modal reales.
