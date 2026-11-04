# Progreso y decisiones

**Actualizado: 2026-11-04.** Hay generación real con Gemma completo **en Kaggle T4** (línea de base
y lotes), no en A100. **Todavía no hubo entrenamiento real.** Esta bitácora centraliza el estado y
enlaza las guías; no reemplaza el informe final.

## Estado de nuestra parte

| Issue | Implementado / validado | Falta para cerrar |
|---|---|---|
| #1 Baseline | **Cerrado (2026-11-04).** 50 casos × original/b/c en Kaggle T4: 10, 10 y 7 contra la referencia 12, 15 y 12 | Repetir en A100 si Modal se destraba |
| #2 Infraestructura | **Cerrado (2026-11-04).** Lock, `uv sync` con inferencia, plantilla de variables, app A100-40GB, W&B, repo HF privado | Invitar a la cátedra a HF y W&B pasó al #16 |
| #3 SFT | Validación de conversaciones, preparación, tokenización, Trainer LoRA y runner persistente; tokenizador real probado | Entrenar y probar corte/reanudación reales; publicación desde Modal y datos/prompt finales |
| #4 Lotes | Coordinador del kit y adaptador admiten lotes; orden probado con dobles. Medido en T4 (2026-11-04): lote 1 y 4 válidos, **8 y 16 fallan con error CUDA** | Medir en A100-40GB; investigar el fallo de lotes grandes en T4 |
| #5 Recursos | Infraestructura para corridas auditables | Mediciones antes/después, costos y proyección en `INFORME.md` |
| #16 Entrega compartida | Guías técnicas y esta bitácora | Informe, README final reproducible, diagrama, Loom y envío con el equipo |

El formato de conversaciones es **propuesto**, pendiente de coordinación con Eze/Yas.
El entrenamiento real también espera el prompt optimizado de Martu. Preparar código no
reemplaza esos entregables. `INFORME.md` todavía no existe.

## Evidencia disponible

### Buscador: conjunto de desarrollo, 60 consultas, normas originales

| Experimento | Completas | Recall | Precisión | Tokens/consulta | Decisión |
|---|---:|---:|---:|---:|---|
| BM25 baseline, 2 secciones | 48/60 | 0,817 | 0,467 | 618 | Default conservado |
| BM25, 4 secciones | 57/60 | 0,950 | 0,271 | 1.270 | Comparación; duplica aproximadamente el contexto |
| Sin palabras comunes, 2 secciones | 50/60 | 0,850 | 0,483 | 591 | Candidato seleccionable `sin_comunes` |
| Peso 0,25 para palabras comunes | 49/60 | 0,833 | 0,475 | 605 | No adoptado |
| Fusión palabras/caracteres, 2 secciones | 48/60 | 0,817 | 0,467 | 612 | No adoptada |

`sin_comunes` gana d022/d026/d028 pero pierde d011 contra baseline. No se ocultó esa
regresión ni se cambió silenciosamente el default. Las fallas principales son de ranking
y vocabulario, no evidencia cortada. Estos números **no miden pacientes resueltos ni test secreto**.
Detalles y reproducción: [buscador](buscador-baseline.md). Comparación por consulta:
`corridas/rag-variantes-20261101/comparacion.json` (artefacto local ignorado por Git).

### Integración y dependencias reales

- Flujo con dos pacientes sintéticos: entorno, buscador, cinco herramientas, orden del lote,
  retry JSON, nueve controles y persistencia probados. Solo la generación era guionada:
  [integración CPU](integracion-cpu.md).
- W&B online desde Modal CPU: corrida [`5f71i1i5`](https://forge.coreweave.com/wandb/francalveyra-ia-gen/tp-admision/runs/5f71i1i5),
  métrica técnica `setup_check=1`. Credencial/escritura verificadas, sin GPU ni pacientes.
- Tokenizador Gemma en revisión fijada: fixture sintético de 167 tokens, 126 supervisados;
  comprobados JSON assistant y cierre de turno, sin encabezados/contexto. Se descargaron
  cuatro archivos (~32,2 MB), **no pesos**. CLI real offline y hashes también pasaron.
- Suite repetida antes de commitear el 2026-11-02: **60 tests, 59 pasan y uno opcional omitido**,
  con `rtk proxy .venv/bin/python -m unittest discover -s tests -v`.
  El test real del tokenizador requiere su grupo/entorno; previamente pasó en su validación específica.

Comando CPU del proyecto (sin build):

```bash
rtk proxy uv run --locked --no-sync python -m unittest discover -s tests -v
```

## Decisiones y motivos

| Decisión | Motivo y costo / límite |
|---|---|
| Reutilizar `AgenteJSON` | Llegar a una primera medición sin reescribir coordinación; no significa inferencia más rápida. Hereda su default de 16 turnos, distinto de 40 llamadas máximas del entorno. |
| Secciones completas con valores YAML | No separar reglas de condiciones y valores; agrega contexto. Siempre cargar la carpeta recibida. |
| Separar modelo, buscador y Modal | Cambiar generación sin tocar herramientas; la infraestructura no decide fichas. |
| Dependencias CPU/GPU opcionales | Entorno local liviano; `package=false` evita builds del checkout. Lock y exports son reproducibles. |
| Máscaras por prefijos verificados | El template Gemma fijado no tiene bloques `generation`; falla explícitamente si el tokenizador no conserva prefijos. |
| LoRA inicial (r=8, alpha=16) | Menos parámetros entrenables que ajuste completo; hiperparámetros ilustrativos y memoria real aún no validados. |
| Volume v2 y exclusión por corrida | Persistir checkpoints y evitar colisiones; locks abandonados requieren inspección manual, no borrado automático. |
| HF personal por ahora | `FranCalveyra/tp-admision` informado por el usuario; transferencia a organización postergada explícitamente. No equivale a acceso compartido verificado. |
| W&B fuera del agente operativo | Auditoría de experimentos sin violar la prohibición de servicios externos durante inferencia. |

## Cronología resumida

### Sesiones previas: diseño y baseline

Se acordaron módulos pequeños, búsqueda de normas completas y reutilización del coordinador.
Se implementaron el buscador, adaptador y pruebas CPU; se midieron alternativas RAG sin adoptar
las que no mejoraban el candidato. El baseline de pacientes quedó pendiente.

### 2026-11-01: accesos y comprobaciones

- Modal autenticado; secreto `huggingface` confirmado por metadata. El intento A100 fue
  bloqueado por falta de medio de pago **antes de cargar el modelo**. Último estado conocido;
  no se volvió a comprobar facturación para esta bitácora.
- Primer smoke W&B falló por orden incorrecto de pasos de imagen; se corrigió y luego
  el smoke CPU online fue exitoso. No confundir imagen construida con modelo validado.
- Preparación y tokenización se desarrollaron sin entrenar. Se detectó Jinja2 faltante en
  el entorno mínimo y se agregó al grupo CPU `tokenizacion`; la comprobación real pasó.

### Trabajo posterior, al 2026-11-02

Se implementaron Trainer LoRA, collator, auditoría de checkpoints y publicación HF opt-in
en el entrenador; luego el runner Modal persistente. El runner Modal publica HF solo con `--publicar` (ver 2026-11-03, publicación desde Modal).
Se probaron con dobles y se importó la definición Modal sin crear recursos remotos.
No hay prueba real de backward, memoria, persistencia o reanudación en A100.

Trabajo previo versionado en cuatro commits:

| Commit | Unidad |
|---|---|
| `a8717df` | Entorno reproducible |
| `cc7cb4c` | Buscador y baseline del agente |
| `1e1de83` | Verificación aislada W&B |
| `3ba6f84` | Contrato y validación de conversaciones |

El 2026-11-02 se versionaron las siguientes unidades, sin ejecutar builds ni entrenamiento:

| Commit | Unidad |
|---|---|
| `53ae275` | Snapshots y configuración reproducible de corrida |
| `58b1dc9` | Tokenización assistant-only y dependencias fijadas |
| `cccf8ef` | Trainer LoRA auditable y reanudación |
| `9ba2fee` | Persistencia de corridas y checkpoints en Modal |

Los estándares compartidos y esta bitácora se guardan en un commit documental separado.
El lock se comprobó con `uv lock --check --no-build` (104 paquetes), usando caché temporal.
El escaneo estático de 25 archivos nuevos/modificados no encontró patrones de credenciales;
no reemplaza una auditoría de seguridad. `.atl/`, secretos y `corridas/` no se incluyen.
Consultar Git para confirmar el estado de publicación remoto; no se cierran issues con estos commits.

### 2026-11-03: carga de adapters en el agente

- **Decisión:** publicar solo adapters LoRA (unos MB por checkpoint) y fusionarlos en memoria
  al cargar. Se descartó publicar el modelo fusionado (~16 GB por checkpoint y un paso extra).
- **Implementado y probado con dobles:** campos `adapter_*` en `Configuracion`, carga con
  `subfolder` y una revisión inmutable, validación de base y revisión contra `adapter_config.json`,
  y `merge_and_unload`. La API de PEFT 0.21.2 se verificó leyendo el código del wheel.
- PEFT pasó al grupo `inferencia`. Lock regenerado (104 paquetes); `requirements-inferencia.txt`
  re-exportado.
- **Corrección de contrato:** `[tool.uv] default-groups = ["inferencia"]`. Antes, el `uv sync`
  de la cátedra instalaba solo `pyyaml`. En macOS no se instala nada extra (marcadores Linux)
  y las imágenes Modal siguen usando los `requirements-*.txt`. Tests: 66 corridos, 65 pasan y 1 opcional se saltea.
- `.env.example`: Fran agrega a mano las variables `ADAPTER_*` (archivo protegido para el agente).
  Guía: [modelo-modal.md](modelo-modal.md#adapter-lora-probado-con-dobles).

### 2026-11-03: publicación HF desde el runner Modal (issue #3)

- **Decisión:** opt-in, apagada por defecto. Una segunda función Modal (`ejecutar_publicando`)
  adjunta el secreto separado `hf-write` (nombre propuesto, **no creado/verificado**) en lugar de
  `huggingface`, para no depender del orden de los secretos de Modal. La función normal no lo necesita. La lógica de subida sigue solo en el entrenador; infra solo pasa `publicar`.
- **Implementado y probado con dobles:** `ejecutar_persistente(..., publicar=)`; `--publicar` exige
  `--online` y falla antes de tomar el lock o tocar el Volume. `hf_repo` es obligatorio en la
  configuración validada. Tests nuevos: modo propagado, recibos en el Volume y conservados al
  reanudar, y rechazo de cambiar el modo. Suite: 68 corridos, 67 pasan y 1 opcional se saltea.
- **Límites:** no hay subida real, secreto real ni GPU; `entrenamiento_app.py` no se importa en
  tests. El token de escritura debe poder descargar Gemma (licencia aceptada). Guía:
  [entrenamiento-modal.md](entrenamiento-modal.md#publicación-hf-opt-in-probada-con-dobles).

### 2026-11-03: métricas de recursos por lote (issues #4 y #5)

- **Decisión:** medir dentro de `ModeloLocal` solo si existe `MODELO_METRICAS`, para que la primera
  corrida GPU de la línea de base ya deje VRAM pico, tokens y segundos por lote. No pagamos una
  corrida extra solo para medir, y la cátedra no se ve afectada.
- **Implementado y probado con dobles:** JSONL por micro-lote. El runner baseline lo devuelve como
  `metricas.jsonl`. Suite: 71 corridos, 70 pasan y 1 opcional se saltea.
- **Límites:** no se midió en GPU; la VRAM es la de PyTorch, no la del dispositivo.
  Guía: [modelo-modal.md](modelo-modal.md#métricas-por-lote-probado-con-dobles).

### 2026-11-03: Kaggle como respaldo temporal de desarrollo

- **Decisión:** usar GPU gratuitas de Kaggle (2x T4) solo para desarrollo, porque Modal exige una tarjeta que
  nadie del equipo puede cargar. La entrega sigue siendo A100-40GB y sus defaults no cambian.
- **Implementado y probado con dobles:** `MODELO_DTYPE` y `MODELO_DEVICE_MAP` en `Configuracion` (validados),
  VRAM por lote sumada sobre las GPU visibles y runner `infra/kaggle_baseline.py` (no pisa salidas, token solo
  desde Kaggle Secrets, sin imprimirlo). Suite: 82 corridos, 81 pasan y 1 opcional se saltea.
- **Límites:** no se ejecutó en Kaggle ni en GPU. Los resultados, tiempos y VRAM no son comparables con A100 y
  se reportan aparte. T4 no tiene bf16 nativo (fp16 puede fallar con Gemma); el torch fijado usa CUDA 13 y
  no sirve en P100; el driver de Kaggle no está verificado. Guía: [kaggle.md](kaggle.md).
- **Primer intento real en Kaggle (commit `38aac31`):** driver 580.178.04 (CUDA 13.0), 2x T4 de 15 GB, Python 3.13.
  `pip install -r requirements-inferencia.txt` dejó `torch 2.15.0+cu132` con CUDA disponible en 2 GPU.
  La corrida `t4-original-01` (3 casos, original) **falló antes de cargar el modelo**: el torchvision de la imagen,
  compilado para torch 2.11, rompe el import de `AutoProcessor` (`operator torchvision::nms does not exist`).
  Solución: `pip uninstall -y torchvision`, agregado a la guía.
- `t4-original-02` y `03` fallaron igual, por torchaudio y luego torchcodec de la imagen (CUDA 13.0 contra 13.2).
  `t4-original-04` mostró un **bug real de la entrega**: `AutoProcessor` de Gemma 4 carga `Gemma4VideoProcessor`,
  que exige torchvision, y torchvision no está en nuestro lock. En la A100 de la cátedra también habría fallado.
- **Decisión:** `ModeloLocal` usa `AutoTokenizer`. El agente es solo texto, y así comparte con el entrenamiento
  (`tokenizacion.py`) el mismo chat template, ya validado con el tokenizer real en CPU. No se agregan
  dependencias de visión ni de audio. Tests: 82 corridos, 81 pasan y 1 opcional se saltea. Falta la corrida real.
- `t4-original-08` (commit `e5c7f7b`) mostró **otro bug de la entrega y del entrenamiento**: forzar
  `attn_implementation="sdpa"` falla porque `AutoModelForCausalLM` arma el Gemma 4 completo y su
  `Gemma4AudioModel` no soporta SDPA. **Decisión:** no forzar la atención. Verificado en
  `transformers/modeling_utils.py:1844-1878` (5.22.1): por defecto prueba SDPA en cada submodelo y usa
  eager solo donde no se soporta. El texto sigue con SDPA (`_supports_sdpa = True`). Cambio en
  `modelo.py` y `entrenamiento.py`, con tests de regresión. Las torres de audio y visión se cargan
  igual y ocupan VRAM; sacarlas es una optimización pendiente.
- **Incidente:** una celda de clonado pasaba el token de GitHub como argumento de git y, al fallar,
  lo imprimió en base64. Fran lo revocó; la guía ahora lo pasa por variables de entorno.

### 2026-11-03: primera corrida real del agente (Kaggle T4, **no comparable con A100**)

`t4-original-09`: commit `2dc0545` limpio, Gemma 4 E4B sin entrenar, fp16, `device_map=auto` en 2x T4,
torch 2.15.0+cu132, lote 1, primeros 3 casos de práctica, normas original. Puntuado local con
`admision.correr puntuar` contra `practica/esperadas-original.jsonl`. Artefactos en `corridas/kaggle/` (no versionados).

| Métrica | Valor |
|---|---|
| Resueltos | 0/3 (parcial medio 0,852: 0,889, 0,889 y 0,778) |
| Controles flojos | `registro` 0,33, `box_antes_del_registro` 0,67, `cobertura` 0,67; el resto 1,0 |
| Llamadas y búsquedas por caso | 4 y 1 |
| Cargar agente / jugar por caso | 63,6 s / 30,1 s |
| Tokens por llamada | entrada 651–1614, generados 28–145; ~8 tokens/s |
| VRAM pico (suma de GPU) | 16,3 GB (9,4 + 6,9) de 30 GB |

- 3 casos no alcanzan para comparar con la referencia (12/50 sin entrenar): es una prueba de punta a punta.
- fp16 en T4 no dio NaN ni salidas vacías en estos casos.
- Sobra VRAM para probar `MODELO_LOTE` > 1.

### 2026-11-03/04: línea de base de 50 casos (Kaggle T4, **no comparable con A100**)

**Intentos fallidos:** dos corridas interactivas de 50 casos y las primeras de b y c se cortaron sin dejar
rastro; `base-original-l4` terminó, pero sus archivos se perdieron cuando la sesión interactiva se apagó.
Desde entonces `runner.json` se escribe al arrancar (`estado: en_curso`), `runner.log` registra la salida
línea a línea y las corridas se lanzan por el MCP de Kaggle como **versión guardada** (persiste el output).

**Corrida válida:** versión 2 del notebook `franciscocalveyra/tp-admision-baseline`, lanzada con
`save_notebook` (SaveAndRunAll, `NvidiaTeslaT4`, 2x T4). Commit `a89f5b9` limpio, Gemma 4 E4B sin entrenar,
fp16, `device_map=auto`, `MODELO_LOTE=4`, 50 casos de práctica por variante, normas completas.
Puntuado local con `admision.correr puntuar` contra `practica/esperadas-{v}.jsonl`.
Artefactos en `corridas/kaggle/base-{original,b,c}-l4` (no versionados).

| Normas | Resueltos | Referencia cátedra | Parcial medio | Segundos por caso |
|---|---|---|---|---|
| original | **10/50** | 12/50 | 0,831 | 15,6 |
| b | **10/50** | 15/50 | 0,820 | 15,8 |
| c | **7/50** | 12/50 | 0,816 | 16,1 |

- Todas sin decisiones faltantes ni errores; ~4,9 llamadas y ~1,3 búsquedas por caso, ~19.800 caracteres
  de contexto por caso. VRAM pico 19,1 GB (suma de 2 GPU), ~17 tokens generados/s por lote.
- Controles más flojos en las tres: `box_antes_del_registro` y `aviso_al_voluntariado` (0,72), `registro`
  (0,68–0,78) y `verificacion` (0,74–0,76). `numero_documento` es perfecto y `circuito` está en 0,98.
- Quedamos **por debajo de la referencia** (27 contra 39 casos en total). Diferencias conocidas: fp16 en T4
  contra (probablemente) bf16 en A100, lote 4 con padding y nuestro buscador. Sin una corrida en A100 no se
  puede separar cuánto aporta cada una.
- Cargar el agente tardó 146 s en original (incluye bajar el modelo) y 34–47 s en b y c.
- **Bug de trazabilidad:** `runner.json` de estas corridas dice `lote: 1`, pero el agente usó lote 4
  (65 lotes para ~245 llamadas). El runner guardaba los defaults en vez de la configuración del entorno.
  Corregido con test: suite de 85 corridos, 84 pasan y 1 opcional se saltea.

### 2026-11-04: efecto del tamaño de lote en T4 (issue #4, **no comparable con A100**)

**Corrida:** versión 3 del notebook `franciscocalveyra/tp-admision-baseline` (SaveAndRunAll, 2x T4), commit
`11bf3d6` limpio, Gemma 4 E4B sin entrenar, fp16, `device_map=auto`, normas **original**, 50 casos de práctica.
Una celda con `for l in 1 8 16; do MODELO_LOTE=$l python infra/kaggle_baseline.py --variante original --cantidad 50 ...`
(carpetas `corridas/kaggle/lote-original-l{1,8,16}`, no versionadas). Puntuado local con `admision.correr puntuar`
contra `practica/esperadas-original.jsonl`. Cuota GPU consumida: ~2.690 s (~45 min). Una sola corrida por lote.
`runner.json` ahora registra el lote real (1, 8 y 16): el bug del 2026-11-03 quedó corregido.

| Lote | Resueltos | Parcial medio | s/caso | `construir_agente` | VRAM pico (suma / por GPU) | Tokens/s | Micro-lotes | Estado |
|---|---|---|---|---|---|---|---|---|
| 1 | **10/50** | 0,831 | **33,3** | 156 s | 16,7 GB (9,6 + 7,1) | 8,2 | 246 | ok |
| 4 (otra sesión) | **10/50** | 0,831 | **15,6** | 146 s | 19,1 GB (10,9 + 8,2) | 17,4 | 65 | ok |
| 8 | 1/50 (inválido) | 0,247 | 7,6 (inválido) | 50 s | 20,2 GB (11,5 + 8,7) | 25,3 | 28 | **cayó** |
| 16 | 0/50 (inválido) | 0,0 | 3,0 (inválido) | 40 s | 22,7 GB (12,8 + 9,8) | 34,5 | 10 | **cayó** |

- **Lotes 8 y 16 no terminaron.** `jugar_lote` falló con `AcceleratorError: CUDA error: unspecified launch failure`
  dentro del forward de Gemma 4 (`project_per_layer_inputs`/`_norm`), no con un OOM explícito. El lote 8 cayó tras
  28 micro-lotes (379 s) y dejó decisión solo en 16 casos (34 sin decisión); el 16 cayó tras 10 micro-lotes
  (152 s) y no dejó ninguna. El proceso devolvió `returncode=0` porque el kit captura el error y lo escribe en
  `tiempos.json` (`error`): **mirar siempre `error` y `sin_decision`, no solo el returncode**.
  Los s/caso y resueltos de 8 y 16 son de una corrida cortada y no sirven para comparar. Causa no diagnosticada
  (no se probó con `CUDA_LAUNCH_BLOCKING=1`); el pico por GPU (12,8 de 15 GB) no apunta a falta de memoria.
- **Lote 1 contra 4: sin regresiones por padding.** Mismos 10 casos resueltos (pev-010, 011, 012, 017, 021, 024,
  027, 030, 049, 050), cero cambios por caso, parcial idéntico, la primera respuesta del modelo es idéntica en los
  50 casos y la conversación completa es idéntica en 49 de 50 (solo difiere pev-023, sin cambiar el resultado).
  Para los 16 casos que el lote 8 alcanzó a decidir, la decisión coincide con la del lote 1 en los 16.
- **Tiempo:** pasar de lote 1 a 4 baja de 33,3 a 15,6 s/caso (2,1x), a costa de +2,4 GB de VRAM. Mismo patrón por
  micro-lote completo (s por conversación): 6,8 (1), 3,1 (4), 1,8 (8), 1,1 (16); los lotes grandes serían más
  rápidos si no fallaran. El lote 4 vino de otra sesión de Kaggle (variación de hardware posible).
- **Recomendación para desarrollo en Kaggle T4:** `MODELO_LOTE=4`. Es el mayor lote validado de punta a punta,
  da 2,1x de velocidad, no cambia resultados y deja ~10 GB libres por GPU. No usar 8 ni 16 hasta entender el
  fallo. Los defaults del código **no cambian**: la entrega en A100-40GB sigue en lote 1 hasta decidirlo con
  medición en esa GPU.
- **Limitaciones:** T4 fp16, no comparable con A100; una corrida por lote; lote 4 de otra sesión; vLLM queda fuera
  de alcance (posible trabajo futuro si hace falta más throughput). Pendiente del issue #4: repetir en A100-40GB.
- **Corrección del runner (probada con dobles):** si `salida/tiempos.json` trae `error`, el runner marca la corrida
  `fallida`, devuelve 1 y guarda `error_agente` en `runner.json`. Así un fallo como el de los lotes 8 y 16 ya no
  parece exitoso. Suite: 87 corridos, 86 pasan y 1 opcional se saltea.

### 2026-11-04: entrenamiento SFT en Kaggle T4 x2, humo (issue #3, **implementado, probado con dobles, sin GPU**)

**Qué se hizo.** El entrenador acepta `precision` (`bf16` default, `fp16`), `dispositivos` (`unico` default, `auto`) y
`guardar_cada` (50 default). Se guardan en `corrida.json` (`ejecucion`) y la política de la identidad se deriva de ellos,
con los defaults es exactamente `POLITICA`: la identidad A100/Modal no cambió. Nuevo runner `infra/kaggle_entrenamiento.py`
(reusa `scripts.entrenar`/`entrenar`, sin duplicar entrenamiento ni publicación) y `scripts/datos_humo.py` (conversaciones
sintéticas). Guía en [kaggle.md](kaggle.md#entrenamiento-sft-humo-en-kaggle).

**Decisiones y motivos.**
- Precisión/dispositivos como parámetros de `entrenar`, no campos de la configuración: el contrato de configuración es
  cerrado y se comparte con preparación y tokenización; esto es una decisión de máquina, no de datos.
- `auto` + fp16 verificado **leyendo** las ruedas fijadas (transformers 5.22.1, peft 0.21.2, accelerate 1.15.0): Trainer marca
  paralelismo de modelo por `hf_device_map` y fuerza `_n_gpu=1` (`trainer.py:459-465,507`); PEFT reenvía `hf_device_map`
  (`peft_model.py:979-986`) y sube el adapter a fp32 por defecto (`mapping_func.py:110`, `tuners_utils.py:2705-2745`);
  Accelerate solo rechaza `device_map` multi-GPU en modo distribuido (`accelerator.py:1814-1821`), y un lanzamiento con
  `python` no lo es; bf16 en T4 lo rechaza Transformers (`training_args.py:1741-1746`); Gemma 4 declara
  `_no_split_modules` y gradient checkpointing (`modeling_gemma4.py:1451-1457`).
- Publicación siempre activa en el runner de Kaggle (HF es el único almacenamiento durable: una versión cortada pierde sus
  archivos), por lo tanto W&B online es obligatorio (la guardia existente `publicar` exige `online`). **Conflicto con "W&B
  opcional":** sin `WANDB_API_KEY` el runner falla antes de gastar GPU; el modo offline solo existe con `--sin-publicar`.
- fp16 exige `scaler.pt` en el checkpoint para reanudar (Trainer lo guarda en `trainer.py:3542-3560`).
- Los tokens van por entorno al subproceso; `ejecutar_con_log` ahora tapa una lista de secretos (HF y W&B).
- Reanudar desde HF baja solo `checkpoint-N/*` en una revisión de 40 hex y valida integridad e identidad antes de cargar el modelo.

**Evidencia.** `rtk proxy uv run --no-sync python -m unittest discover -s tests`: 104 corridos, 103 pasan y 1 opcional se
saltea (antes 87/86/1; +4 del entrenador, +13 del runner y los datos de humo). Tests con dobles: defaults A100 sin
cambios, fp16/`device_map` llegan a `from_pretrained` y `TrainingArguments`, identidad y reanudación incluyen los campos
nuevos, secretos fuera de argv/`runner.json`/log, publicación forzada y falla temprana, descarga con revisión y
`allow_patterns`, salida protegida.

**Limitaciones.** No se corrió en GPU ni en Kaggle. fp16 con Gemma 4 para **entrenar** no está probado (overflow posible);
no se midió la VRAM de entrenamiento en 2 T4 (el pico de inferencia fue ~19 GB); no se verificó el reparto de capas ni
que `device_map="auto"` deje margen en la GPU 0 para activaciones. Los datos de humo son sintéticos: prueban pipeline,
no calidad. La subida real a HF y W&B tampoco se probó. Modal no cambió (A100 40 GB sigue siendo la entrega).

**Primeros intentos reales (versiones 4 y 5 del notebook, commits `6f9abc0` y `5e1837e`).**
- Funcionó: Secrets en versión guardada, instalación de `requirements-entrenamiento.txt`, datos de humo,
  preparación, tokenización, W&B online (las corridas fallidas quedan registradas) y carga del modelo en 2 T4.
- v4 falló con un `ValueError` sin mensaje: el CLI solo mostraba el tipo. Ahora imprime el traceback por stderr
  (el runner tapa los secretos). `estado.json` sigue guardando solo el tipo, porque se puede publicar.
- v5 mostró **un bug que también rompía la A100**: `target_modules=["q_proj","v_proj"]` alcanzaba también a las
  torres de visión y audio de Gemma 4, cuyas capas son `Gemma4ClippableLinear` y PEFT no las soporta.
  **Decisión:** LoRA solo sobre la atención del modelo de lenguaje, con el patrón `LORA_OBJETIVO`
  (`re.fullmatch` en PEFT, verificado en `peft/utils/other.py:1521`; nombres en `modeling_gemma4.py:2166`).
  Test de regresión. Suite: 106 corridos, 105 pasan y 1 opcional se saltea. Ningún checkpoint llegó a publicarse.

**Humo validado en GPU real (versiones 6 y 7, commit `acc95b2`, Kaggle 2x T4, fp16, `device_map=auto`).**
Datos sintéticos de `scripts.datos_humo` (4 conversaciones, 8 pasos, checkpoint cada 2). Artefactos en
`corridas/kaggle/humo-entrena-1-v6` y `humo-entrena-2-v7` (no versionados).

| Corrida | Resultado | Publicado en `FranCalveyra/tp-admision` (revisión) | Pérdida pasos 1→8 |
|---|---|---|---|
| v6, desde cero | `terminada`, 7 min | checkpoint-2 `f11b081`, 4 `70808f8`, 6 `3932007`, 8 `0d6d84a`, final `2fa71b9` | 2,679 → 2,178 |
| v7, reanuda desde checkpoint-2 bajado de HF | `terminada`, 9 min | checkpoint-4 `32363aa`, 6 `f88db48`, 8 `79121d5`, final `391c3eb` | pasos 3–8 iguales a v6 hasta el 4.º decimal |

- fp16 en T4 entrena sin NaN; la norma del gradiente queda entre 1,08 y 1,74. W&B online registró las dos corridas.
- La reanudación restaura bien optimizador, scheduler, RNG y `scaler.pt`: la curva reanudada reproduce la
  original (2,5196/2,4173/2,3468 contra 2,5196/2,4179/2,3464).
- **Queda probado en GPU (T4):** recibe conversaciones, configuración y semilla; publica cada checkpoint con
  revisión; registra en W&B; reanuda desde un checkpoint publicado. **Falta:** datos reales (Eze/Yas), prompt
  final (Martu), medir VRAM de entrenamiento con secuencias reales y la ruta Modal/A100.
- `snapshot_download` deja una carpeta `.cache` dentro del run; no afectó la reanudación.

## Próximos pasos y bloqueos

1. Fijar en `configuracion.py` el adapter final (repo, revisión y subcarpeta) cuando exista.
2. Crear y verificar el secreto HF de escritura (`hf-write`, nombre propuesto) y probar una
   subida real a un repo privado; el runner opt-in ya existe (probado con dobles).
3. Coordinar conversaciones curadas y prompt final; no entrenar sobre práctica.
4. Facturación de Modal bloqueada: pide tarjeta. Mientras tanto, probar el baseline en Kaggle T4 x2 ([kaggle.md](kaggle.md)),
   con autorización explícita y resultados aparte. Con Modal resuelto y autorización de GPU: probar carga, entrenamiento mínimo,
   checkpoint y reanudación; luego baseline y lotes reales en A100-40GB.
5. Completar accesos compartidos, mediciones de recursos y documentación final. Compartir el repo
   privado de HF y el proyecto W&B con la cátedra queda **para cerca de la entrega** (decisión del
   2026-11-03). La consigna no dice qué usuarios invitar: hay que preguntarle a la cátedra.

Gemma comprimido en Mac es una posibilidad de desarrollo, no instalada ni medida; no
reemplaza la ejecución final A100. El token HF actual es de lectura y no habilita publicación.

## Guías específicas

| Tema | Documento |
|---|---|
| Entorno del equipo | [entorno-equipo.md](entorno-equipo.md) |
| RAG y experimentos | [buscador-baseline.md](buscador-baseline.md) |
| Modelo y ejecución baseline | [modelo-modal.md](modelo-modal.md) |
| Integración sin GPU | [integracion-cpu.md](integracion-cpu.md) |
| W&B | [wandb.md](wandb.md) |
| Conversaciones propuestas | [conversaciones-entrenamiento.md](conversaciones-entrenamiento.md) |
| Preparación de corrida | [preparacion-entrenamiento.md](preparacion-entrenamiento.md) |
| Tokens y máscaras | [tokenizacion-entrenamiento.md](tokenizacion-entrenamiento.md) |
| Trainer SFT | [entrenamiento-sft.md](entrenamiento-sft.md) |
| Persistencia Modal | [entrenamiento-modal.md](entrenamiento-modal.md) |
| Respaldo Kaggle (T4 x2), inferencia y entrenamiento humo | [kaggle.md](kaggle.md) |

## Cómo mantener esta bitácora

Después de cada unidad: actualizar fecha y estado; agregar decisión/motivo o experimento
(también fallido); citar comando, artefacto y limitación; renovar bloqueos y próximo paso.
No convertir métricas de desarrollo en resultados finales ni marcar validado lo que solo
está implementado. Las reglas comunes viven en [AGENTS.md](../AGENTS.md).
