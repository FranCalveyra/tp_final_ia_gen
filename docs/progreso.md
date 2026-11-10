# Progreso y decisiones

**Actualizado: 2026-11-05.** Hay generación y humo de entrenamiento LoRA con publicación/reanudación HF
**en Kaggle T4**, no en A100. **Todavía no hubo entrenamiento con datos reales curados.** La carga solo texto
está validada en T4. Esta bitácora centraliza evidencia y enlaza guías; no reemplaza el informe final.

**Ruta vigente:** Modal descartado definitivamente por decisión del usuario (2026-11-05), sin tarjeta.
Kaggle es la plataforma operativa de desarrollo; evidencia propia T4 x2. A100-40GB final sigue pendiente:
menú de `franciscocalveyra/tp-admision-baseline` verificado sin A100 (2026-11-05); determinar otra vía/contexto
sin asumir GPU/VRAM ni equiparar T4 con A100.

**Known Issue abierto:** [KI-A100-001](../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb).
La consigna exige A100-40GB **también para entrenar**, además de evaluación/recursos; sin excepción acordada.
Se seguirá desarrollando en T4, pendiente de confirmar el alcance concreto antes de la próxima ejecución GPU.

## Estado de nuestra parte

| Issue | Implementado / validado | Falta para cerrar |
|---|---|---|
| #1 Baseline | **Cerrado (2026-11-04).** 50 casos × original/b/c en Kaggle T4: 10, 10 y 7 contra la referencia 12, 15 y 12 | Repetir en A100 por vía disponible a determinar |
| #2 Infraestructura | **Cerrado (2026-11-04).** Lock, `uv sync` con inferencia, plantilla de variables, app A100-40GB (ahora legacy), W&B, repo HF privado | Invitar a la cátedra a HF y W&B pasó al #16 |
| [#3 SFT](https://github.com/FranCalveyra/tp_final_ia_gen/issues/3) | Conversor, validación, snapshots/tokenización, Trainer LoRA y runner persistente; humo GPU T4 con HF/W&B y reanudación desde checkpoint HF | Cuatro criterios técnicos respaldados; aceptación/accesos y uso efectivo por Martu pendientes. KI-A100-001 mantiene deuda entrenamiento/evaluación/recursos |
| [#4 Lotes](https://github.com/FranCalveyra/tp_final_ia_gen/issues/4) | **Cerrado completed (2026-11-05), alcance funcional T4.** Carga única, lote activo, conversación completa; 1→4 baja 33,3→15,6 s/caso, mismos 10/50 | Deuda A100-40GB continúa en KI-A100-001; adapter final pendiente y causa CUDA T4 de lotes grandes no identificada |
| #5 Recursos | Métricas y borrador Parte 5 con inferencia T4 y humo de entrenamiento solo texto: 15,675 GB pico | Recursos/costos A100 y antes/después con adapter final; integrar `INFORME.md` |
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
| `prefijos`: sin comunes + prefijo 4 + umbral 0,6, hasta 3 secciones (2026-11-10) | 55/60 | 0,933 | 0,686 | 549 | Candidato seleccionable; propuesto como default, sin adoptar |
| Referencia de la cátedra, 2 secciones | 51/60 | 0,875 | 0,492 | 580 | Comparación |

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
  `t4-original-04` mostró un **bug real de la entrega**: `AutoProcessor` de Gemma 4 carga `Gemma 4VideoProcessor`,
  que exige torchvision, y torchvision no está en nuestro lock. En la A100 de la cátedra también habría fallado.
- **Decisión:** `ModeloLocal` usa `AutoTokenizer`. El agente es solo texto, y así comparte con el entrenamiento
  (`tokenizacion.py`) el mismo chat template, ya validado con el tokenizer real en CPU. No se agregan
  dependencias de visión ni de audio. Tests: 82 corridos, 81 pasan y 1 opcional se saltea. Falta la corrida real.
- `t4-original-08` (commit `e5c7f7b`) mostró **otro bug de la entrega y del entrenamiento**: forzar
  `attn_implementation="sdpa"` falla porque `AutoModelForCausalLM` arma el Gemma 4 completo y su
  `Gemma 4AudioModel` no soporta SDPA. **Decisión:** no forzar la atención. Verificado en
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
  torres de visión y audio de Gemma 4, cuyas capas son `Gemma 4ClippableLinear` y PEFT no las soporta.
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

### 2026-11-04: borrador de la Parte 5 del informe (issue #5)

- Borrador en [informe-parte5.md](informe-parte5.md) con lo medido en T4 (s/caso, VRAM, lotes, humo de
  entrenamiento) y cada número con su artefacto. `INFORME.md` no se creó: es compartido.
- **Pendiente (PENDIENTE en el borrador):** todo lo que depende del modelo final o de A100-40GB (tiempo y VRAM de
  entrenamiento real, s/caso y costo en A100, lote máximo estable, antes/después). La VRAM de entrenamiento
  no se registró en el humo.

### 2026-11-04: diagnóstico del fallo CUDA con lote 8 (issue #4, Kaggle T4)

Versión 8 del notebook, commit `522d540`, original, 50 casos, `MODELO_LOTE=8`, con `CUDA_LAUNCH_BLOCKING=1` y
`TORCH_SHOW_CPP_STACKTRACES=1`. Artefactos en `corridas/kaggle/diag-l8` (no versionados).

- **No falló:** 10/50 resueltos, los mismos casos que con lote 1 (cero cambios por caso), parcial 0,831, 0 sin
  decisión. 17,4 s/caso, 36 micro-lotes, VRAM pico 21,7 GB y reservada 30,2 GB (suma de 2 GPU) sin romperse.
- **Lectura:** serializar los kernels evita el error. Apunta a concurrencia o asincronía del modelo repartido en
  2 T4 (`device_map=auto`) y no a falta de memoria. Es una hipótesis: no se identificó la línea.
- **Decisión:** no adoptarlo. Con bloqueo, el lote 8 (17,4 s/caso) es más lento que el lote 4 normal (15,6 s/caso).
  En Kaggle seguimos con **lote 4**. En la A100 el modelo entra en una sola GPU, así que este escenario no aplica
  y el lote máximo se mide ahí. No se probó `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (~25 min de
  cuota sin garantía).
- Cuota de esta corrida: ~40 min, más lenta por el bloqueo.

### 2026-11-05: VRAM de entrenamiento registrada (issue #5)

- El entrenador agrega la VRAM pico y la reservada (sumadas sobre las GPU visibles y por GPU, desde el arranque
  del proceso) a cada log de W&B y al resultado final en `estado.json`. El runner de Kaggle la copia a
  `runner.json` (`vram`). La ruta A100/Modal la recibe igual.
- Probado con dobles: 107 corridos, 106 pasan y 1 opcional se saltea. Todavía no hay medición real; sale
  de la primera corrida con datos reales.

### 2026-11-05: Gemma 4 solo con el modelo de lenguaje (issues #4 y #5)

- **Decisión:** inferencia y entrenamiento cargan Gemma 4 con `vision_config` y `audio_config` en `None`
  (`admision_tp/carga.py`). El agente es solo texto: así no se crean las torres de visión y audio ni sus
  embedders, y sus pesos no ocupan VRAM. Verificado en transformers 5.22.1: ambas configuraciones son
  opcionales (`configuration_gemma4.py:339-340`), `Gemma 4Model` no crea esas partes si son `None`
  (`modeling_gemma4.py:2164-2180`) y el `forward` solo las usa con `pixel_values` o `input_features`.
  El LoRA apunta a `language_model`, que no cambia.
- Probado con dobles: 109 corridos, 108 pasan y 1 opcional se saltea.
- **Validado en GPU (Kaggle 2x T4, versión 9, commit `a828467`):**

  | Original, 50 casos, lote 4 | Con torres (`base-original-l4`) | Solo texto (`solo-texto-l4`) |
  |---|---|---|
  | Resueltos | 10/50 | 10/50, mismos casos, **trazas idénticas en los 50** |
  | Segundos por caso | 15,6 | 15,7 |
  | VRAM pico (suma / por GPU) | 19,1 GB (10,9 + 8,2) | **18,1 GB (10,6 + 7,5)** |
  | Reservada | 29,2 GB | 28,3 GB |

  El humo de entrenamiento sin torres (`humo-solo-texto`) da la misma curva (2,679 → 2,178), publica los 5
  checkpoints y registra por primera vez la **VRAM de entrenamiento: 15,7 GB pico** (9,5 + 6,1; 15,8 reservada),
  con datos sintéticos cortos, así que no es representativa de la corrida real. El notebook terminó en `ERROR`
  por la celda final de resumen; las dos corridas terminaron con `returncode 0`.
- Ahorro modesto (~1 GB, ~5 %) sin ningún cambio de resultados. Se mantiene.

## Próximos pasos y bloqueos

1. Fijar en `configuracion.py` el adapter final (repo, revisión y subcarpeta) cuando exista.
2. Confirmar accesos individuales Kaggle/HF/W&B del equipo con el [checklist de handoff](handoff-equipo.md#checklist-de-accesos-hecho-versus-pendiente-de-comprobar);
   no crear Secrets ni habilitar facturación Modal, plataforma descartada.
3. Coordinar conversaciones curadas y prompt final; no entrenar sobre práctica.
4. Determinar disponibilidad y VRAM de una vía A100-40GB sin lanzar sesiones para probar permisos.
   El menú del notebook baseline de `franciscocalveyra` ya se verificó sin A100; no se inspeccionó VRAM
   en sesión ni se verificaron otros contextos/cuentas. Kaggle reemplaza operativamente a Modal;
   su T4 x2 validada no demuestra hardware final.
   Cuando exista ruta y autorización específica, medir carga/backward/memoria y lotes allí.
5. Completar accesos compartidos, mediciones de recursos y documentación final. Compartir el repo
   privado de HF y el proyecto W&B con la cátedra queda **para cerca de la entrega** (decisión del
   2026-11-03). La consigna no dice qué usuarios invitar: hay que preguntarle a la cátedra.

Gemma comprimido en Mac es una posibilidad de desarrollo, no instalada ni medida; no
reemplaza la ejecución final A100. El token HF de inferencia es de lectura y no habilita publicación; no confundirlo con el token de escritura
usado en Kaggle. Los Secrets de Modal quedan como referencia legacy, no como setup pendiente.

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
| Persistencia Modal (legacy, no usar) | [entrenamiento-modal.md](entrenamiento-modal.md) |
| Respaldo Kaggle (T4 x2), inferencia y entrenamiento humo | [kaggle.md](kaggle.md) |
| Parte 5 del informe (recursos y producción), borrador con mediciones T4 | [informe-parte5.md](informe-parte5.md) |

## Cómo mantener esta bitácora

Después de cada unidad: actualizar fecha y estado; agregar decisión/motivo o experimento
(también fallido); citar comando, artefacto y limitación; renovar bloqueos y próximo paso.
No convertir métricas de desarrollo en resultados finales ni marcar validado lo que solo
está implementado. Las reglas comunes viven en [AGENTS.md](../AGENTS.md).


## 2026-11-05 — Importación de trazas nuevas de la cátedra

- Importados por API desde `austral-ing-ai/talksmith-ing`, carpeta `missions/TPFinal`, commit inmutable `876d8894c1cd339ffb968f12754a21f460ffb393`. El fetch no respondió y se canceló; no se mezclaron historiales Git ni se modificó `admision/`.
- `entrenamiento/trazas.jsonl`: 800 registros, SHA-256 `adc42d2769cdb90e8509ab4fc71daf02f8c8934137315d06e9f87bd402ff7290`.
- `historial/trazas.jsonl`: 2.000 registros, SHA-256 `02a7a25bb73bfa199fe0a8a74d5a8332815b786e37056692a58167c9ec69cda8`.
- Verificación CPU: cada línea parsea como JSON y los cuatro archivos importados coinciden con el hash Git blob publicado por GitHub. Esto NO valida corrección de decisiones, compatibilidad con el conversor ni ausencia de contaminación. Sin GPU, pesos, builds, commits ni push.
- README y consigna originales actualizados conservados en `docs/referencias/kit-876d889/`; no reemplazan nuestros documentos. Sus enlaces relativos remiten a la estructura del kit original. La cátedra especifica conversión de pasos a mensajes y confirma que su referencia entrenó con las 800 trazas.
- Triage del subagente: asignados a Fran siguen abiertos #3 (SFT real), #4 (lotes A100/adapter final), #5 (recursos finales); #16 entrega es compartido. El humo sintético T4 no cierra SFT real ni demuestra funcionamiento A100-40GB.
- Próximo paso recomendado: #3, auditar procedencia, separación de datos y compatibilidad de las trazas con el pipeline, coordinando conversaciones curadas con Eze/Yas y prompt con Martu antes de entrenar. No se ejecutó entrenamiento.

### 2026-11-05: auditoría CPU de trazas nuevas (issue #3)

- Auditadas las 800 trazas de referencia y las 2.000 del historial importadas de upstream `876d889`:
  [informe y límites](auditoria-trazas.md). **No entrenan directamente:** el esquema crudo no satisface
  conversaciones v1; una transformación exploratoria en memoria sí pasa su estructura en 2.800/2.800.
- Las 2.000 trazas alinean con el CSV en ids, fecha, operador y los ocho campos de ficha. No se evaluó
  corrección normativa: el historial sigue requiriendo clasificación/curación. Las 800 varían parámetros
  entre casos; no deben reemplazarse sus búsquedas por las normas originales fijas.
- Validación CPU: `rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest test_conversaciones test_preparacion test_tokenizacion -q`:
  18 ejecutados, 17 pasan y 1 opcional omitido. Sin GPU, tokenizador, pesos, entrenamiento ni publicaciones.
- Próximo mínimo: acordar procedencia y prompt; conversor determinista de trazas a v1 que conserve acciones/
  observaciones y excluya la observación terminal. No implementado en esta auditoría.

### 2026-11-05: conversor determinista de trazas a conversaciones v1 (issue #3)

- Implementadas las interfaces públicas acordadas `convertir_traza` y `convertir_archivo`, con CLI CPU,
  origen/prompt obligatorios y referencias a fuente/id. Conserva acciones y observaciones; solo omite el
  resultado terminal. Valida secuencia, ficha y contrato v1 antes de publicar; rechaza ids duplicados y no
  sobrescribe salidas (tampoco symlinks). [Guía y comandos](conversaciones-entrenamiento.md#conversor-cpu-de-trazas-implementado).
- TDD red→green por unidad, sin mocks internos. Comando CPU completo:
  `rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
  **116 ejecutados, 115 pasan y 1 opcional omitido**. Suite pertinente: 25/24/1.
  Los rojos esperados y un fallo inicial del propio test de rechazo (contexto `assertRaises` fuera del subtest)
  se corrigieron antes del verde; no hubo builds, instalaciones ni GPU.
- Exploración de fuentes: 800 + 2.000 conversiones válidas en memoria usando el prompt actual, sin escribir
  conversaciones finales. Sigue pendiente acordar procedencia de referencia, curar historial y recibir el
  prompt final; convertir estructura no valida decisiones ni contaminación. No se cerró #3 ni se entrenó.


### 2026-11-05: auditoría final de infraestructura #3 y handoff del equipo

- Criterios revisados contra trainer, persistencia, tests y artefactos locales reales: mecanismos implementados;
  humo T4 v6/v7 terminado, HF con revisiones y mismo run_id W&B al reanudar. **No cierra #3 completamente:**
  A100-40GB/Modal y corte/recuperación reales siguen sin validar; datos/prompt finales son dependencias externas.
  [Matriz y comandos para Martu](entrenamiento-sft.md), sin agregar otro trainer ni nuevas seams de tests.
- Corregidas afirmaciones obsoletas en la guía: sí hubo humo GPU y publicación HF en Kaggle, sí se carga adapter
  en inferencia y el collator ya preserva padding `-100`. El Secret `hf-write` pendiente es específico de Modal.
- Ergonomía pendiente, no bloqueo del trainer: Modal no expone cadencia de checkpoints; usa50. Para humo con
  checkpoint temprano existe CLI directa `--guardar-cada 2`; en Modal hay que planificar suficientes pasos
  y comprobar un checkpoint confirmado. No se autorizó ni ejecutó ese ensayo.
- Verificación: `rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest test_entrenamiento test_persistencia test_kaggle_entrenamiento test_cli_entrenar test_preparacion test_tokenizacion test_conversion_trazas -q`:
  **52 ejecutados, 51 pasan y 1 opcional omitido**. `--help` de entrenar/preparar/tokenizar correcto sin GPU.
- Asignaciones GitHub realizadas y verificadas por la sesión principal: Eze `ezequielrickert` #6–9,
  Yas `Yasmindesivo` #10–12, Martu `MartuGiordanelli` #13–15 y los cuatro #16.
  [Handoff por persona](handoff-equipo.md): contratos, outputs, accesos propios y dependencias inevitables.
  Paquete preparado para `feat/training-handoff`; commit/push autorizados posteriormente por el usuario.
  Verificar publicación y hash comunicado por chat; no está integrado en `main` ni basta actualizar esa rama.


### 2026-11-05: accesos concretos y auditoría de pendientes #4/#5

- [Checklist de handoff](handoff-equipo.md#checklist-de-accesos-hecho-versus-pendiente-de-comprobar):
  GitHub código/asignaciones hechos (paquete inicial `3d7da60`, rama `feat/training-handoff`); accesos
  individuales Kaggle/HF/W&B **no verificados**, sin invitaciones ni cambios remotos en esta unidad.
  Notebooks inferencia para Eze/Martu y entrenamiento para Martu, con copia/versión separada y Secrets propios.
- Guía Kaggle corregida con artefactos reales y [matriz SFT](entrenamiento-sft.md); no confundir T4 histórico
  con estado actual de cuota/driver ni con validación A100. Informe Parte 5 ahora distingue humo 15,675 GB,
  diagnóstico lote 8 posterior y lote 16 no repetido con bloqueo; causa CUDA continúa siendo hipótesis.

| Issue | Nuestra parte ya implementada / evidencia | Bloqueo propio de validación | Dependencia de equipo | ¿Cerrar completo solo con CPU? |
|---|---|---|---|---|
| #3 | Pipeline y handoff; humo LoRA T4, HF/W&B/resume | A100-40GB/Modal, corte/recuperación real | Datos curados y prompt para corrida final | No |
| #4 | Carga única (`entrega.py`/`ModeloLocal`), lote activo (`AgenteJSON`), devuelve mensajes completos; medición 1/4 T4 | Lotes/rendimiento estable en A100; default sigue 1 | Adapter/prompt final para validación final | No |
| #5 | Métricas y borrador con fuentes, tiempos T4 y humo VRAM | Recursos/costo/capacidad A100 medidos | Modelo final para antes/después e informe compartido | No |

- Verificación local de artefactos: `salida/tiempos.json` da 33,273/15,642/17,366 s por caso para
  lote1/lote 4/lote 8 bloqueado; los tres completos 10/50 y 0 sin decisión. Las identidades y trazas por caso se
  documentaron en el barrido previo (conversación idéntica 49/50 para 1/4, mismo resultado 50/50).
  Lotes 8/16 originales tienen error CUDA aunque sus `runner.json` históricos indiquen rc 0.
- **Se puede terminar CPU:** documentación/evidencia y checklist de acceso, no los tres issues completos.
  No hay razón para inventar otro cambio de código. Próximo propio: habilitar facturación/permisos Modal y
  pedir autorización para un humo en 1×A100-40GB con checkpoint confirmado y recuperación; runner admite
  timeout 3600 s por defecto. Tiempo efectivo y costo A100 no medidos: confirmar tarifa/cuota y presupuesto
  antes de ejecutar; no extrapolar desde T4 ni encadenar entrenamiento final sin autorización aparte.
- Checks de esta unidad documental: suite CPU completa `rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
  **116 ejecutados, 115 pasan y 1 opcional omitido**; `rtk git diff --check` limpio y sin enlaces locales a
  archivos inexistentes en los cuatro documentos modificados. No se validaron anchors ni recursos remotos.


### 2026-11-05: Modal descartado definitivamente; Kaggle es la ruta operativa

- **Decisión explícita del usuario:** no pondrá tarjeta; Modal se descarta, no queda pendiente destrabarlo.
  Esta decisión reemplaza los planes previos de setup/corte/publicación en Modal, conservados arriba como
  historia. No se borraron código ni dependencias y no se ejecutó infraestructura.
- Kaggle deja de ser fallback temporal. La evidencia propia sigue siendo T4 x2: inferencia/lotes y humo SFT,
  publicación versionada HF, W&B y reanudación real desde un checkpoint HF completo. Eso permite proponer
  el cierre **de infraestructura genérica #3**, sin exigir un nuevo corte en Modal ni cerrar el issue automáticamente.
- #3 puede renombrarse **“Pipeline SFT reproducible con checkpoints HF, W&B y reanudación en Kaggle”**;
  sustituir el requisito de plataforma Modal por Kaggle y conservar el caveat explícito: A100-40GB aún no
  validada, datos/prompt/entrenamiento finales pendientes. La sesión principal decidirá actualizar el tracker.
- #4/#5 mantienen mediciones de hardware final y antes/después pendientes. A100-40GB continúa como requisito;
  vía alternativa a determinar. Kaggle CLI lista A100 con posibles restricciones; disponibilidad y VRAM de
  nuestras cuentas no verificadas ([fuente y límites](kaggle.md)). No se lanzó GPU para comprobarlo.
- Guías de estado/ruta actual y Parte 5 distinguen escenario Modal histórico/ilustrativo de próximos pasos.
  `entrenamiento-modal.md` y el bloque Modal de entorno quedan **legacy / no usar**. No se presenta la tarifa
  histórica de Modal como presupuesto de la plataforma futura ni se extrapolan recursos T4 a A100.

- Verificación documental de esta decisión: `rtk git diff --check` limpio; sin destinos locales inexistentes
  en los siete documentos revisados. No se probaron enlaces remotos/anchors ni se repitieron tests CPU por
  ser cambios solo de documentación; no hubo GPU, descargas, instalaciones, builds ni escrituras cloud.


### 2026-11-05: disponibilidad de acelerador revisada sin iniciar GPU

- Sesión principal verificó en Brave la cuenta `franciscocalveyra`, notebook `tp-admision-baseline`,
  Settings → Accelerator: solo `None`, `GPU T4 x2` (ya seleccionado) y `TPU v5e-8`; **no aparece A100**.
  [Vista consultada](https://www.kaggle.com/code/franciscocalveyra/tp-admision-baseline/edit/run/365769288).
- Draft Session apagada antes y después; sin ejecutar celdas, iniciar GPU, guardar ni cambiar Settings.
  VRAM no inspeccionada en sesión. Es evidencia de ese menú/cuenta/notebook el día indicado, **no demuestra**
  ausencia de A100 en todo Kaggle ni describe cuentas ajenas. Catálogo global A100 no concede permisos.
- Guía Kaggle y handoff actualizados; requisitos A100-40GB y vía final pendiente no cambiaron.
  Verificación de docs: diffcheck limpio; sin cambios de código, pruebas GPU ni operaciones remotas propias.


### 2026-11-05: avanzar en T4 con deuda A100 explícita

- Usuario decidió probar T4 mientras tanto y registrar A100 como deuda futura:
  [KI-A100-001 en README](../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb),
  añadido del equipo sin alterar el contrato original. Fran coordina; vinculado a #3/#4/#5 existentes.
- **Corrección importante:** `mission.md` línea 98 exige A100-40GB tanto para entrenamiento como para corrida
  de cátedra; línea 220 exige recursos/tiempos en ese hardware. No es solo una validación de entrega final.
  No hay excepción acordada. Resolver con evidencia A100 completa o aceptación explícita de alternativa
  por la cátedra y la validación que acuerden, no con una inferencia desde T4.
- T4 autorizado en principio para desarrollo; datos/configuración, timeout y consumo de la corrida concreta
  aún requieren confirmación. No se ejecutó GPU, build, test, descarga ni cambio remoto en esta unidad docs.
- Handoff enlaza la deuda. `rtk git diff --check` limpio; historial de decisiones anteriores preservado.


### 2026-11-05: cierre funcional de lotes y aceptación pendiente del handoff SFT

- Usuario aprobó terminar handoff operativo y actualizar #3/#4 sin GPU. #3 conserva cuatro criterios
  técnicos con evidencia T4; último criterio de uso por Martu sigue pendiente. Recepción, `--help` y
  preflight de permisos no equivalen a ejecutar sus experimentos.
- Sesión principal actualizó y verificó GitHub: [#3](https://github.com/FranCalveyra/tp_final_ia_gen/issues/3)
  sigue **OPEN**, nuevo título “Pipeline SFT reproducible con checkpoints HF, W&B y reanudación en Kaggle”,
  primeros cuatro criterios marcados y uso Martu pendiente.
- [#4](https://github.com/FranCalveyra/tp_final_ia_gen/issues/4) verificado **CLOSED / COMPLETED**, sus cuatro
  criterios funcionales marcados con evidencia T4. El body/comentario conserva explícitos KI-A100-001 y
  adapter final pendientes. Esta edición documental no hizo operaciones externas ni consume GPU.
- [Handoff](handoff-equipo.md#pendientes-por-persona-y-aceptación-operativa) enumera accesos pendientes por
  Eze/Martu/Yas y datos de cuenta faltantes; usernames Kaggle/HF/W&B no se infieren de GitHub. Yas puede
  curar en CPU sin esos accesos adicionales. No se enviaron invitaciones ni se compartieron Secrets.
- **KI-A100-001 permanece abierto** para entrenamiento/evaluación/recursos; #5 no se cierra. Sin excepción
  de cátedra ni consumo GPU nuevo. Checks documentales: diffcheck limpio y enlaces locales revisados.

### 2026-11-05: cerrar aporte documental propio de recursos e infraestructura

- Parte 5 corregida contra artefactos locales: el runner reportó 15.360 MiB por T4 en esas sesiones
  (=16,106 GB decimales, 32,212 GB para ambas). Métricas usan bytes/10⁹; la afirmación reservada 30,3 GB
  sobre capacidad 30 GB mezclaba GiB/GB y se retiró. Asignada/reservada no se suman; picos por GPU no
  necesariamente coinciden. No se infiere diagnóstico CUDA a partir de esos agregados.
- También se retiró “GPU-segundos”: son duraciones de pared del runner, no cuota facturada ni cómputo
  por dispositivo. Separados `train_runtime`, runner y sesión; 340 s v6/v7 no es la cuota total.
- Latencia heurística 5×micro-lote omitía espera entre micro-lotes. Ahora se informa finalización bajo 50
  casos desde inicio común (`admision/correr.py`, `ultima_llamada`): lote 4 mín 302,221/mediana 670,895/
  máx 782,102 s; throughput amortizado 15,642 s/caso no es latencia individual. Sin SLO interactivo medido.
- Fórmulas de capacidad/costo de producción con variables explícitas, sin nueva tarifa ni presupuesto.
  [Matriz #5](informe-parte5.md#9-integración-de-criterios-5): **trabajos futuros redactados**, tres criterios
  cuantitativos aún parciales por modelo/hardware final. #5 no se cierra; no fabricar más cambios CPU.
- Preparado [aporte Parte 4 infraestructura](informe-parte4-infra.md) para integración conjunta #16:
  enlaces a contratos/evidencia y límites, sin apropiarse de experimentos de Martu ni crear `INFORME.md`.
  Scripts CPU stdlib verificaron duraciones, conversiones de unidades y cuantiles sobre artefactos ya existentes;
  sin GPU, descargas, instalaciones, builds, tests nuevos ni escrituras cloud.

- Verificación final de esta unidad: `rtk git diff --check` limpio y enlaces a archivos locales existentes
  en Parte 4 infra/Parte 5/bitácora. Sin pruebas GPU ni tests nuevos; criterio trabajos futuros listo para integrar,
  no cumplimiento total #5 ni entrega #16 terminada.


### 2026-11-05: integración autorizada en main

- Usuario autorizó integrar y publicar todo en `main`. Sesión principal verificó `origin/main=fb74ce9`
  como ancestro (0/5 de divergencia), cambió a `main` y realizó fast-forward local hasta `1353cce`.
- [Handoff](handoff-equipo.md) actualizado para `main`, árbol limpio y `pull --ff-only`; rama feature/hash
  inicial conservados como historia. El commit de esta actualización y el push deben verificarse por la
  sesión principal antes de informar éxito remoto; usar el hash comunicado por chat, no uno autorreferente.
- Esta unidad solo cambió los dos documentos; diffcheck limpio. Sin tests, GPU, builds ni operaciones Git/cloud propias.


### 2026-11-10: buscador `prefijos` (issue #6, CPU, rama `feat/6-buscador-eze`)

- **Estado:** implementado y probado por interfaz pública; seleccionable con `BUSCADOR_VARIANTE=prefijos`.
  Default sin cambios (`baseline`). Sin probar con el agente ni en GPU.
- **Resultado dev (60 consultas):** 55/60, recall 0,933, precisión 0,686, 549 tokens/consulta, contra
  referencia 51/60, 0,875, 0,492, 580. Gana 9 y pierde 2 consultas (d027, d052) contra `baseline`;
  pérdidas explicadas por colisiones de prefijo y el umbral. b/c: mismo ranking; con evidencia ajustada a
  sus valores, mismas cifras que original.
- **Decisión:** proponer `prefijos` como default (más recall con menos tokens), pendiente de decisión del
  equipo y de una comparación con el agente, porque cambia cuántas secciones ve (1 a 3, media 1,82).
  Rechazados: chunks por párrafo, recorte de párrafos, otros stemmers, pesos de título y ajustes BM25.
- **Evidencia:** [guía del buscador](buscador-baseline.md#candidato-prefijos-2026-11-10-issue-6) y
  `corridas/rag-prefijos-20261110/` y `corridas/rag-exploracion-20261110/` (locales). Suite: 121 tests, 1 omitido opcional.
- **Exploración reproducible:** `rtk proxy uv run --locked --no-sync python -m scripts.explorar_buscadores
  --salida corridas/rag-exploracion-20261110/original.json` reproduce todas las filas rechazadas de la guía.
- **Límite:** hiperparámetros elegidos con el mismo dev; no es evaluación independiente.
- **Próximo paso:** correr práctica con el agente `baseline` vs `prefijos` (k y tokens por caso) antes de
  cambiar el default.
