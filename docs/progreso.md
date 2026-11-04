# Progreso y decisiones

**Actualizado: 2026-11-03.** Hay implementación y pruebas CPU; **todavía no hubo generación
con Gemma completo ni entrenamiento real**. Esta bitácora centraliza el estado y enlaza
las guías; no reemplaza el informe final ni implica cierre de issues.

## Estado de nuestra parte

| Issue | Implementado / validado | Falta para cerrar |
|---|---|---|
| #1 Baseline | Fábricas, buscador, adaptador CUDA y prueba de flujo con modelo guionado | Carga real y 50 casos × normas original/b/c, con trazas y puntuación |
| #2 Infraestructura | Lock, entorno CPU, plantilla de variables, definición A100-40GB; W&B online verificado | Validación GPU, permisos compartidos con equipo/cátedra y publicación HF |
| #3 SFT | Validación de conversaciones, preparación, tokenización, Trainer LoRA y runner persistente; tokenizador real probado | Entrenar y probar corte/reanudación reales; publicación desde Modal y datos/prompt finales |
| #4 Lotes | Coordinador del kit y adaptador admiten lotes; orden probado con dobles | Medir concurrencia, tiempos y memoria con modelo real |
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
| Respaldo Kaggle (T4 x2) | [kaggle.md](kaggle.md) |

## Cómo mantener esta bitácora

Después de cada unidad: actualizar fecha y estado; agregar decisión/motivo o experimento
(también fallido); citar comando, artefacto y limitación; renovar bloqueos y próximo paso.
No convertir métricas de desarrollo en resultados finales ni marcar validado lo que solo
está implementado. Las reglas comunes viven en [AGENTS.md](../AGENTS.md).
