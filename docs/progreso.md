# Progreso y decisiones

**Actualizado: 2026-11-11.** Hay generación y humo de entrenamiento LoRA con publicación/reanudación HF
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
| [#5 Recursos](https://github.com/FranCalveyra/tp_final_ia_gen/issues/5) | **Cerrado (2026-11-13), alcance T4.** SFT real (363 pasos, 19,97 GB pico), lote, costo Kaggle, proyección y antes/después 27/150 → 100/150 (adapter v39) en [Parte 5](informe-parte5.md) | A100 como limitación conocida (KI-A100-001); integrar `INFORME.md` (#16) |
| [#6 Buscador](https://github.com/FranCalveyra/tp_final_ia_gen/issues/6) | **Implementado y comparado con el agente (2026-11-10).** `evaluar_rag`: `baseline` 48/60, `prefijos` 55/60 (referencia 51/60). Agente sin entrenar en Kaggle T4: `baseline` 10/50, `prefijos` 6/50; default `baseline` | Revisión del equipo de la Parte 2 del informe; opcional: repetir con el modelo entrenado |
| [#7 Ciclo del agente](https://github.com/FranCalveyra/tp_final_ia_gen/issues/7) | **Implementado y validado en GPU T4 (2026-11-10).** `entrega.py` usa `AgentePrompt` + `PromptSpec`; tope de 40 llamadas del kit; guarda de contexto por caso; medición offline del contexto. Kaggle v23 (`AgentePrompt`, 50 casos) y v24 (entrega nueva, 10 casos): conversaciones idénticas a `AgenteJSON` (v22) | Enviar a Martu la nota de formato (#13); validación A100 sigue en KI-A100-001 |
| [#9 Evaluación completa](https://github.com/FranCalveyra/tp_final_ia_gen/issues/9) | **Cerrado; filas medidas en GPU T4 (2026-11-11).** `sin_entrenar` (v25) 10/10/7 y `solo_prompt` (v27, `procedimiento-v1`) 14/16/12 en práctica original/b/c | Filas `solo_entrenamiento` y `ambos` cuando haya adapter publicado (#15); A100 en KI-A100-001 |
| #10 Historial | **Implementado (2026-11-05).** Clasificador determinista, pruebas, `historial_clasificado.csv`, análisis reutilizable de costumbres e informe | Mantenerlo compatible con la curación del issue #11 |
| #11 Costumbres | **Implementado (2026-11-05).** Trazas validadas, dos costumbres confirmadas, patrón por operador documentado y resumen para el equipo | Revisar con Martu y Eze antes de usarlo en el prompt final |
| [#13 Prompt](https://github.com/FranCalveyra/tp_final_ia_gen/issues/13) | **Validado en GPU T4 (Eze, 2026-11-11).** Elegido `procedimiento-v1` en el dev (9/40 contra 8/40); `prefijos` pierde (4/40); C falla en CUDA de forma reproducible. Práctica (solo reporte): 42/150 contra 27/150, a ~2,6× de contexto | #8 regenerado con B en `v3-procedimiento`. Decisión de equipo: default de `entrega.py` y qué dataset entrena #15 ([detalle](optimizacion-prompt.md#decisión)). **Pendiente de la consigna:** el formato de los mensajes no se optimizó, solo el mensaje de sistema ([estado](optimizacion-prompt.md#estado-2026-11-11)) |
| [#14 Análisis de errores](https://github.com/FranCalveyra/tp_final_ia_gen/issues/14) (Eze) | **Implementado, probado con dobles y aplicado a corridas reales (2026-11-10).** `scripts.analizar_errores`: fallas por control, tipo de caso y fuente (formato/costumbre/buscador/modelo), comparación entre corridas; registro de pasadas v22, v22 `prefijos`, v23, v24 y las de #13 (dev v25 y práctica v27) en [analisis-errores.md](analisis-errores.md) | Analizar los checkpoints de #15 contra la pasada 6 con el mismo comando; las reglas de fuente son heurísticas sin precisión medida |
| #16 Entrega compartida | Guías técnicas y esta bitácora | Informe, README final reproducible, diagrama, Loom y envío con el equipo |

El formato de conversaciones es **propuesto**, pendiente de coordinación con Eze/Yas.
El entrenamiento real también espera el prompt optimizado de Martu. Preparar código no
reemplaza esos entregables. `INFORME.md` y el resumen del issue #11 contienen la síntesis actualizada.

### 2026-11-05: clasificación del historial (issue #10)

- Se agregó [`scripts/clasificar_historial.py`](../scripts/clasificar_historial.py), que carga los parámetros y las secciones observables de `normas/original/`, clasifica por precedencia normativa y aprende la costumbre etaria a partir de frecuencias.
- Resultado corregido y reproducible: 1.631 `correcta`, 279 `error` y 90 `ruido`, sobre 2.000 filas. Se conservaron las costumbres etaria y de selección de prepaga; el issue #11 amplió la evidencia con las trazas.
- Evidencia: [`INFORME.md`](../INFORME.md), [`costumbres.md`](../costumbres.md) y [`historial_costumbres.json`](../historial_costumbres.json). Pruebas: `python -m unittest tests.test_clasificar_historial -v` (3/3).
- Limitación: el CSV no permite comprobar boxes disponibles, evaluación médica, retriage, camas, aislamiento ni derivaciones; esas normas quedaron documentadas como no verificables.

## Evidencia disponible

### 2026-11-05: seleccion de datos de entrenamiento (issue #12)

- Se seleccionaron 1.595 conversaciones reales: filas `correcta`, sin errores, ruido ni las 36 variaciones ambiguas de aviso a menores concentradas en operadores.
- Se generaron 5.000 candidatos con `admision.generador`, semilla 12012, y se conservaron 500 sinteticos: 100 por circuito y al menos 60 por cada bucket especial (ART, dos credenciales, acompanante y sin nadie).
- El dataset final contiene 2.095 conversaciones v1: 1.595 de historial y 500 sinteticas. Las 500 fichas sinteticas pasan las reglas normativas y el conjunto combinado pasa el validador sin errores.
- Auditoria corregida: la costumbre etaria solo fija `no` para menores de 75. En mayores se usa la frecuencia observada 175/306 con umbral determinista por hash; quedaron 74 avisos y 37 sin aviso entre 111 mayores. Cambiaron 37 conversaciones sinteticas y el nuevo SHA-256 esta en el manifiesto.
- No se usan casos de `practica/`. Procedencia, balance, IDs y SHA-256 quedan en [`datos/entrenamiento/manifiesto.json`](../datos/entrenamiento/manifiesto.json). Pruebas especificas: 8/8.

### 2026-11-05: descubrimiento de costumbres (issue #11)

- Las 2.000 trazas de `historial/trazas.jsonl` coinciden por ID, operador y ficha final con `historial/historial.csv`; se usan como evidencia del proceso, no como etiquetas independientes.
- Costumbres confirmadas: no avisar al voluntariado a menores de 75 anos (1.298/1.415 `no` en filas limpias menores, ocho estratos `motivo x cobertura`) y priorizar prepaga con dos credenciales (162/171, 94,7%).
- Patron adicional no confirmado como costumbre hospitalaria: OP-04, OP-11 y OP-17 concentran 117/225 avisos a menores, mientras los otros 17 operadores tienen 0/1.190. Se documentan los IDs y no se agrega una regla automatica.
- Evidencia y resumen: [`costumbres.md`](../costumbres.md), [`historial_costumbres.json`](../historial_costumbres.json) y [`resumen para el equipo`](resumen-costumbres-issue-11.md). Pruebas especificas: 5/5.

### 2026-11-05: auditoria y correccion de issue #10

- Se corrigio la seleccion de cobertura con dos credenciales: ya no se asume la primera; se usa la credencial efectivamente verificada y se conserva la preferencia observada por prepaga como costumbre separada.
- La deteccion de costumbres excluye errores normativos y compara estratos `motivo x cobertura`. Se conservaron 8 estratos comparables para la costumbre etaria y se documento la preferencia por prepaga en 162/171 filas limpias.
- Nueva salida: 1.631 `correcta`, 279 `error`, 90 `ruido`. Cambiaron 115 filas respecto de `959838a`: 71 error->correcta, 34 ruido->correcta, 5 error->ruido y 5 correcta->ruido.
- Pruebas especificas: 4/4. La suite completa sigue teniendo fallas ajenas en Kaggle/entrenamiento, integracion CPU, persistencia, preparacion y variantes del buscador.

### Buscador: conjunto de desarrollo, 60 consultas, normas originales

| Experimento | Completas | Recall | Precisión | Tokens/consulta | Decisión |
|---|---:|---:|---:|---:|---|
| BM25 baseline, 2 secciones | 48/60 | 0,817 | 0,467 | 618 | Default conservado |
| BM25, 4 secciones | 57/60 | 0,950 | 0,271 | 1.270 | Comparación; duplica aproximadamente el contexto |
| Sin palabras comunes, 2 secciones | 50/60 | 0,850 | 0,483 | 591 | Candidato seleccionable `sin_comunes` |
| Peso 0,25 para palabras comunes | 49/60 | 0,833 | 0,475 | 605 | No adoptado |
| Fusión palabras/caracteres, 2 secciones | 48/60 | 0,817 | 0,467 | 612 | No adoptada |
| `prefijos`: sin comunes + prefijo 4 + umbral 0,6, hasta 3 secciones (2026-11-10) | 55/60 | 0,933 | 0,686 | 549 | Seleccionable; no adoptado tras compararlo con el agente (6/50 contra 10/50) |
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
  con `.venv/bin/python -m unittest discover -s tests -v`.
  El test real del tokenizador requiere su grupo/entorno; previamente pasó en su validación específica.

Comando CPU del proyecto (sin build):

```bash
uv run --locked --no-sync python -m unittest discover -s tests -v
```

## Decisiones y motivos

| Decisión | Motivo y costo / límite |
|---|---|
| `AgentePrompt` en la entrega (2026-11-10, antes `AgenteJSON`) | Mismo protocolo y textos que `AgenteJSON` con el baseline, pero configurable por `PromptSpec`. El corte sale del tope de 40 llamadas del kit, no de 16 turnos fijos. |
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

**Evidencia.** `uv run --no-sync python -m unittest discover -s tests`: 104 corridos, 103 pasan y 1 opcional se
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
  entrenamiento) y cada número con su artefacto. `INFORME.md` se mantiene como informe compartido.
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

### 2026-11-05: infraestructura de prompts configurables (issue #13, en progreso)

- Se agregó `PromptSpec` con textos versionados, SHA-256 canónico y validación estricta de
  placeholders e integridad al cargar. El baseline vive como configuración explícita y una
  prueba lo compara con el mensaje y los formatos actuales de `admision.ejemplo`.
- Se agregó `AgentePrompt`, que conserva el protocolo JSON y `jugar_lote` pero recibe todos
  los textos mediante `PromptSpec`. Con el baseline, sus conversaciones, llamadas y entradas
  al modelo simulado coinciden exactamente con `AgenteJSON`; otras variantes pueden cambiar
  system, inicio, observaciones y corrección sin tocar el kit.
- Verificación CPU focalizada: 10 tests, todos pasan. Suite completa: 119 ejecutados,
  118 pasan y uno se omite porque requiere un snapshot local explícito del tokenizer real.
  Solo se usaron modelos y entornos simulados para el código nuevo: no se cargó un modelo
  real, no se generaron datos y no se implementó DSPy/GEPA. Diseño y comandos:
  [optimización de prompt](optimizacion-prompt.md).

### 2026-11-09: tokens y métrica de candidatos (issue #13, en progreso)

- `admision_tp.evaluacion_prompts` mide los cuatro textos configurables y el contexto
  inicial mediante IDs del tokenizer inyectado, sin aproximación por caracteres; conserva
  opciones de chat y hash de plantilla. No descarga ni carga el tokenizer por su cuenta.
- La métrica reutiliza los nueve `CONTROLES` y resultados de `admision.correr.puntuar`.
  La clave prioriza casos 9/9, después parciales y finalmente errores/eficiencia; también
  cuenta respuestas sin JSON con el mismo parser que el coordinador.
- Verificación CPU focalizada: 5 tests con tokenizer, trazas y resultados artificiales,
  todos pasan. Esto prueba el contrato y el orden, no tokens de Gemma ni calidad del agente.
  [Diseño y límites](optimizacion-prompt.md#longitud-y-métrica-de-candidatos).

### 2026-11-09: runner auditable de candidatos (issue #13, en progreso)

- `scripts.evaluar_prompts` recibe prompt, configuración, casos, esperadas y normas desde
  archivos externos. Juega antes de copiar/leer esperadas y luego reutiliza `puntuar`; no
  genera oráculos, casos, costumbres ni conversaciones.
- Cada salida es exclusiva y conserva snapshots, hashes por archivo, estado Git, identidad
  de prompt, configuración inmutable, trazas, nueve controles, tiempos y métricas. Los
  intentos fallidos también dejan manifiesto; no se sobrescriben.
- Verificación focalizada completa del issue #13: 21 tests pasan; suite CPU: 137
  ejecutados, 136 pasan y uno se omite por requerir tokenizer real local. Se usó Python
  3.12 del sistema porque no hay `uv`/`.venv`; no se instaló `transformers`, no se cargó
  Gemma ni se ejecutó GPU. [Uso, artefactos y fronteras](optimizacion-prompt.md#runner-reproducible).

### 2026-11-10: ciclo del agente y contexto (issue #7)

- **Entrega:** `entrega.construir_agente()` arma `AgentePrompt(ModeloLocal, PromptSpec)`. El prompt sale de
  `PROMPT_SPEC` (ruta relativa a la raíz del repo) o, sin variable, de `configuraciones/prompts/baseline.json`;
  se valida su sha256 antes de cargar el modelo. Prueba diferencial: 40 semillas × 4 casos con respuestas
  aleatorias (acciones válidas, errores de herramienta, texto sin JSON, fichas incompletas) sobre el `Entorno`
  y el buscador reales dan conversaciones, logs y fichas idénticas a la entrega anterior (`AgenteJSON`).
- **Tope de llamadas:** el límite fijo de 16 turnos se reemplazó por `MAX_LLAMADAS × max_sin_json` = 120
  turnos (40 llamadas con hasta dos reintentos sin JSON antes de cada una). El caso termina cuando el entorno
  lo termina o al llegar a 40 llamadas. Es la única diferencia de conducta con la entrega anterior sin guarda
  activada; en v22 el máximo observado fue 8 turnos, así que no habría cambiado ningún caso de esas corridas.
- **Guarda de contexto:** `ModeloLocal.medir_contexto` mide una conversación con el mismo chat template que la
  generación. Antes de cada llamada, un caso que no entra en `max_contexto − max_tokens_nuevos` termina sin
  decisión y su conversación termina con un mensaje `[agente] Caso terminado sin registrar decisión…`; los
  demás casos del lote siguen (antes el `ValueError` vaciaba todas las conversaciones). Sin esa capacidad en el
  modelo, el agente se comporta como antes. Nada se recorta.
- **Contexto, medido offline** con `python -m scripts.medir_contexto` sobre las 100 trazas v22 (Kaggle T4,
  Gemma sin entrenar; artefactos locales `corridas/kaggle/v22/`), que reproduce exactamente las 100
  conversaciones grabadas antes de comparar. Reparto de `caracteres_de_contexto`: sistema 55 %, búsquedas 25 %,
  otras observaciones 13 %, respuestas 5 %, mensaje inicial 2 %. El mejor cambio sin pérdida del agente
  (`{herramienta}:\n{observacion}`) ahorra 0,67 %; quitar el nombre de la herramienta, 1,4 %; compactar líneas en
  blanco, 0,10 %; resumir secciones repetidas, 0,05–0,79 %. **No se adoptó ninguno**: el ahorro no justifica
  regenerar los datos de #8, y los dos últimos cambian la presentación del buscador (#6). La palanca real es el
  mensaje de sistema (#13). Detalle y contrato con Martu: [optimización de prompt](optimizacion-prompt.md#contrato-de-formato-7--13).
- **Fuente única de formato:** un test rejuega todas las conversaciones de `generar_conversaciones` con
  `AgentePrompt` y comprueba que son idénticas a lo que el agente arma en inferencia.
- Verificación: suite CPU completa con `uv run --locked --no-sync python -m unittest discover -s tests`.
  **Todo con dobles:** no se cargó Gemma ni su tokenizer (no está en caché local), no se usó GPU.
- **Límites:** `medir_contexto` no se probó con el tokenizer real; el runner de #13 sigue fijando 16 turnos en
  sus configuraciones (decisión de Martu); la guarda también actúa en ese runner porque usa `ModeloLocal`.
- **Validación en GPU (Kaggle T4 x2, Gemma 4 E4B sin entrenar, float16, `device_map=auto`, lote 4, normas original;
  puntuado localmente; artefactos en `corridas/kaggle/` locales):**
  - **v23 — `AgentePrompt` con Gemma real** (commit `5d4b657`, runner de #13 con `--solo-plomeria`, prompt
    `baseline`, 50 casos): **50/50 conversaciones idénticas** a `AgenteJSON` en v22; mismas decisiones, 10/50,
    parcial 0,831, 19.954 caracteres de contexto por caso. Runner `completada`, propósito `plomeria`, no comparable
    (sin clave). Largo del prompt con el **tokenizer real**: sistema 615 tokens, contexto inicial 650. ~19 min de GPU.
  - **v24 — `entrega.py` nueva** (commit `2c16014`, `infra/kaggle_baseline.py`, 10 casos): sin errores; la guarda
    midió cada llamada con el tokenizer real; **10/10 conversaciones y decisiones idénticas** a v22; 0 casos
    cortados por la guarda; 0 sin decisión. ~19 min de GPU (13,5 de instalación).
  - Conclusión: la entrega nueva corre con el modelo real y no cambia decisiones. El tope de 40 llamadas y la
    guarda no se activaron (máximo 8 turnos y ~4.000 tokens por llamada); su comportamiento está cubierto por
    los tests con dobles. T4 no equivale a A100 (KI-A100-001).

## Próximos pasos y bloqueos

1. #13 (Eze): autorizar y correr la celda de fase 1 en Kaggle (~1 h 35 min a 2 h de T4), rankear y pasar a la fase 2
   ([optimización de prompt](optimizacion-prompt.md#fase-1-2026-11-10-eze-conjunto-de-desarrollo-candidatos-y-evaluación-preparada)).
2. Con autorización, correr `configuraciones/evaluacion/actual.json` en Kaggle ([evaluacion.md](evaluacion.md));
   después completar la plantilla de cuatro filas cuando existan prompt optimizado y adapter.
3. Fijar en `configuracion.py` el adapter final (repo, revisión y subcarpeta) cuando exista.
4. Confirmar accesos individuales Kaggle/HF/W&B del equipo con el [checklist de handoff](handoff-equipo.md#checklist-de-accesos-hecho-versus-pendiente-de-comprobar);
   no crear Secrets ni habilitar facturación Modal, plataforma descartada.
5. Coordinar conversaciones curadas y prompt final; no entrenar sobre práctica.
6. Determinar disponibilidad y VRAM de una vía A100-40GB sin lanzar sesiones para probar permisos.
   El menú del notebook baseline de `franciscocalveyra` ya se verificó sin A100; no se inspeccionó VRAM
   en sesión ni se verificaron otros contextos/cuentas. Kaggle reemplaza operativamente a Modal;
   su T4 x2 validada no demuestra hardware final.
   Cuando exista ruta y autorización específica, medir carga/backward/memoria y lotes allí.
7. Completar accesos compartidos, mediciones de recursos y documentación final. Compartir el repo
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
| Optimización de prompt, infraestructura inicial y contrato de formato con #7 | [optimizacion-prompt.md](optimizacion-prompt.md) |
| Evaluación completa (50 casos × original/b/c) y tabla de la Parte 4 | [evaluacion.md](evaluacion.md) |
| Análisis de errores por corrida y registro de pasadas | [analisis-errores.md](analisis-errores.md) |

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
- Validación CPU: `env PYTHONPATH=tests .venv/bin/python -m unittest test_conversaciones test_preparacion test_tokenizacion -q`:
  18 ejecutados, 17 pasan y 1 opcional omitido. Sin GPU, tokenizador, pesos, entrenamiento ni publicaciones.
- Próximo mínimo: acordar procedencia y prompt; conversor determinista de trazas a v1 que conserve acciones/
  observaciones y excluya la observación terminal. No implementado en esta auditoría.

### 2026-11-05: conversor determinista de trazas a conversaciones v1 (issue #3)

- Implementadas las interfaces públicas acordadas `convertir_traza` y `convertir_archivo`, con CLI CPU,
  origen/prompt obligatorios y referencias a fuente/id. Conserva acciones y observaciones; solo omite el
  resultado terminal. Valida secuencia, ficha y contrato v1 antes de publicar; rechaza ids duplicados y no
  sobrescribe salidas (tampoco symlinks). [Guía y comandos](conversaciones-entrenamiento.md#conversor-cpu-de-trazas-implementado).
- TDD red→green por unidad, sin mocks internos. Comando CPU completo:
  `env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
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
- Verificación: `env PYTHONPATH=tests .venv/bin/python -m unittest test_entrenamiento test_persistencia test_kaggle_entrenamiento test_cli_entrenar test_preparacion test_tokenizacion test_conversion_trazas -q`:
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
- Checks de esta unidad documental: suite CPU completa `env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
  **116 ejecutados, 115 pasan y 1 opcional omitido**; `git diff --check` limpio y sin enlaces locales a
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

- Verificación documental de esta decisión: `git diff --check` limpio; sin destinos locales inexistentes
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
- Handoff enlaza la deuda. `git diff --check` limpio; historial de decisiones anteriores preservado.


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

- Verificación final de esta unidad: `git diff --check` limpio y enlaces a archivos locales existentes
  en Parte 4 infra/Parte 5/bitácora. Sin pruebas GPU ni tests nuevos; criterio trabajos futuros listo para integrar,
  no cumplimiento total #5 ni entrega #16 terminada.


### 2026-11-05: integración autorizada en main

- Usuario autorizó integrar y publicar todo en `main`. Sesión principal verificó `origin/main=fb74ce9`
  como ancestro (0/5 de divergencia), cambió a `main` y realizó fast-forward local hasta `1353cce`.
- [Handoff](handoff-equipo.md) actualizado para `main`, árbol limpio y `pull --ff-only`; rama feature/hash
  inicial conservados como historia. El commit de esta actualización y el push deben verificarse por la
  sesión principal antes de informar éxito remoto; usar el hash comunicado por chat, no uno autorreferente.
- Esta unidad solo cambió los dos documentos; diffcheck limpio. Sin tests, GPU, builds ni operaciones Git/cloud propias.

### 2026-11-10: reproducibilidad byte a byte de clasificación y manifiesto

- `scripts/clasificar_historial.py` escribía `historial_clasificado.csv` con CRLF (default de `csv`),
  pero el versionado usa LF: regenerar daba mismo contenido con otros bytes. Ahora usa `lineterminator="\n"`;
  es el único writer CSV del script. Test nuevo en `tests/test_clasificar_historial.py` exige salida sin `\r`.
- `datos/entrenamiento/manifiesto.json` tenía `validacion.sha256` obsoleto (`9e62303a…`) frente al
  `conversaciones.jsonl` versionado (`59d718a6…`). Se reemplazó por el manifiesto regenerado con
  `python -m scripts.preparar_datos_entrenamiento --salida <tmp nuevo>`; el diff JSON muestra solo ese hash.
  Test nuevo en `tests/test_preparar_datos_entrenamiento.py` compara el hash del manifiesto con el archivo.
- Motivo: la sección de auditabilidad de la consigna exige artefactos regenerables y verificables.
- Evidencia: regeneración en carpetas temporales nuevas; `cmp` idéntico para `historial_clasificado.csv`,
  `costumbres.md`, `historial_costumbres.json`, `conversaciones.jsonl` y `auditoria_sinteticos.jsonl`.
  Ninguna clasificación ni dato de entrenamiento cambió; ningún doc citaba el hash viejo.
  Suite: `uv run --locked --no-sync python -m unittest discover -s tests` → 150 tests OK (1 skipped). Sin GPU ni builds.

### 2026-11-10: corrección de curado y datos de entrenamiento (#10-#12)

- **NOR-COB-04** (`scripts/clasificar_historial.py`, `cobertura_esperada`): con número terminado en
  P-COB-2 la ficha es `sin_cobertura` y se notifica a la entidad esté activa o no la afiliación (§1, §3;
  §4 excluye ART). Antes se exigía no notificar si estaba inactiva. El generador sintético usa la misma función.
- **Voluntariado**: la costumbre se infiere buscando `edad >= umbral` + condición observable del triage sobre
  filas limpias; sale 75 años y **sin acompañante** (1.609/1.760 limpias vs 1.507 sólo por edad). Los 17
  operadores sin desvíos la cumplen 185/185, 141/141 y 1.348/1.348. Se quitó la compuerta de estratos
  `motivo × cobertura`, que dejaba desvíos idénticos como `correcta` o `ruido` según el estrato.
- **Dos credenciales**: costumbre completa = verificar sólo la prepaga; inactiva → `sin_cobertura` sin recurrir a
  la obra social (45/45 filas; una verificación en 193/193 trazas). Sin cambios de etiqueta.
- **Sintéticos**: aviso determinista por la costumbre completa (antes muestreo 175/306); búsquedas con
  `entrega.construir_buscador` (BM25 `baseline`, `k_entorno=2`) en lugar de `BuscadorSintetico` (eliminado).
  Las 400 conversaciones con circuito recuperan NOR-TDE-01/02/03. El selector de historial ya no excluye
  "ambiguas": falla si una fila `correcta` contradice la costumbre.
- Resultado: 1.600 `correcta` / 240 `error` / 160 `ruido` (antes 1.631/279/90). Cambios: 42 error→correcta,
  3 error→ruido y 6 correcta→error por NOR-COB-04; 67 correcta→ruido por la costumbre del voluntariado (todas
  de OP-04/11/17). Dataset: 1.600 reales + 500 sintéticos = 2.100 (antes 1.595 + 500 = 2.095); 53 fichas
  sintéticas cambiaron y las 500 conversaciones sintéticas cambian sus observaciones de búsqueda.
- Evidencia: clasificador y `preparar_datos_entrenamiento` regenerados dos veces en carpetas nuevas con `cmp`
  idéntico; cruce con las 800 trazas de referencia (sólo verificación, no se agregan al dataset): 74/74, 67/67,
  659/659 voluntariado; 62/62 notifican con terminación reservada inactiva; 65/65 verifican la prepaga y 20/20
  inactivas → `sin_cobertura`. Suite: `uv run --locked --no-sync python -m unittest discover -s tests` →
  159 tests OK (1 skipped). Sin GPU, builds ni descargas.
- Límites: las normas siguen en `normas/original` (variar valores es decisión pendiente). En sintéticos,
  NOR-COB-03 (246), NOR-ADM-03 (114) y NOR-COB-04 (38) aparecen en `fundamentos` sin ser recuperadas por ninguna
  búsqueda del guion. La conversión de historial sigue usando los resultados de búsqueda originales de las
  trazas (issue #8). NOR-ADM-01 §4 se codifica más estricto que el texto (box anticipado no exigido = error),
  pero afecta 0 filas; quedó comentado. Detalle en [INFORME.md](../INFORME.md) y [costumbres.md](../costumbres.md).

### 2026-11-10: robustez del runner de prompts (#13) para Kaggle

- Alcance: solo el runner de evaluación de Martu; no se optimizó prompt, no hay DSPy/GEPA y
  `entrega.py` no cambió. Detalle en [optimización de prompt](optimizacion-prompt.md#guardas-de-datos-y-propósito).
- Guarda de práctica: el runner copiaba los casos antes de llamar a `jugar`, así que el chequeo
  `revisar_que_no_haya_esperadas` del kit nunca veía la carpeta original (se habían corrido 50 casos de
  `practica/`). Ahora se llama a esa función del kit sobre las rutas originales y se rechazan casos/esperadas
  que resuelven dentro de `practica/` (también vía symlink). Solo `--solo-plomeria` lo permite, con
  `proposito=plomeria` en el manifiesto. Toda corrida registra `proposito` y `guardas`.
- Comparabilidad: solo `estado=completada` + `proposito=evaluacion` tiene `clave_comparacion`;
  `clave_comparacion`/`clave_de_manifiesto` levantan `ValueError` para el resto. Test con un doble que
  lanza el mismo `ValueError` de contexto excedido que `ModeloLocal`. El CLI sale con 1 si `completada_con_error`.
- Kaggle: `configuraciones/evaluacion-prompt-kaggle-t4.json` (float16, `device_map=auto`); el ejemplo A100
  no cambió. `HF_TOKEN` se toma del entorno o del Kaggle Secret como en `infra/kaggle_baseline.py`, sin registrarlo.
- Casos/esperadas malformados: `ValueError` claro, manifiesto `fallida` y código 1, sin traceback.
- Tests de equivalencia `AgentePrompt` = `AgenteJSON`: tres respuestas sin JSON, `max_turnos`, límite de
  40 llamadas del `Entorno` real y lotes de distinto largo.
- Evidencia: `UV_PROJECT_ENVIRONMENT=<.venv sincronizado del checkout principal> uv run --locked --no-sync
  python -m unittest discover -s tests` → 169 tests OK (1 skipped), +19 nuevos. **Probado con dobles**:
  sin tokenizer real, sin GPU ni Kaggle; la carga real del modelo en T4 y el Secret no se ejercitaron.
- Próximo paso: correr plomería en Kaggle con autorización y, para seleccionar, un conjunto prompt-dev
  generado fuera de práctica/test.

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
- **Exploración reproducible:** `uv run --locked --no-sync python -m scripts.explorar_buscadores
  --salida corridas/rag-exploracion-20261110/original.json` reproduce todas las filas rechazadas de la guía.
- **Límite:** hiperparámetros elegidos con el mismo dev; no es evaluación independiente.
- **Próximo paso:** correr práctica con el agente `baseline` vs `prefijos` (k y tokens por caso) antes de
  cambiar el default.

### 2026-11-10: runner Kaggle registra el buscador (issue #6)

- **Estado:** implementado y probado con dobles. `runner.json` incluye `buscador` (variante efectiva de
  `BUSCADOR_VARIANTE`, default `baseline`, y su configuración); una variante desconocida falla antes de crear
  la salida, sin gastar GPU. Comprobado en CPU que el subproceso construye la misma variante que se registra.
- **Motivo:** la comparación `baseline` vs `prefijos` con el agente necesita trazar cada resultado a su buscador.
- **Evidencia:** suite 124 tests, 1 omitido opcional. Sin ejecución en Kaggle todavía.

### 2026-11-10: conversaciones de entrenamiento v2 (issue #8, CPU, rama `feat/8-conversaciones-eze`)

- **Estado:** implementado, probado por interfaz pública con los datos reales del repo y generado en
  `datos/conversaciones/v2-baseline/` (2.900 conversaciones: 800 `referencia`, 1.600 `historial`,
  500 `sintetico`; 0 excluidas). `datos/entrenamiento/` (versión de Yas) no se tocó.
- **Qué cambia:** nuevo origen `referencia` en el contrato v1; el validador acepta el formato de
  observación del prompt; el conversor de Fran recibe un `PromptSpec` (y `buscar` para re-ejecutar
  búsquedas; `--sistema` sigue como obsoleto). `scripts.generar_conversaciones` reconstruye los valores de
  cada traza desde sus líneas `Valores:`, re-ejecuta cada búsqueda con el buscador de la entrega sobre una
  copia de las normas con esos valores, revalida la ficha con `razones_normativas` y las costumbres, y
  genera sintéticos con valores sorteados entre los de la referencia. Detalle y reglas:
  [contrato](conversaciones-entrenamiento.md#dataset-v2-scriptsgenerar_conversaciones-issue-8).
- **Decisiones del equipo (después de la auditoría):** (1) buscador `baseline`, porque con el agente en
  Kaggle (50 casos, normas originales) resolvió 10/50 contra 6/50 de `prefijos`. (2) En 142 conversaciones
  reales de accidente de trabajo (49 de referencia y 93 de historial) la consulta combinada no traía
  NOR-COB-02; se agregó una búsqueda «accidente de trabajo ART» después de esa consulta, sin modificar las
  originales. (3) Se quitaron de `fundamentos` las secciones no recuperadas: 2.326 citas, una por conversación
  (NOR-COB-01 468, NOR-COB-02 659, NOR-COB-03 1.199). Motivo de (2) y (3): la misión pide que el modelo
  responda con la evidencia recuperada, no de memoria. Las dos transformaciones quedan registradas por
  conversación en `auditoria.jsonl` y contadas en el manifiesto.
- **Por qué:** con los valores de `normas/original`, 584 de las 800 fichas de referencia serían erróneas;
  las búsquedas deben mostrar los valores con los que se decidió cada ficha. El guion sintético anterior
  citaba secciones que no buscaba; ahora busca cada sección rectora y cada fundamento (500/500).
- **Evidencia:** `uv run --locked --no-sync python -m scripts.generar_conversaciones` (20 s, CPU);
  validador 2.900/2.900 sin errores (SHA-256 `3e6b8897…d2ee`); dos regeneraciones en carpetas temporales,
  `cmp` idéntico en `conversaciones.jsonl`, `auditoria.jsonl` y `manifiesto.json`; la fila reconstruida
  desde cada traza coincide con `historial.csv` en las 2.000 filas; ningún valor mostrado difiere de las
  trazas originales. Suite: `uv run --locked --no-sync python -m unittest discover -s tests` → 203 tests OK
  (1 omitido). Auditoría independiente del coordinador: 60 conversaciones y 188 búsquedas sin diferencias. El manifiesto registra `dirty=true`: se generó antes del commit.
- **Cobertura de evidencia tras el curado:** recuperan todas sus secciones rectoras 800/800, 1.600/1.600 y
  500/500, y ninguna conversación cita fundamentos no recuperados.
- **Límites:** sin tokenizador real (largos en caracteres: mediana ≈10,8 k reales, 15,6 k sintéticas,
  máximo 21,9 k); contaminación con práctica por identidad exacta de triage/números, no paráfrasis.
  No se entrenó ni se usó GPU. Si el prompt final cambia `formato_observacion`, `preparacion`/`tokenizacion`
  todavía validan con el formato por defecto.
- **Próximo paso:** regenerar después del commit para que el manifiesto tenga `dirty=false`, y tokenizar
  (con autorización) para medir los largos reales.

### 2026-11-10: `prefijos` contra `baseline` con el agente (issue #6, Kaggle T4)

- **Estado:** validado en GPU (Kaggle T4 x2), modelo **sin entrenar**. Notebook
  `franciscocalveyra/tp-admision-baseline` v22, commit `4e3613a` limpio, float16, `device_map=auto`,
  `MODELO_LOTE=4`, 50 casos de práctica, normas original; puntuado local con `admision.correr puntuar`.
- **Resultado:** `baseline` 10/50, parcial 0,831 (reproduce el baseline del 2026-11-04); `prefijos` 6/50,
  0,796. Gana 3 casos y pierde 7 (prueba de signos p≈0,34, no concluyente). Contexto por caso 19.954 →
  20.689 caracteres: con 1 a 3 secciones por búsqueda el agente recibe más texto de otras áreas, no menos.
- **Decisión (usuario):** default `baseline`, sin cambiar `entrega.py`; `prefijos` queda seleccionable como
  resultado negativo documentado. El dataset del #8 se genera con `baseline`.
- **Costo:** 2.968 s de sesión T4 (~20 min de `pip install`), más 642 s del humo v19. No comparable con A100.
- **Evidencia:** [comparación con el agente](buscador-baseline.md#comparación-con-el-agente-2026-11-10),
  Parte 2 de [INFORME.md](../INFORME.md#parte-2-buscador) y `corridas/kaggle/v22/` (local, ignorado por Git).
- **Próximo paso:** si alcanzan tiempo y cuota, repetir con el modelo entrenado.

### 2026-11-10: prompt, fase 1 (issue #13, Eze, CPU, rama `feat/13-prompt-eze`)

- **Estado:** implementado y probado por interfaz pública con dobles CPU. **No se corrió ningún
  candidato con Gemma ni en GPU**; no hay afirmaciones de calidad. #13 pasa de Martu a Eze.
- **Conjunto de desarrollo** `datos/prompt-dev/v1` (`python -m scripts.generar_prompt_dev`, ~2 s):
  semilla 20261110 (≠ 12012), 2 carpetas de normas (textos originales, valores sorteados entre los de las
  800 trazas de referencia, nunca b/c) × 20 casos con cuotas exactas por circuito, dificultad de
  cobertura y estrato del aviso. Esperadas del deductor fijo de Yas + `verificacion` en formato del kit;
  las 40 se resuelven 9/9 en el `Entorno` del kit. Sin pacientes de práctica (solo `casos.jsonl`), de las
  trazas de referencia/historial, de `datos/entrenamiento`, `datos/conversaciones/v2-baseline` ni de los
  5.000 candidatos de la semilla 12012 (1 candidato excluido). Esperadas fuera del alcance de la guarda del kit.
- **Candidatos** (solo cambia `system_prompt`; formato de #7 intacto): `corto-v1` (A), `procedimiento-v1` (B),
  `procedimiento-costumbres-v1` (C), `procedimiento-costumbres-corto-v1` (D). Motivo: en v22 35/50 casos
  buscaron una sola vez y los controles de box y aviso fallaron siempre por `false`. Ningún prompt trae
  valores de normas; el 75 de C/D es el umbral de una costumbre inferida del historial, no un parámetro.
- **Medición CPU** (`scripts.medir_contexto`, 50 trazas v22 rejugadas, mide formato y no conducta):
  sistema 2.232 → A 1.632 / B 3.224 / C 3.639 / D 2.405 caracteres; contexto por caso 19.954 → 17.002 /
  24.835 / 26.877 / 20.805. Tabla y tokens estimados en
  [optimización de prompt](optimizacion-prompt.md#medición-cpu).
- **Runner:** acepta todas las variantes del buscador (antes solo `baseline`/`sin_comunes`); nuevas
  configuraciones `evaluacion-prompt-kaggle-t4-l4.json` (lote 4, `max_turnos` 120 = tope de la entrega)
  y `...-prefijos.json`. `scripts.rankear_prompts` ordena por resueltos sumados y después tokens de sistema,
  y rechaza corridas no comparables o incompletas.
- **Kaggle preparado, no ejecutado:** celda única de fase 1 (5 prompts × 2 carpetas, ~1 h 35 min a 2 h de T4)
  y celda de fase 2 (ganador con `prefijos`, ~25–45 min) en la guía. Requiere autorización de GPU.
- **Evidencia:** `uv run --locked --no-sync python -m unittest discover -s tests` → 245 tests OK
  (1 omitido), 17 nuevos (`test_prompt_dev`, `test_seleccion_prompts`). El manifiesto del dev registra
  `dirty=true` (generado antes del commit); un test regenera el conjunto y exige bytes idénticos.
- **Abierto:** aceptar que las hipótesis salen del perfil agregado de fallas en práctica (la selección no);
  DSPy/GEPA postergado por costo de GPU; tokens reales de A–D recién con la corrida.
- **Próximo paso:** autorizar y correr la fase 1 en Kaggle, rankear y revisar caso por caso.

### 2026-11-10: evaluación completa y tabla de la Parte 4 (issue #9, CPU, rama `feat/9-evaluacion-eze`)

- **Estado:** implementado y probado con dobles; tabla validada con artefactos reales; **sin GPU**.
  Guía: [evaluacion.md](evaluacion.md).
- **Qué hay:** un plan JSON (`configuraciones/evaluacion/actual.json`: solo `sin_entrenar` × original/b/c × 50;
  `plantilla-cuatro-filas.json`, rechazada por el runner hasta completarla). `infra/kaggle_evaluacion.py` valida
  el plan entero antes de gastar GPU (sha256 de prompts, adapter con commit HF, etiquetas únicas) y corre cada
  combinación con `kaggle_baseline.correr` en su carpeta, con entorno propio; una falla no frena las demás.
  `scripts.tabla_evaluacion` puntúa con `admision.correr.puntuar` y escribe `tabla.md`/`tabla.json` con la
  procedencia de cada número, referencia de la cátedra rotulada y faltantes como «—».
- **Decisión:** no hay un segundo runner; el plan solo orquesta el existente y la procedencia sale de
  `runner.json`. El comando de jugar se reconstruye de `runner.json` (el runner no guarda su línea de comando);
  el plan sí guarda el suyo en `evaluacion.json`.
- **Evidencia:** `python -m scripts.tabla_evaluacion --corrida sin_entrenar=corridas/kaggle/v22/baseline-original-n50-l4
  --salida corridas/tablas/2026-11-10-v22-sin-entrenar` da 10/50, parcial 0,831, 19.954 caracteres/caso y
  15,16 s/caso, con `resultados.jsonl` idéntico byte a byte al puntuado previo; b y c «sin corrida».
  Kaggle T4, no comparable con A100. Tests: `tests/test_kaggle_evaluacion.py`, `tests/test_tabla_evaluacion.py`.
- **Limitación:** v22 no registró el prompt en `runner.json` (la tabla lo marca); v24 no tiene `runner.json`
  y se rechaza. El runner de planes no corrió en GPU.
- **Próximo paso:** con autorización, correr `actual.json` (~50–65 min de T4 estimados) y armar la tabla.

### 2026-11-10: análisis de errores por corrida (issue #14, CPU, rama `feat/14-analisis-eze`)

- **Estado:** implementado y probado con dobles (`tests/test_analizar_errores.py`, 9 tests con el `Entorno` real del
  kit y un buscador falso); aplicado a artefactos reales de Kaggle T4; **sin GPU**. Guía y registro de pasadas:
  [analisis-errores.md](analisis-errores.md).
- **Qué hay:** `python -m scripts.analizar_errores analizar --corrida <carpeta> --salida <nueva>` puntúa con
  `admision.correr.puntuar`, cuenta fallas por control y por tipo de caso y asigna a cada control fallado una fuente
  con reglas explícitas (formato > costumbre > buscador > modelo, con «no buscó» vs «buscó y decidió mal»). Las
  rectoras reutilizan `secciones_rectoras` del #8. `comparar --antes --despues` da ganados/perdidos, controles
  arreglados/rotos, cambios de fuente y de costo; rechaza conjuntos de casos distintos salvo
  `--permitir-interseccion`. Nunca pisa una salida.
- **Evidencia:** v22 baseline 10/50: 76 controles fallados = 0 formato, 16 costumbre (14 del aviso), 27 buscador
  (NOR-ADM-01 la rectora más ausente), 21 modelo sin buscar y 12 modelo con evidencia. Las tasas por control coinciden
  con `resumen.por_control` del kit y con los `resultados.jsonl`/`resumen.json` previos en v22 y v23. `prefijos`
  contra baseline: +3/−7 casos, fallas del buscador 27 → 30 y modelo sin buscar 21 → 33. v23 (`AgentePrompt`) contra
  v22: 0 diferencias. v24: 10 casos sin `runner.json`, idénticos a v22 en esos casos. Salidas en
  `corridas/analisis/2026-11-10-*` (local, ignorado por Git).
- **Decisión:** las conclusiones de práctica solo alimentan candidatos genéricos de #13 (buscar cada tema por separado,
  buscar cobertura y verificar, ART con lesión laboral); la selección va en prompt-dev (#13) y con datos permitidos
  (#15). Las corridas de #13 y #15 se analizan con este mismo comando y se registran como nuevas pasadas.
- **Limitación:** las fuentes son heurísticas (relevancia por palabras clave y secciones devueltas), sin precisión
  medida contra etiquetas humanas; las rectoras de cobertura del caso común amplían las del #8. v22 no registró el
  prompt; v24 no tiene procedencia.
- **Próximo paso:** analizar la primera corrida de candidatos de #13 en prompt-dev contra su baseline.

### 2026-11-11: prompt elegido y fila `solo_prompt` (issues #13 y #9, Kaggle T4 v25–v27)

- **Estado:** validado en GPU T4. Detalle, tablas y comandos en
  [optimización de prompt](optimizacion-prompt.md#resultados-kaggle-v25v27).
- **Fase 1, dev (40 casos, v25–v26):** B `procedimiento-v1` 9/40, `baseline` 8/40, D 2/40, A 1/40. C
  `procedimiento-costumbres-v1` termina con `CUDA error: unspecified launch failure` en 5 de 7 intentos, siempre en
  el mismo punto: no comparable. Acortar empeora; el procedimiento sube las búsquedas de ~1,3 a ~3,6 por caso y
  el contexto ~2,7×. Con `prefijos`, B cae a 4/40: se mantiene `baseline`.
- **Decisión:** prompt elegido B con el criterio declarado (resueltos en el dev, después tokens). En el dev la
  ventaja está dentro del ruido (signos p=1,0).
- **Práctica, solo reporte (v27, 3 × 50 casos):** `solo_prompt` 14/16/12 contra
  `sin_entrenar` 10/10/7 (v25), por encima de la referencia sin entrenar de la cátedra (12/15/12). Caso por caso:
  23 ganados y 8 perdidos (versiones no independientes). Sube box, plazo y registro; bajan verificación y entidad;
  el aviso no cambia. Análisis en [análisis de errores, pasadas 5–6](analisis-errores.md).
- **#8 regenerado con B:** `datos/conversaciones/v3-procedimiento` (~20 s CPU, 2.900 válidas, `dirty=false`,
  regeneración idéntica); frente a v2 solo cambia el mensaje de sistema. v2 queda igual.
- **Sin cambiar (decisión de equipo):** default de `entrega.py` (sigue `baseline`; recomendación: B) y qué dataset
  entrena #15 (Fran; la consigna pide el prompt optimizado, o sea v3).
- **Cuota:** v25 ~1 h 58 min, v26 1 h 3 min, v27 1 h 12 min de T4 (límites 2× la estimación); usado 5 h 52 min de 30 h hasta el 14/11.
- **Evidencia:** 259 tests CPU OK (1 omitido). Se quitó un prefijo de herramienta local que había quedado en
  `docs/evaluacion.md`.
- **Próximo paso:** decidir default y datos de #8 con Fran; si sobra GPU, diagnosticar el fallo de C con lote 1.

### 2026-11-11: primer SFT real, OOM en la pérdida (issues #3 y #15, Kaggle T4 v28)

- **Decisión:** un solo entrenamiento con `v3-procedimiento` (la consigna pide entrenar con el prompt optimizado,
  `mission.md:138`). El adapter se evalúa con `baseline` (fila `solo_entrenamiento`) y con B (fila `ambos`).
  Límite: `solo_entrenamiento` se mide con un prompt distinto al del entrenamiento y puede subestimarse.
- **Corrida v28** (commit `8843004` limpio, `sft-ejemplo.json`, 2.900 conversaciones, 363 pasos previstos):
  tokenización OK (9,99 M tokens, 1,02 M supervisados; mediana 3.331, máx. 6.322 por conversación).
  **Falló en el paso 0** por `torch.OutOfMemoryError` en `ForCausalLMLoss` (`logits.float()`, 3,29 GiB en GPU 0).
  Sin publicaciones HF. Unos 10 min de T4. Artefactos: `corridas/kaggle/v28` (no versionados).
- **Causa:** el modelo proyecta todas las posiciones al vocabulario de 262k, pero solo ~10 % está supervisado.
  Los humos usaron conversaciones cortas y no lo mostraron.
- **Arreglo:** `perdida_supervisada` con `logits_to_keep` booleano, en un `Trainer` propio
  ([detalle](entrenamiento-sft.md)). Probado con dobles y equivalencia numérica con torch CPU; **no validado en GPU**.
- **Corrida v29** (commit `10e117c`): falló en el paso 0 con `RuntimeError: indices should be either on cpu or on
  the same device as the indexed tensor (cuda:1)`. Con `device_map="auto"` la última capa queda en la segunda GPU y la
  máscara se había creado en la de `input_ids`. Arreglo: máscara en CPU (`e30ad97`). Unos 15 min de T4.
- **v30:** falló antes de cargar el modelo por timeout de W&B en `wandb.init` (~6 min). **v31** (con
  `WANDB_INIT_TIMEOUT=300`): repitió el error de dispositivo. La máscara en CPU no alcanza: el hook de accelerate
  (`hooks.py:192`) mueve todos los tensores del forward a la primera GPU. Arreglo: posiciones como lista de `int`,
  que `send_to_device` devuelve intacta (accelerate 1.15.0, `operations.py:174-176`); unión de posiciones del lote
  y `-100` donde una fila no tiene objetivo. Equivalencia con torch CPU revalidada.
- **v32** (commit `6169241`): **entrena en GPU real**, pero Kaggle la cortó por el límite de 8 h en el paso
  315/363 (~85 s/paso en 2 T4). Pérdida: 0,44 al inicio, ~0,03 en el paso 100 y ~0,005 en el paso 268. W&B run
  `3add2742ddfd4b42`. Checkpoints publicados en `FranCalveyra/tp-admision`: 50 `0829fa33`, 100 `4a8f7c09`,
  150 `a37aae49`, 200 `6ed633b9`, 250 `af2d6061`, **300 `83dd1d205763e4885b66ac14b177799785b3fd69`**.
  Recibos en `corridas/kaggle/v32b` (no versionado). Faltan 63 pasos (~1,5 h) para la época completa.
- **v33 (2026-11-12):** reanudó desde checkpoint-300 (mismo commit y run W&B `3add2742ddfd4b42`) y **terminó la
  época**: `terminada`, returncode 0, 1 h 31 min. Pérdida final ~0,006. VRAM pico 19,97 GB sumados (12,1 + 7,87 en
  2 T4). Publicó checkpoint-350 `00175ae4`, checkpoint-363 `fc35d543` y **adapter `final`
  `dc41674a8a2561a724884be2170ff6808efa6e96`**. Recibos en `corridas/kaggle/v33` (no versionado).
  Validado en GPU T4; **A100-40GB sigue sin validar**. Pérdida baja en train no prueba calidad del agente.
- **v34 (2026-11-12), evaluación del adapter `final`** (plan `configuraciones/evaluacion/entrenado.json`, commit
  `3adf12e`, 6 corridas terminadas, ~2 h de T4). Puntuado con `scripts.tabla_evaluacion` en
  `corridas/tablas/20261112-entrenado` (no versionado). Práctica solo para reportar.

  | Fila | original | b | c |
  |---|---|---|---|
  | sin_entrenar (v25) | 10 | 10 | 7 |
  | solo_prompt (v27) | 14 | 16 | 12 |
  | solo_entrenamiento (v34) | 11 | 13 | 12 |
  | **ambos (v34)** | **30** | **29** | **33** |
  | cátedra: SFT simple | 25 | 24 | 25 |

  `ambos` supera la referencia SFT de la cátedra en las tres versiones de normas, así que el aprendizaje
  sobrevive al cambio de normas. El adapter solo rinde con el prompt con el que se entrenó: con `baseline`
  apenas mejora. En `ambos`, todos los controles quedan ≥ 0,90 salvo `aviso_al_voluntariado` (0,72 en las
  tres versiones), que es la falla principal que queda. Costo: ~25 s/caso y ~51 k caracteres de contexto,
  contra ~19 s y ~29 k con `baseline`. Validado en T4, no en A100.
  La tabla de 4 filas en un solo `tabla.md` necesita las salidas de v25/v27, que no están en este checkout.
- **Default de la entrega (2026-11-12):** `entrega.py` carga `procedimiento-v1` y `Configuracion.desde_entorno`
  usa el adapter `FranCalveyra/tp-admision@dc41674a/final` si no hay `ADAPTER_REPO`; `ADAPTER_REPO=` vacío pide la
  base sola. El runner de evaluación vacía `ADAPTER_*` en las filas sin adapter. 274 tests CPU OK (1 omitido).
  **Bloqueo:** el repo HF debe seguir privado (mission.md:194) y compartirse con la cátedra; sin la invitación
  (#16, faltan sus usuarios) su `HF_TOKEN` no podrá leer el adapter.
- **PR #24** abierto con la rama (`fix/3-perdida-supervisada` → `main`).
- **Pasada 7 (#14, CPU):** `ambos` contra `solo_prompt` gana 16/13/21 casos y no pierde ninguno. Las fallas de
  verificación y entidad desaparecen. El aviso queda igual: el modelo responde `false` en todos los 75+ sin
  acompañante, aunque el entrenamiento tiene la regla completa (302 positivos). Detalle e hipótesis en
  [análisis de errores](analisis-errores.md#2026-11-12--pasada-7-fila-ambos-contra-solo_prompt-en-práctica-solo-reporte).
  Salidas de v27 descargadas de Kaggle a `corridas/kaggle/v27` (no versionado).
- **Hipótesis del aviso confirmada en el dev (Kaggle v35, ~40 min de T4):** adapter + `procedimiento-v1` resuelve
  26/40 (sin adapter: 9/40); nunca responde aviso `true`, y los 8 mayores de 75 sin acompañante fallan solo por eso.
  La evaluación de prompts ahora acepta un `adapter` opcional (`configuraciones/evaluacion-prompt-kaggle-t4-l4-entrenado.json`).
- **Remedio elegido (2026-11-12, Fran): seguir entrenando el adapter** en vez de reentrenar (≈9,5 h, casi toda la
  cuota) o escribir la regla en el prompt (la consigna dice que las costumbres las enseña el entrenamiento).
  `adapter_inicial` en la configuración del SFT; `datos/conversaciones/v4-aviso` (633: 302 positivos, 231 negativos
  de 75+ con acompañante, 100 del resto; semilla 15); `configuraciones/sft-aviso.json` (lr 1e-4, 1 época). Ver
  [entrenamiento-sft](entrenamiento-sft.md#5-seguir-entrenando-un-adapter-publicado-adapter_inicial).
- **Kaggle v36 lanzado** (commit `fb353c0`, T4 x2, tope 5 h): entrena y evalúa el adapter nuevo en el dev en la
  misma sesión. Publica checkpoints en el repo HF privado y una corrida nueva en W&B.
- **Próximo paso:** decidir **solo con el dev** si el adapter de la v36 reemplaza al de la entrega: contra v35
  (26/40, aviso 32/40), que mejore el aviso sin romper otros controles. Recién después, y sea cual sea la
  decisión, correr la práctica solo para reportar; su resultado no cambia la elección (revisión del PR #24).
- **README "cómo correr" (#16, sin GPU):** sección del grupo al principio de `README.md` con los seis puntos de
  mission.md:224 (entorno, un caso, buscador, evaluación completa, de dónde sale cada archivo, diagrama Mermaid);
  el manual del kit queda abajo con sus anclas intactas. Comandos tomados de los manifiestos; no se corrieron en
  A100. **Pendiente:** `.env.example` todavía tiene `ADAPTER_REPO=` vacío (exportarlo pide el modelo base sin
  adapter) y el comentario de `PROMPT_SPEC` dice baseline; el README avisa, pero hay que comentar esas líneas.
  Si v36 cambia el adapter de la entrega, actualizar la revisión en la tabla del README.
- **#5 recursos (borrador `docs/informe-parte5.md`, sin GPU):** agregados el SFT real (2.900 conversaciones,
  ~10 M tokens, 363 pasos a ~80–85 s, ~8,1 h de cómputo, 19,97 GB pico en 2 T4) y el antes/después en práctica
  (27/150 → 92/150; +59 % s/caso y 2,4× tokens de entrada por el prompt, VRAM 18,1 → 19,6 GB; ~145 casos/h por
  proceso con la entrega). Todo en T4: A100 sigue pendiente (KI-A100-001). Si v36 cambia el adapter, rehacer la
  fila `ambos`.
- **Formato de los mensajes (#15, fase 3, CPU):** candidato `procedimiento-formato-v1` (mismo sistema que B, cambia
  observación y JSON inválido) con test; plan en [optimización de prompt](optimizacion-prompt.md#fase-3-2026-11-12-formato-de-los-mensajes-sin-entrenar).
  Se mide en el dev sin adapter después de v36. No puede ir a la entrega sin reentrenar. Respaldo sin GPU:
  reportar el análisis de tamaño (≤1,4 %).
- **Auditoría de la entrega contra mission.md** ([auditoria-entrega.md](auditoria-entrega.md), CPU): el hueco más
  grave era que ningún número era auditable (trazas solo en `corridas/`, ignorado). **Resuelto:** las evaluaciones
  reportadas (v25, v27, v34 en práctica; v25 fase 1 y v35 en el dev) quedan versionadas en `evaluaciones/` sin
  editar (escaneadas: sin secretos), y la tabla de la Parte 4 se regeneró desde ahí con procedencia limpia
  (`evaluaciones/practica/tabla`, 10/10/7 · 14/16/12 · 11/13/12 · 30/29/33).
- **Lote de la entrega = 4 (decisión de Fran, 2026-11-12):** `Configuracion.desde_entorno` usa `LOTE_ENTREGA = 4`
  sin `MODELO_LOTE`, para que la entrega corra como se midieron todos los números (antes: 1, ~2× más lento por caso
  en T4). `Configuracion()` sigue en 1. Riesgo: lote 4 en A100 (una GPU, bf16) sin medir; en T4 los fallos fueron
  con 8 y 16, repartido en 2 GPU.
- **Revisión del PR #24 (estándar + especificación, 2026-11-12):** arreglado (1) la práctica no decide la v36:
  solo el dev, y el límite de haber visto la falla en práctica queda declarado; (2) fila desincronizada de la
  auditoría; (3) `ADAPTER_REVISION`/`ADAPTER_SUBCARPETA` sin `ADAPTER_REPO` ahora fallan en vez de cambiar en
  silencio el adapter de la entrega; (4) `validar_adapter` en `configuracion.py` es la única validación de la
  referencia `{repo, revision, subcarpeta}` (preparación del SFT, evaluación de prompts y planes). Quedan
  documentados, sin arreglar: `AdapterRef` como tipo, internos de `preparacion` usados por
  `subconjunto_aviso`, `TrainerSupervisado` anidada.
- **`scripts.ver_caso`** reemplaza los `jq` del README («un caso»): muestra conversación, búsquedas con las
  secciones que trajo el buscador, ficha y 9 controles; sirve también para la demo del Loom. Borrador del mail de
  entrega con su checklist en [mail-entrega.md](mail-entrega.md).
- **v36 (refuerzo del aviso) NO adoptada (decisión de Fran, solo con el dev):** 23/40 contra 26/40 de v35; el
  aviso pasa de 0/8 a 6/8 en mayores sin acompañante pero da `true` en 12 de 32 que no corresponde
  ([pasada 8](analisis-errores.md)). La entrega sigue con `dc41674a`; la práctica de la v36 no se corrió. Validó en
  GPU `adapter_inicial`. Corridas y recibos versionados en `evaluaciones/` (también los del SFT v32/v33).
- **Fase 3, formato de los mensajes (Kaggle v37, ~20 min de jugar):** `procedimiento-formato-v1` sin adapter da
  10/40 contra 9/40 de B en el dev (2 ganados, 1 perdido; controles que se compensan); contexto −2 %, tiempo −8 %.
  Dentro del ruido: **no se adopta** (además exigiría reentrenar). Versionado en `evaluaciones/dev/v37-formato`.
- **PR #24 mergeado a `main` (`49e8c85`, 2026-11-12) y #3 cerrado.** La entrega en `main`: `procedimiento-v1` +
  adapter `dc41674a` + lote 4. Pendientes: acceso de la cátedra a HF/W&B (#16), A100 (KI-A100-001, #5),
  `.env.example` (Fran), INFORME y Loom (equipo).
- **Motivo de no usar Modal, explícito donde lo lee la cátedra** (Known Issue del README y Parte 5): habilitar su
  crédito exige cargar una tarjeta de crédito y decidimos no hacerlo.
- **`.env.example` corregido (Fran):** `ADAPTER_*` comentadas (exportarlo ya no apaga el adapter), comentario de
  `PROMPT_SPEC` al día y `MODELO_LOTE=4`. Cierra el hueco 5 de la auditoría.
- **#14 cerrado.** **#15, peso en el token del aviso (rama `feat/15-peso-aviso`, CPU):** `peso_aviso` en la
  configuración del SFT; `pesos_del_aviso` ubica el valor del aviso y falla si no está una vez; datos
  `v5-aviso-peso` (27 % positivos) y `sft-aviso-peso.json` (peso 20, lr 1e-4). Probado con dobles (293 tests).
  Siguiente: entrenar y medir en el dev; se adopta solo si supera 26/40 sin romper el resto.
- **v38 (peso 20 en el token del aviso) NO adoptada (solo con el dev):** Kaggle v38 entrenó 142 pasos (11.750 s,
  20,29 GB pico, W&B `f9326b10342b4be4`, `final` `17e9fe3d`, commit `984fc18`) y dio 26/40, igual que v35 (aviso
  30/40 contra 32, 1/8 positivos, 3 falsos `true`, y cobertura 3, registro 1 y `box_antes_del_registro` 1 rotos;
  ganó `pd-n2-002`, `pd-n1-008`, `pd-n1-012` y perdió `pd-n2-006`, `pd-n2-009`, `pd-n2-017`). La regla pedía superar
  26: la entrega sigue con `dc41674a` y la práctica de la v38 no se corrió ([pasada 9](analisis-errores.md)).
  Corridas en `evaluaciones/dev/v38-peso`, recibos en `evaluaciones/entrenamiento/v38-peso`; guía en
  [entrenamiento](entrenamiento-sft.md). Sin probar: DPO con pares que difieran solo en el aviso.
- **#15, datos v6 con la edad y el acompañante antes de la ficha (rama `feat/15-aviso-datos`, CPU):** hipótesis:
  el adapter no aprende el aviso como un solo token, pero sí puede copiar los dos hechos de la nota y decidir
  después. El último mensaje (`registrar_decision`) pasa a `Edad: N años. Acompañante: X.` + salto de línea + JSON;
  el validador acepta ese prefijo solo ahí y `extraer_json` de la cátedra ya lo lee. El umbral no se escribe: los
  valores se copian de la nota. Datos `v6-aviso-datos` (1.133 registros válidos, 0 errores, manifiesto con
  `dirty=false`) con `scripts.anteponer_datos_aviso` sobre `v5-aviso-peso`; `sft-aviso-datos.json` es copia exacta
  de `sft-aviso-peso.json` (v39 difiere de v38 solo en el prefijo). Probado con dobles; sin tokenizar con el modelo
  real.
- **v39 (edad y acompañante antes de la decisión) ADOPTADA como adapter de la entrega (2026-11-13, solo con el dev):**
  Kaggle v39 continuó `dc41674a` 142 pasos sobre `v6-aviso-datos` (11.735 s, 20,36 GB pico, W&B `41ec01f292e64306`,
  código `5a36ecf`, `final` `770c9fac84d5718b0b2cf4c40c3451cad322ac9e`) y dio **31/40 en el dev contra 26/40 de v35**
  (5/8 positivos atrapados contra 0/8, 2 falsos `true`; ganó 7 casos y perdió `pd-n1-010` y `pd-n2-013`). Superó la barra
  acordada antes (≥30/40, con ganancias en los 75+ sin acompañante). El prefijo apareció en 38 de 40 mensajes finales y
  `extraer_json` de la cátedra lo lee. Quedan fallas de aviso (6), cobertura (4) y `box_antes_del_registro` (3)
  ([pasada 10](analisis-errores.md)). `ADAPTER_ENTREGA` apunta a `770c9fac` (commit `1405aab`) y el plan
  `configuraciones/evaluacion/entrenado-v39.json` fija las filas con adapter. Corridas en `evaluaciones/dev/v39-datos`,
  recibos en `evaluaciones/entrenamiento/v39-datos`; guía en [entrenamiento](entrenamiento-sft.md).
  Práctica con el adapter v39: ver la entrada de Kaggle v40 más abajo. Probado con dobles el prefijo; validado en GPU
  (T4) el entrenamiento, el dev y la práctica; A100 sigue sin validar.
- **Práctica con el adapter v39 (Kaggle v40, 2026-11-13, solo para reportar):** plan
  `configuraciones/evaluacion/entrenado-v39.json`, `FranCalveyra/tp-admision@770c9fac/final`, T4 x2, lote 4; salidas en
  `evaluaciones/practica/v40-entrenado-v39/` (escaneadas: sin secretos) y tabla vigente en
  `evaluaciones/practica/tabla-v39` (`sucio=False`; la del adapter `dc41674a` queda en `practica/tabla`).
  - Resueltos original / b / c: `sin_entrenar` 10/10/7, `solo_prompt` 14/16/12, `solo_entrenamiento` 17/14/20 (51/150;
    antes 36/150), **`ambos` 36/28/36 = 100/150** (antes 92/150). Referencia de la cátedra: SFT perfecto 25/24/25.
  - Honesto sobre b: `ambos` perdió un caso (29 → 28). Ganó `pev-008`, `pev-025`, `pev-031`, `pev-042` y `pev-043` en
    las tres versiones y perdió `pev-007`, `pev-021`, `pev-022` y `pev-045` en las tres. Control de aviso 0,82 / 0,70 / 0,78
    contra 0,72 / 0,72 / 0,72.
  - Costo: `ambos` pasó de ~19-25 a 27,8–28,8 s/caso en T4 (lote 4) por los tokens extra y ~6,7 llamadas por caso.
    `solo_entrenamiento` tiene 3/3/1 casos sin decisión.
  - No se usó para elegir: el adapter ya estaba adoptado con el dev. Nota en [análisis de errores](analisis-errores.md)
    (pasada 10). Tokens y VRAM de las filas v39 no se recalcularon. Validado en GPU (T4); A100 sigue sin validar.
- **#5 cerrado (2026-11-13), alcance T4:** lo medible está en el [borrador de la Parte 5](informe-parte5.md):
  - SFT real: tiempo, VRAM y costo en Kaggle.
  - Casos a la vez al entrenar y al evaluar.
  - Proyección a producción, como hipótesis.
  - Antes/después: de 27/150 a 100/150 en práctica con el adapter v39 (92/150 con el anterior).
  - Trabajos futuros.

  La A100-40GB queda como **limitación conocida**, no como tarea: no hay vía de acceso y Modal se descartó
  ([KI-A100-001](../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb)). En #16
  quedó anotada así, para mencionarla en el mail y en el informe; la integración al `INFORME.md` sigue en #16.

### 2026-11-13: `INFORME.md` completo (issue #16, CPU, rama `docs/informe-completo`)

- **Estado:** `INFORME.md` reorganizado en resumen + Partes 1 a 5 + limitaciones y trabajos futuros, con cada pasada
  (incluidas las fallidas: v28–v31, v36, v37, v38, `prefijos`, prompts A/C/D) y la procedencia de cada número. La
  Parte 5 integra [informe-parte5.md](informe-parte5.md), que queda como detalle.
- **Celdas «sin recalcular» de la Parte 5 completadas** con un cálculo propio sobre
  `evaluaciones/practica/v40-entrenado-v39/*/metricas.jsonl` (el mismo cálculo reproduce las filas de v34): `ambos` v39
  14.101–14.196 tokens de entrada y 353–355 generados por caso, 19,94 GB pico, ancho 4.395; `solo_entrenamiento` v39
  12.146–14.204 / 333–338, 20,25–21,71 GB, ancho 7.636 (debajo del tope de 8.192). Rendimiento de la entrega:
  ~125–129 casos/h por proceso en T4 (antes ~145 con `dc41674a`).
- **Pendiente declarado:** todo lo de A100-40GB (KI-A100-001); sin GPU, build ni commit en esta unidad.
- **Correcciones de la auditoría aplicadas:** `v4-aviso` es 302/633 = 47,7 % positivos; ~350 supervisados por
  conversación; carga del modelo 20,5–155,0 s según las corridas versionadas; fila de párrafos del buscador separada;
  origen (comando, configuración y checkpoint) en 5.1, 5.3 y la fase 1 de la Parte 3; anexo de la Parte 5 al día con v39/v40.
- **Evidencia de recursos versionada:** las corridas que citan la Parte 5 y su anexo (`base-{original,b,c}-l4`,
  `lote-original-l{1,8,16}`, `diag-l8`, `solo-texto-l4` y los humos de entrenamiento, 4,7 MB) pasaron de `corridas/kaggle/`
  a `evaluaciones/recursos/` sin editar. El escaneo de secretos dio limpio: solo aparece el aviso de W&B que nombra la
  variable, sin ningún valor, y no hay pesos. Sigue sin evidencia la comparación de buscadores con el agente de la Parte 2
  (`corridas/kaggle/v22` y `corridas/rag-*-20261110` no existen ni en local).

