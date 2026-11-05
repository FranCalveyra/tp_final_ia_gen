# INFORME, Parte 5: recursos y proyección a producción (BORRADOR)

> **Estado: borrador del issue #5. No es el informe final.** Todo lo medido hasta hoy (2026-11-04)
> salió de **Kaggle 2x T4 con fp16**, con Gemma 4 E4B **sin entrenar** (o con datos de humo
> sintéticos). **Nada se midió en A100-40GB ni con el modelo final.** Esos números figuran como
> **PENDIENTE**. Los números de T4 no se pueden transferir a A100 y acá no inventamos factores.
> Para pegar en `INFORME.md` cuando el equipo lo cree; la bitácora vive en [progreso.md](progreso.md).

## Resumen (lo que ya se puede afirmar)

| Pregunta de la consigna | Respuesta hoy | Estado |
|---|---|---|
| Tiempo por caso (inferencia) | 33,3 s (lote 1) y 15,6 s (lote 4) en T4 | validado en GPU (T4) |
| VRAM de inferencia | 16,7 GB (lote 1) a 19,1 GB (lote 4), suma de 2 T4 | validado en GPU (T4) |
| Casos a la vez al evaluar | 1 y 4 válidos; **8 y 16 se caen** (CUDA, sin diagnosticar) | validado en GPU (T4) |
| Casos a la vez al entrenar | 1 conversación por paso (`batch_size=1`, acumulación 1) en el humo | validado en GPU (T4), solo humo |
| Tiempo y VRAM de entrenamiento | humo de 8 pasos: 17,4 s; VRAM de entrenamiento no registrada en el humo (ya se registra) | humo en T4; real: PENDIENTE |
| Costo | Kaggle: 0 USD. A100: fórmula, sin medición | PENDIENTE (A100) |
| Simultáneos en producción | proyección cualitativa, ver sección 5 | hipótesis, A100: PENDIENTE |
| Antes/después con el modelo final | no existe aún | PENDIENTE |

**Leyenda de validación.** *implementado*: existe el código. *probado con dobles*: tests sin GPU.
*validado en GPU (T4)*: corrió en Kaggle. *validado en A100*: **ninguno todavía**.

**Cómo leer este documento:** secciones 1 a 4 son medición (T4); 5 es proyección (hipótesis);
6 lista lo pendiente; 7 son trabajos futuros; 8 es el origen de cada número.

---

## 1. Medición de inferencia (T4 x2, Gemma 4 E4B sin entrenar)

Condiciones comunes: fp16, `device_map=auto` en 2 T4 de 15 GB (30 GB en total), torch
2.15.0+cu132, 50 casos de práctica, tope 512 tokens nuevos por llamada y 8192 de contexto
(`runner.json`). Puntuado con `admision.correr puntuar`. **Una sola corrida por configuración.**

### 1.1 Línea de base: 50 casos, lote 4

| Normas | Resueltos | Parcial medio | s/caso | `construir_agente` | Micro-lotes | Tokens/s |
|---|---|---|---|---|---|---|
| original | 10/50 | 0,831 | 15,6 | 145,6 s (incluye bajar el modelo) | 65 | 17,4 |
| b | 10/50 | 0,820 | 15,8 | 47,2 s | 64 | 17,2 |
| c | 7/50 | 0,816 | 16,1 | 34,1 s | 64 | 16,8 |

Fuentes: `corridas/kaggle/base-{original,b,c}-l4/{puntuacion.json,salida/tiempos.json,metricas.jsonl}`.
Tokens/s = tokens generados sobre segundos de `generate`, sumados sobre los micro-lotes
(cálculo propio). Los resueltos no son comparables con la referencia de la cátedra (12, 15, 12,
medida en otra GPU y otra precisión); acá importan los recursos.

### 1.2 Por caso, no solo promedios (variante original, lote 1)

Cada caso es una conversación de 3 a 8 llamadas al modelo. Mínimo / mediana / máximo sobre los
50 casos (cálculo propio sobre `puntuacion/resultados.jsonl` y `salida/trazas.jsonl`):

| Magnitud | Mín | Mediana | Máx | Fuente |
|---|---|---|---|---|
| Llamadas al modelo por caso | 3 | 5 | 8 | `lote-original-l1/puntuacion/resultados.jsonl` |
| Búsquedas por caso | 1 | 1 | 3 | ídem |
| `caracteres_de_contexto` (lo que informa `admision.correr`) | 9.447 | 17.820 | 53.810 | ídem |
| Caracteres de la conversación completa | 4.558 | 6.409 | 11.400 | `lote-original-l1/salida/trazas.jsonl` |
| Mensaje de sistema (caracteres) | 2.232 | 2.232 | 2.232 | ídem |
| Tokens de entrada por llamada | 651 | 955,5 | 3.072 | `lote-original-l1/metricas.jsonl` |
| Tokens generados por llamada | 24 | 45 | 145 | ídem |
| Segundos por llamada (lote 1) | 3,2 | 5,4 | 17,7 | ídem |

- Promedios en la corrida de lote 1: 4,9 llamadas y 1,34 búsquedas por caso; 1.169 tokens de
  entrada y 55,2 generados por llamada (`puntuacion.json`, `metricas.jsonl`).
- El mensaje de sistema es ~35 % de los caracteres de la conversación en la mediana (2.232 de
  6.409, cálculo propio; la consigna menciona un quinto en la referencia de la cátedra). Se repite
  en cada llamada: es el primer candidato para recortar (sección 7).
- La entrada más larga (3.072 tokens) marca el ancho máximo de secuencia visto; sale de casos con
  3 búsquedas. El tiempo por caso de `trazas.jsonl` es el reloj acumulado de la corrida, no el
  costo propio del caso: no lo usamos.
- No hay un recorte de contexto silencioso: 0 llamadas inválidas en las cuatro corridas completas
  (`invalidas` en `resultados.jsonl`).

### 1.3 Tiempo de carga

`construir_agente` fue 146 s en la primera corrida de cada sesión (baja el modelo) y 34 a 50 s con el
modelo en caché (`salida/tiempos.json` de cada corrida). Es un costo fijo por proceso: pesa en
producción si las instancias se levantan seguido (sección 5).

---

## 2. Casos a la vez al evaluar (lote)

Mismo conjunto (original, 50 casos), solo cambia `MODELO_LOTE`. Commit `11bf3d6` limpio para 1, 8
y 16; el lote 4 viene de otra sesión de Kaggle (commit `a89f5b9`), con posible variación de hardware.

| Lote | Estado | s/caso | VRAM pico suma (GPU0 + GPU1) | VRAM reservada suma | Tokens/s | Micro-lotes | Segundos por micro-lote |
|---|---|---|---|---|---|---|---|
| 1 | ok, 10/50 | **33,3** | 16,7 GB (9,6 + 7,1) | 20,5 GB | 8,2 | 246 | 6,8 |
| 4 | ok, 10/50 | **15,6** | 19,1 GB (10,9 + 8,2) | 29,2 GB | 17,4 | 65 | 12,0 |
| 8 | **cayó** (28 micro-lotes) | inválido | 20,2 GB (11,5 + 8,7) | 30,3 GB | 25,3 | 28 | 13,5 |
| 16 | **cayó** (10 micro-lotes) | inválido | 22,7 GB (12,8 + 9,8) | 28,0 GB | 34,5 | 10 | 15,1 |

Fuentes: `corridas/kaggle/{lote-original-l1,base-original-l4,lote-original-l8,lote-original-l16}/{salida/tiempos.json,metricas.jsonl,runner.json}`.
VRAM = máximo por GPU sobre los micro-lotes, sumado (cálculo propio). Los s/caso de 8 y 16 son de
corridas cortadas: no sirven para comparar.

**Lo que se concluye con evidencia**
- Lote 1 a 4: 2,1 veces más rápido (33,3 a 15,6 s/caso) por 2,4 GB más de VRAM pico. Sin
  regresiones: los mismos 10 casos resueltos, primera respuesta idéntica en 50 de 50 y conversación
  completa idéntica en 49 de 50 (bitácora, 2026-11-04).
- Cada micro-lote tarda 6,8 s con 1 conversación y 12,0 s con ~3,8: el cómputo por conversación
  baja (6,8 a 3,1 s), pero la **latencia de cada llamada casi se duplica**. Importa en producción.
- Lote 8 y 16: `AcceleratorError: CUDA error: unspecified launch failure` dentro de
  `project_per_layer_inputs` (`salida/tiempos.json`, campo `error_detalle`). El lote 8 dejó 34 casos
  sin decisión y el 16, los 50 (`puntuacion/resumen.json`). **Causa sin diagnosticar** (no se probó
  `CUDA_LAUNCH_BLOCKING=1`); se está investigando aparte.
- Observación, **no diagnóstico**: la VRAM *reservada* (caché de PyTorch) llegó a 30,3 GB en el lote 8
  sobre 30 GB totales de las dos T4, aunque el pico *asignado* por GPU fue 11,5 GB. No sabemos si
  hay relación con el fallo; no usar "sobran 10 GB por GPU" como argumento sin ese dato.

**Al evaluar en la A100-40GB:** lote por defecto 1, sin medir. **PENDIENTE** repetir 1/4/8/16 en
A100-40GB y fijar el lote de la entrega con evidencia (el default del código no cambió).

---

## 3. Entrenamiento: tiempo, VRAM y casos a la vez

Lo único medido es el **humo** (conversaciones sintéticas de `scripts.datos_humo`: prueban el
pipeline, **no** la calidad ni los tiempos del entrenamiento real). Kaggle T4 x2, fp16,
`device_map=auto`, commit `acc95b2` limpio, LoRA solo en la atención del modelo de lenguaje
(r=8, alpha 16), `corrida.json`.

| Corrida | Qué hizo | Tiempo del runner | `train_runtime` | Pérdida paso 1 a 8 |
|---|---|---|---|---|
| v6 (`humo-entrena-1-v6`) | desde cero, 8 pasos, checkpoint cada 2 | 163 s | 17,4 s (2,18 s/paso) | 2,679 a 2,178 |
| v7 (`humo-entrena-2-v7`) | reanuda desde `checkpoint-2` bajado de HF | 177 s | 15,8 s (pasos 3 a 8) | pasos 3 a 8 iguales a v6 |

Fuentes: `corridas/kaggle/humo-entrena-{1-v6,2-v7}/{runner.json,runner.log,entrenamiento/checkpoint-*/trainer_state.json}`.
El tiempo del runner excluye la instalación de dependencias; la carga de pesos tardó ~43 s
(`runner.log`). La bitácora reporta 7 y 9 min de sesión completa.

- **Casos a la vez al entrenar:** `batch_size=1`, `gradient_accumulation_steps=1`, 2 épocas,
  `max_seq_length=8192` (`entrenamiento/corrida.json`). O sea, **1 conversación por paso**. Es la
  configuración del humo, no una decisión de la corrida final.
- **VRAM de entrenamiento (humo, solo texto): 15,7 GB pico** (9,5 + 6,1 por GPU; 15,8 GB reservada),
  de `corridas/kaggle/humo-solo-texto/runner.json` (`vram`). Son 4 conversaciones sintéticas cortas con
  batch 1: **no representa** la corrida real. **PENDIENTE:** el valor con secuencias reales, que el
  entrenador ya registra en W&B, `estado.json` y `runner.json`.
- **Checkpoints y publicación:** cada checkpoint se sube a HF con revisión, y la reanudación
  reprodujo la curva (2,5196/2,4173/2,3468 contra 2,5196/2,4179/2,3464). Pasa en T4, no en A100.
- **Tiempo total del entrenamiento real: PENDIENTE.** Depende de cuántas conversaciones tenga el
  dataset final (Eze/Yas), de su largo en tokens y del prompt optimizado (Martu). Fórmula:
  `pasos = conversaciones x épocas / (batch_size x acumulación)`; `tiempo = pasos x s/paso` +
  carga (~44 s en T4) + subidas. El 2,18 s/paso del humo **no** sirve para extrapolar: eran
  secuencias cortas y sintéticas.
- El humo gasta poco: unos 6 min de GPU en total para las dos corridas (163 + 177 s), más
  instalación y carga por sesión. Los intentos fallidos v4 y v5 (157 y 156 s,
  `humo-entrena-1/runner.json` y `humo-entrena-1-v5/runner.json`) también consumen cuota.

---

## 4. Costo de punta a punta

### 4.1 Lo medido: Kaggle (desarrollo)

Kaggle es gratis; su cuota ronda las 30 h/semana de GPU, aproximada y fijada por Kaggle
([kaggle.md](kaggle.md)). Uso, derivado de `runner.json` (inicio a fin del runner, sin
instalación del entorno):

| Trabajo | GPU-segundos | Fuente |
|---|---|---|
| Línea de base original, b y c (lote 4) | 930 + 840 + 841 = 2.611 s (0,73 h) | `base-*-l4/runner.json` |
| Barrido de lotes 1, 8 y 16 | 1.822 + 434 + 193 = 2.449 s (0,68 h) | `lote-original-l{1,8,16}/runner.json` |
| Humo de entrenamiento (v6 + v7) | 163 + 177 = 340 s (0,09 h) | `humo-entrena-{1-v6,2-v7}/runner.json` |

La bitácora estima ~2.690 s de cuota para el barrido, porque suma el arranque de la sesión.
Hay otras corridas (`t4-original-09`, intentos fallidos anteriores) sin `runner.json` completo:
**el total real de cuota usado es mayor y no está auditado**. Costo en dinero: **0 USD**.

### 4.2 La proyección: Modal A100-40GB (HIPOTÉTICA, sin medición)

`mission.md`: A100-40GB ~2,10 USD/h y 30 USD de crédito por integrante y por mes, o sea unas
14,3 h por persona (30 / 2,10). La facturación de Modal está bloqueada por la tarjeta, así que
**no corrió nada en Modal**.

```
costo = horas de GPU x 2,10 USD/h
costo por caso = (segundos por caso / 3600) x 2,10 + carga del modelo amortizada
```

Los s/caso de T4 **no se transfieren** a A100 y no asumimos una aceleración. Para dar escala,
la fórmula aplicada a tiempos **de T4**, solo como referencia de orden y no como predicción:

| Si el caso tardara... | Costo por caso | Costo de 50 casos |
|---|---|---|
| 15,6 s (lote 4 en T4) | 0,0091 USD | 0,46 USD |
| 33,3 s (lote 1 en T4) | 0,0194 USD | 0,97 USD |

Cargar el modelo (34 a 146 s en T4) equivale a 0,02 a 0,09 USD por arranque a 2,10 USD/h.
**PENDIENTE (A100):** s/caso, carga, costo por caso, costo del entrenamiento real y costo total de
desarrollo. **Mantenimiento:** hoy no hay datos; se reportará como costo por caso x volumen
esperado cuando exista el s/caso en A100.

---

## 5. Producción: cuántos casos simultáneos y cómo escalar (HIPÓTESIS)

Nada de esto se validó en A100. Es un razonamiento sobre las mediciones de T4, con incertidumbre
explícita.

**Qué es "simultáneo".** Un caso es una conversación de ~5 llamadas secuenciales (mediana, sección
1.2). En el kit, "casos a la vez" = tamaño del lote. Con un solo proceso y un lote fijo no hay cola
ni atención continua: los casos entran y salen por tandas.

| Dato medido (T4 x2) | Lote 1 | Lote 4 |
|---|---|---|
| Rendimiento de un proceso | 108 casos/h (3600 / 33,3) | 230 casos/h (3600 / 15,6) |
| Latencia estimada de un caso (5 llamadas x s por micro-lote) | ~34 s (5 x 6,8) | ~60 s (5 x 12,0) |
| VRAM pico (suma de 2 T4) | 16,7 GB | 19,1 GB |

Derivados de la sección 2 (cálculo propio). La latencia es una estimación con la mediana de 5
llamadas, sin cola de espera ni búsquedas simultáneas.

**Lectura.** El lote sube el rendimiento (2,1 veces de 1 a 4) pero empeora la latencia por
llamada. Para tráfico interactivo conviene lote bajo; para procesar muchos casos, lote mayor.

**Cuántos tolera la A100-40GB (a medir).**
- En T4 la VRAM pico crece de 16,7 a 22,7 GB al pasar de lote 1 a 16 (puntos medidos, sección 2,
  sin extrapolar más allá). 22,7 GB es menos que los 40 GB de la A100, lo que *sugiere* margen para
  lotes grandes, pero las dos GPU no son comparables (cada T4 tiene 15 GB y el modelo se reparte;
  la A100 tiene una sola GPU de 40 GB). El fallo CUDA de 8 y 16 desaparece con `CUDA_LAUNCH_BLOCKING=1`
  (`corridas/kaggle/diag-l8`), lo que apunta a concurrencia del modelo repartido en 2 GPU y no a memoria:
  en la A100 ese escenario no existe, pero hay que medirlo.
- Desde 2026-11-05 el modelo se carga **sin las torres de visión y audio**: con el mismo lote 4, la VRAM pico
  bajó de 19,1 a 18,1 GB con trazas idénticas en los 50 casos (`corridas/kaggle/solo-texto-l4`). Las cifras
  anteriores de esta sección incluyen las torres.
- No medimos KV cache por conversación ni el techo de lote antes de quedarse sin memoria.
- **PENDIENTE (A100-40GB):** repetir el barrido 1/4/8/16/32 con `MODELO_METRICAS`, registrar el
  máximo lote estable, la VRAM reservada y el s/caso; recién ahí se fija "casos simultáneos".

**Cómo escalaría si sube la demanda** (diseño, no implementado ni validado):
1. **Réplicas:** más contenedores Modal con una A100-40GB cada uno, con el modelo y el adapter
   cargados una vez por contenedor. El costo crece lineal con la cantidad de réplicas activas.
   Cuidar el arranque en frío (34 a 146 s de carga en T4): contenedores tibios si hay picos.
2. **Servidor con batching continuo (vLLM):** admite casos que entran y salen del lote en cada
   paso, sin esperar la tanda. Es la mejora más grande esperable en rendimiento, **no probada**:
   queda como trabajo futuro (verificar soporte de Gemma 4 y de adapters LoRA).
3. **Menos tokens por caso:** recortar prompt y contexto baja cómputo y VRAM por caso (sección 7).

**Consumo proyectado.** Con la fórmula de 4.2: `costo mensual = casos por mes x costo por caso`.
Sin s/caso en A100 no se publica un número. **PENDIENTE.**

---

## 6. Qué falta (todo lo que depende del modelo final o de A100)

| Pendiente | Depende de | Issue |
|---|---|---|
| Tiempo de entrenamiento real y su costo | dataset final, prompt final, A100/Modal | #3, #5 |
| VRAM de entrenamiento con secuencias reales | corrida real (ya se registra en `estado.json` y `runner.json`) | #3, #5 |
| s/caso, carga y VRAM de inferencia en A100-40GB | Modal habilitado | #4, #5 |
| Lote máximo estable en A100-40GB | idem | #4 |
| Antes/después con el modelo final (resueltos, s/caso, VRAM, costo) | modelo entrenado y comparación con los mismos 50 casos | #5 |
| Causa del fallo CUDA en lotes 8 y 16 | acotada: desaparece con `CUDA_LAUNCH_BLOCKING=1` (concurrencia en 2 GPU, hipótesis); sin arreglo útil en T4 | #4 |
| Costo total de desarrollo y de mantenimiento | todo lo anterior | #5 |

**Antes del resultado final (ya medido, T4):** secciones 1 y 2. **Después:** repetir las mismas
tablas con el adapter entrenado, mismo comando, mismos 50 casos y normas original, b y c.

---

## 7. Trabajos futuros

| Idea | Beneficio esperado | Evidencia/estado |
|---|---|---|
| Corregir el fallo de lotes >= 8 | más casos a la vez | CUDA `unspecified launch failure`; probar `CUDA_LAUNCH_BLOCKING=1` |
| vLLM u otro servidor con batching continuo | más rendimiento y mejor latencia | no probado; chequear Gemma 4 y LoRA |
| Medir en A100-40GB | todos los números finales | bloqueado por facturación de Modal |
| Reducir el mensaje de sistema y el contexto de búsquedas | menos cómputo y VRAM por llamada y por entrenamiento | sistema = 2.232 caracteres (~35 % de la conversación mediana); búsqueda: la consigna dice que es más de la mitad en la referencia |
| Repetir cada medición varias veces | intervalos en vez de una sola corrida | hoy una sola por configuración |

---

## 8. De dónde sale cada número

Artefactos locales en `corridas/kaggle/` (no versionados, regenerables solo volviendo a correr en
Kaggle; la bitácora [progreso.md](progreso.md) los resume con commit y fecha).

| Número | Artefacto o cálculo |
|---|---|
| s/caso, `construir_agente` | `<corrida>/salida/tiempos.json` |
| Resueltos, parcial, llamadas, búsquedas, caracteres de contexto, `sin_decision` | `<corrida>/puntuacion.json` o `puntuacion/resumen.json`, y `puntuacion/resultados.jsonl` |
| Tokens, tokens/s, micro-lotes, VRAM pico y reservada (por GPU y suma) | `<corrida>/metricas.jsonl` (máximo por GPU, suma de los dos máximos; tokens/s = tokens generados / segundos) |
| Commit, GPU, dtype, lote, inicio y fin | `<corrida>/runner.json` |
| Pasos, pérdida, `train_runtime` | `humo-entrena-*/runner.log` y `entrenamiento/checkpoint-*/trainer_state.json` |
| Configuración de entrenamiento y LoRA | `humo-entrena-1-v6/entrenamiento/corrida.json` |
| Tarifa Modal y crédito mensual | `mission.md` (línea de cómputo) |
| Cuota Kaggle | [kaggle.md](kaggle.md) |

Limitaciones comunes: T4 fp16 (no comparable con A100), una corrida por configuración, el lote 4
viene de otra sesión, y las corridas de lote 8 y 16 están incompletas.
