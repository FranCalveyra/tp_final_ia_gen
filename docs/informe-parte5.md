# INFORME, Parte 5: recursos y proyección a producción (BORRADOR)

> **Estado: borrador del issue #5. No es el informe final.** Todo lo medido hasta hoy (2026-11-05)
> salió de **Kaggle 2x T4 con fp16**, con Gemma 4 E4B **sin entrenar** (o con datos de humo
> sintéticos). **Nada se midió en A100-40GB ni con el modelo final.** Esos números figuran como
> **PENDIENTE**. Los números de T4 no se pueden transferir a A100 y acá no inventamos factores.
> Para pegar en `INFORME.md` cuando el equipo lo cree; la bitácora vive en [progreso.md](progreso.md).

## Resumen (lo que ya se puede afirmar)

| Pregunta de la consigna | Respuesta hoy | Estado |
|---|---|---|
| Tiempo por caso (inferencia) | 33,3 s (lote 1) y 15,6 s (lote 4) en T4 | validado en GPU (T4) |
| VRAM de inferencia | 16,7 GB (lote 1), 19,1 GB (lote 4 con torres), 18,1 GB (lote 4 solo texto), suma de 2 T4 | validado en GPU (T4) |
| Casos a la vez al evaluar | 1/4 válidos; 8/16 fallan sin bloqueo; 8 válido con `CUDA_LAUNCH_BLOCKING=1`, más lento que 4 | T4; causa definitiva pendiente |
| Casos a la vez al entrenar | 1 conversación por paso (`batch_size=1`, acumulación 1) en el humo | validado en GPU (T4), solo humo |
| Tiempo y VRAM de entrenamiento | v6: 17,4 s; humo solo texto: 15,675 GB pico, 15,818 GB reservada | humo en T4; datos reales: PENDIENTE |
| Costo | Kaggle: 0 USD. A100: fórmula, sin medición | PENDIENTE (A100) |
| Simultáneos en producción | proyección cualitativa, ver sección 5 | hipótesis, A100: PENDIENTE |
| Antes/después con el modelo final | no existe aún | PENDIENTE |

**Leyenda de validación.** *implementado*: existe el código. *probado con dobles*: tests sin GPU.
*validado en GPU (T4)*: corrió en Kaggle. *validado en A100*: **ninguno todavía**.

**Cómo leer este documento:** secciones 1 a 4 son medición (T4); 5 es proyección (hipótesis);
6 lista lo pendiente; 7 son trabajos futuros; 8 es el origen de cada número.

---

## 1. Medición de inferencia (T4 x2, Gemma 4 E4B sin entrenar)

Condiciones comunes: fp16, `device_map=auto` en 2 T4 que el runner reportó como **15.360 MiB cada una**, torch
2.15.0+cu132, 50 casos de práctica, tope 512 tokens nuevos por llamada y 8192 de contexto
(`runner.json`). En esas sesiones, capacidad reportada: 15 GiB = 16,106 GB decimales por GPU,
32,212 GB entre ambas. `_vram` divide bytes por 10⁹: sus GB son decimales, no GiB.
Puntuado con `admision.correr puntuar`. **Una sola corrida por configuración.**

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
  3 búsquedas. `segundos` de `trazas.jsonl` es tiempo hasta la última llamada de ese caso desde el inicio
  común de `jugar_lote`, no tiempo exclusivo de cómputo. Sirve como finalización bajo la carga de 50 casos
  (sección 5), no como costo individual ni latencia de un paciente aislado.
- No hay un recorte de contexto silencioso: 0 llamadas inválidas en las cuatro corridas completas
  (`invalidas` en `resultados.jsonl`).

### 1.3 Tiempo de carga

`construir_agente` varía según corrida y caché: 145,6 s en `base-original-l4`, 156,0 s en
`lote-original-l1` y 125,7 s en `solo-texto-l4`; las variantes b/c previas registraron 47,2/34,1 s
(`salida/tiempos.json`). No separa descarga de carga. Es un costo fijo por proceso: pesa en
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
VRAM = máximo por GPU sobre los micro-lotes, sumado (cálculo propio). Es **cota superior**, porque los
picos por GPU pueden no coincidir. Reservada incluye memoria asignada más caché: **no sumar ambas columnas**
ni confundirlas con memoria física disponible. Los s/caso de 8/16 originales son de corridas cortadas: no sirven para comparar.

**Lo que se concluye con evidencia**
- Lote 1 a 4: 2,1 veces más rápido (33,3 a 15,6 s/caso) por 2,4 GB más de VRAM pico. Sin
  regresiones: los mismos 10 casos resueltos, primera respuesta idéntica en 50 de 50 y conversación
  completa idéntica en 49 de 50 (bitácora, 2026-11-04).
- Cada micro-lote tarda 6,8 s con 1 conversación y 12,0 s con ~3,8: el cómputo por conversación
  baja (6,8 a 3,1 s), pero la **latencia de cada llamada casi se duplica**. Importa en producción.
- Lote 8 y 16: `AcceleratorError: CUDA error: unspecified launch failure` dentro de
  `project_per_layer_inputs` (`salida/tiempos.json`, campo `error_detalle`). El lote 8 dejó 34 casos
  sin decisión y el 16, los 50 (`puntuacion/resumen.json`). Esos artefactos históricos incluso dicen
  `runner.returncode=0`: prevalece el error del kit; el runner fue corregido después.
  **Diagnóstico posterior solo para lote 8:** `diag-l8`, con `CUDA_LAUNCH_BLOCKING=1`, completó 50 casos,
  10/50 resueltos y 0 sin decisión, 17,366 s/caso (más lento que lote 4: 15,642). La causa definitiva no se
  identificó; concurrencia/asincronía entre GPU es hipótesis, no diagnóstico probado. No se repitió lote 16 con bloqueo.
- Observación, **no diagnóstico**: la VRAM *reservada* (caché de PyTorch) llegó a 30,3 GB en el lote 8
  por debajo de 32,212 GB de capacidad reportada entre esas dos GPU (15.360 MiB cada una), no por encima.
  El pico asignado fue 11,5+8,7 GB; la suma reservada no prueba el estado instantáneo ni memoria libre de
  cada GPU. No hay diagnóstico de OOM ni de ausencia de presión de memoria a partir de esos agregados.

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
| v7 (`humo-entrena-2-v7`) | reanuda desde `checkpoint-2` bajado de HF | 177 s | 15,8 s (pasos 3 a 8) | pasos 3 a 8 cercanos a v6, no idénticos |

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
  `pasos_aprox = conversaciones x épocas / (batch_size x acumulación)` (redondeos/último lote y
  acumulación por época los resuelve Trainer); `tiempo_runner ≈ tiempo_train + carga + verificaciones + subidas`.
  `train_runtime` y tiempo de runner no son intercambiables, y ninguno incluye necesariamente la sesión completa. El 2,18 s/paso del humo **no** sirve para extrapolar: eran
  secuencias cortas y sintéticas.
- Los dos runners sumaron 340 s de reloj de pared (163+177), no “GPU-segundos” ni cuota medida.
  Incluyen carga/verificaciones/subidas; instalación/arranque externo no está incluido. Los intentos fallidos v4 y v5 (157 y 156 s,
  `humo-entrena-1/runner.json` y `humo-entrena-1-v5/runner.json`) también consumen cuota.

---

## 4. Costo de punta a punta

### 4.1 Lo medido: Kaggle (desarrollo)

Kaggle es gratis; su cuota ronda las 30 h/semana de GPU, aproximada y fijada por Kaggle
([kaggle.md](kaggle.md)). Uso, derivado de `runner.json` (inicio a fin del runner, sin
instalación del entorno):

| Trabajo | Segundos de runner (reloj de pared) | Fuente |
|---|---|---|
| Línea de base original, b y c (lote 4) | 930 + 840 + 841 = 2.611 s (0,73 h) | `base-*-l4/runner.json` |
| Barrido de lotes 1, 8 y 16 | 1.822 + 434 + 193 = 2.449 s (0,68 h) | `lote-original-l{1,8,16}/runner.json` |
| Humo de entrenamiento (v6 + v7) | 163 + 177 = 340 s (0,09 h) | `humo-entrena-{1-v6,2-v7}/runner.json` |

La bitácora hizo una estimación histórica de ~2.690 s incluyendo arranque para el barrido; **no es cuota
facturada verificada**. No multiplicamos tiempos por número de GPU ni inferimos cuota desde estas duraciones.
Hay otras corridas (`t4-original-09`, intentos fallidos anteriores) sin `runner.json` completo:
**el total real de cuota usado es mayor y no está auditado**. Costo en dinero: **0 USD**.

### 4.2 Escenario histórico ilustrativo: Modal A100-40GB (no operativo)

`mission.md`: A100-40GB ~2,10 USD/h y 30 USD de crédito por integrante y por mes, o sea unas
14,3 h por persona (30 / 2,10). Son cifras de la consigna, no cotización actual.
**Modal fue descartado definitivamente el 2026-11-05**; no corrió entrenamiento GPU allí.
Estas fórmulas se conservan como escenario histórico/ilustrativo, no un plan ni presupuesto vigente.
La vía A100-40GB futura, su tarifa y sus permisos están por determinar.

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

Cargar el modelo (34 a 156 s en T4) equivale a 0,02 a 0,09 USD por arranque a 2,10 USD/h.
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
| Finalización desde inicio común, mín / mediana / máx (50 casos) | 743,030 /1.446,819 /1.663,648 s | 302,221 /670,895 /782,102 s |
| VRAM pico (suma de 2 T4) | 16,7 GB | 19,1 GB |

Rendimiento derivado de sección 2; finalización calculada sobre `salida/trazas.jsonl` de lote 1/4,
campo `segundos`. `admision/correr.py` actualiza ese reloj tras cada herramienta desde el inicio común.
Los 50 casos entran juntos; `MODELO_LOTE=4` controla micro-lotes de inferencia, **no significa solo 4 casos
en vuelo**. La espera entre micro-lotes sí afecta ese tiempo de finalización. La antigua aproximación
“5 llamadas × duración micro-lote” la omitía y se retiró. No son latencias de servicio interactivo/SLO medidos.

**Lectura.** Cada llamada por micro-lote tarda más con 4, pero el lote de 50 termina antes; no extrapolar
que eso empeora la latencia de cada caso ni recomendar un lote para tráfico interactivo sin medirlo.

**Cuántos tolera la A100-40GB (a medir).**
- En T4 la VRAM pico crece de 16,7 a 22,7 GB al pasar de lote 1 a 16 (puntos medidos, sección 2,
  sin extrapolar más allá). 22,7 GB es menos que los 40 GB de la A100, lo que *sugiere* margen para
  lotes grandes, pero las dos GPU no son comparables (las corridas reportan 15.360 MiB por T4 y el modelo se reparte;
  la A100 tiene una sola GPU de 40 GB). En **lote 8** no reapareció el fallo con `CUDA_LAUNCH_BLOCKING=1`
  (`corridas/kaggle/diag-l8`); no se probó lote 16 con ese flag. Sugiere concurrencia/asincronía del modelo
  repartido en 2 GPU, pero no identifica la causa ni descarta todas las formas de presión de memoria.
  En A100 cambia la topología: sigue siendo necesario medirlo.
- Desde 2026-11-05 el modelo se carga **sin las torres de visión y audio**: con el mismo lote 4, la VRAM pico
  bajó de 19,1 a 18,1 GB con trazas idénticas en los 50 casos (`corridas/kaggle/solo-texto-l4`). Las cifras
  anteriores de esta sección incluyen las torres.
- No medimos KV cache por conversación ni el techo de lote antes de quedarse sin memoria.
- **PENDIENTE (A100-40GB):** repetir el barrido 1/4/8/16/32 con `MODELO_METRICAS`, registrar el
  máximo lote estable, la VRAM reservada y el s/caso; recién ahí se fija "casos simultáneos".

**Cómo escalaría si sube la demanda** (diseño, no implementado ni validado):
1. **Réplicas:** más procesos/contenedores con una A100-40GB cada uno en la plataforma futura, con el modelo y el adapter
   cargados una vez por contenedor. El costo crece lineal con la cantidad de réplicas activas.
   Cuidar el arranque en frío (34 a 156 s de carga en T4): contenedores tibios si hay picos.
2. **Servidor con batching continuo (vLLM):** admite casos que entran y salen del lote en cada
   paso, sin esperar la tanda. Es la mejora más grande esperable en rendimiento, **no probada**:
   queda como trabajo futuro (verificar soporte de Gemma 4 y de adapters LoRA).
3. **Menos tokens por caso:** recortar prompt y contexto baja cómputo y VRAM por caso (sección 7).

**Modelo de capacidad/costo (variables, no predicción):** `Q=3600×N/T_lote` casos/h medidos en una instancia,
con `N` casos y `T_lote` segundos sin carga inicial. Para demanda `λ` casos/h y utilización objetivo `0<ρ<1`,
`réplicas=ceil(λ/(ρ×Q))`: aproximación de capacidad, no garantía de latencia. `ρ` y el SLO siguen sin fijar.
Con `r` réplicas, `H` horas provisionadas por réplica/mes, tarifa `p` USD/h **por instancia del hardware elegido**,
volumen `V` casos/mes y costos adicionales `C_otros`: `C_mes=r×H×p+C_otros`, `C_caso=C_mes/V` (`V>0`).
Incluye horas ociosas; no equivale a cobrar solo segundos de generación. Para un lote aislado se puede usar
`C_caso≈p×(T_carga+T_lote)/(3600×N)` más costos adicionales asignados. Plataforma, tarifa, demanda,
utilización, latencia y costos adicionales futuros no están medidos: **no se publica presupuesto de producción**.

---

## 6. Qué falta (todo lo que depende del modelo final o de A100)

| Pendiente | Depende de | Issue |
|---|---|---|
| Tiempo de entrenamiento real y su costo | dataset final, prompt final, vía A100-40GB a determinar | #3, #5 |
| VRAM de entrenamiento con secuencias reales | corrida real (ya se registra en `estado.json` y `runner.json`) | #3, #5 |
| s/caso, carga y VRAM de inferencia en A100-40GB | vía A100-40GB disponible y autorizada | #4, #5 |
| Lote máximo estable en A100-40GB | idem | #4 |
| Antes/después con el modelo final (resueltos, s/caso, VRAM, costo) | modelo entrenado y comparación con los mismos 50 casos | #5 |
| Causa del fallo CUDA en lotes 8 y 16 | lote 8 completa con bloqueo, más lento que 4; lote 16 con bloqueo no probado; causa definitiva pendiente | #4 |
| Costo total de desarrollo y de mantenimiento | todo lo anterior | #5 |

**Antes del resultado final (ya medido, T4):** secciones 1 y 2. **Después:** repetir las mismas
tablas con el adapter entrenado, mismo comando, mismos 50 casos y normas original, b y c.

---

## 7. Trabajos futuros

| Idea | Beneficio esperado | Evidencia/estado |
|---|---|---|
| Corregir el fallo de lotes >= 8 | más casos a la vez | bloqueo probado en 8 sin fallo, pero más lento que 4; causa pendiente, no adoptado |
| vLLM u otro servidor con batching continuo | más rendimiento y mejor latencia | no probado; chequear Gemma 4 y LoRA |
| Medir en A100-40GB | todos los números finales | vía alternativa a determinar; Modal descartado |
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

## 9. Integración de criterios #5

| Criterio del issue | Qué está cubierto | Qué impide marcarlo completo |
|---|---|---|
| Tiempo/VRAM/costo de entrenamiento | Humo T4 y fronteras trainer/runner/sesión; fuentes auditadas | Corrida real y A100-40GB, antes/después final; costo total no auditado |
| Casos a la vez entrenando/evaluando | Humo batch 1/acumulación 1 y barrido inferencia T4 | Configuración final y validación A100; no extrapolar paralelismo desde acumulación |
| Producción/consumo/escalado | Fórmulas con variables y alternativas explícitas | SLO/demanda/tarifa/capacidad y recursos finales sin medir |
| Trabajos futuros | **Redacción cubierta** en sección 7, con motivos/evidencia | Integrar al `INFORME.md` compartido; no implica mejoras ya ejecutadas |

#5 continúa abierto. La deuda de hardware es [KI-A100-001](../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb).
