# Guion del Loom (15–20 minutos)

El orden y el contenido son los que pide la consigna ([mission.md, «Presentación en Loom»](../mission.md)): primero
el resultado y después cómo llegamos. Cada bloque dice **quién lo graba, cuánto dura, qué se ve en pantalla y qué se
dice**. Todos los números salen del [informe](../INFORME.md); no digan ninguno que no esté ahí.

| # | Bloque | Quién (propuesta) | Duración | Acumulado |
|---|---|---|---:|---:|
| 1 | El modelo final | Fran | 2:30 | 2:30 |
| 2 | Demo | Fran | 4:00 | 6:30 |
| 3 | El historial | Yas | 3:00 | 9:30 |
| 4 | Buscador y prompt | Eze (buscador) y Martu (prompt) | 3:30 | 13:00 |
| 5 | Entrenamiento | Martu y Fran | 3:30 | 16:30 |
| 6 | Recursos y producción | Fran | 2:30 | 19:00 |

El reparto sigue las partes de cada uno y se puede cambiar. Hay ~1 minuto de margen hasta los 20: si un bloque se
pasa, recorten del 3 o del 4, nunca de la demo.

## Antes de grabar

- **Cada uno graba su bloque por separado** con pantalla y cámara, y después se unen en orden en un solo video de
  Loom. Arranquen cada bloque diciendo su nombre y el bloque («Soy Yas, les cuento el historial»).
- **Pantalla:** el informe abierto en GitHub (renderizado), con zoom para que las tablas se lean. Tengan abiertas de
  antemano las pestañas que usa su bloque.
- **Números:** léanlos de la tabla en pantalla, no de memoria. Si alguno no coincide con el informe, el que vale es
  el informe.
- **Lo que no hay que decir:** que algo corrió en A100 (nada corrió ahí) o que elegimos algo mirando práctica (todo se
  eligió con el dev propio de 40 casos; práctica es solo para reportar).

## 1. El modelo final (2:30) — Fran

**Pantalla:** [Resumen del informe](../INFORME.md#resumen), la tabla de resultados.

**Qué decir:**

- La entrega es Gemma 4 E4B con un adapter LoRA (`FranCalveyra/tp-admision`, revisión `770c9fac`), el prompt
  `procedimiento-v1`, el buscador BM25 `baseline` y lote 4.
- En los 50 casos de práctica resuelve **36, 28 y 36** con las normas original, b y c: **100 de 150**. La referencia de
  la cátedra con fine-tuning supervisado y datos perfectos resuelve 25, 24 y 25 (74), y sin entrenar 12, 15 y 12 (39).
  Nuestro modelo sin entrenar, con el prompt de la cátedra, arrancaba en 10, 10 y 7 (27).
- **Costo por caso:** 27,8–28,8 segundos por caso y 19,94 GB de VRAM pico en Kaggle con 2 T4, en fp16 y lote 4.
- **Cómo se eligió:** prompt y adapter se eligieron solo con un conjunto de desarrollo propio de 40 casos (el adapter
  final dio 31/40 contra 26/40 del anterior). La práctica se corrió después y no cambió ninguna decisión.
- **Limitación principal:** no tuvimos acceso a una A100 de 40 GB. Todo se midió en T4.

## 2. Demo (4:00) — Fran

La consigna pide el agente corriendo **con normas b o c**, con dos casos de práctica: uno que activa un circuito
tiempo-dependiente y uno que se decide por una costumbre. Usamos **normas c** y estos dos casos, que la entrega
resuelve completos (9/9) en la corrida versionada v40:

| Caso | Paciente | Qué muestra |
|---|---|---|
| `pev-017` | Patricia Gauna, 76 años, «se le torció la boca y perdió fuerza en la pierna», viene **con** su esposa | Activa el **Código ACV**. El plazo sale de la norma: 10 min en original, 20 en b, **15 en c**. Tiene 76 pero está acompañada: **no** corresponde aviso al voluntariado |
| `pev-042` | María Inés Quiroga, 85 años, herida cortante en el trayecto al trabajo, viene **sola** | Se decide por la **costumbre**: 85 años y sin acompañante, así que **avisa al voluntariado**, algo que ninguna norma escrita dice. Además cubre la ART por el accidente in itinere |

Los dos juntos muestran la costumbre completa: no alcanza con tener 75 o más, además hay que estar sin acompañante.

### Cómo correrlo (necesita GPU: Kaggle, sesión con 2× T4)

En una terminal de un notebook de Kaggle con el repo clonado, las dependencias de `requirements-inferencia.txt` y el
secreto `HF_TOKEN` con acceso al repo privado del adapter:

```bash
mkdir -p corridas/demo && sed -n '17p;42p' practica/casos.jsonl > corridas/demo/casos.jsonl   # pev-017 y pev-042
python -m admision.correr practica --entrega entrega.py --casos corridas/demo/casos.jsonl \
    --esperadas practica/esperadas-c.jsonl --normas normas/c --salida corridas/demo/c
python -m scripts.ver_caso corridas/demo/c --caso pev-017
python -m scripts.ver_caso corridas/demo/c --caso pev-042
```

Cargar el modelo tarda entre 20 s y 2,5 min, y cada caso unos 30 s. Graben la corrida y corten la espera.

**Plan B, sin GPU:** mostrar las trazas versionadas de v40, que son exactamente lo que hizo la entrega:

```bash
mkdir -p corridas/demo-v40 && cp evaluaciones/practica/v40-entrenado-v39/ambos-c/salida/trazas.jsonl corridas/demo-v40/
uv run python -m admision.correr puntuar --trazas corridas/demo-v40/trazas.jsonl \
    --esperadas practica/esperadas-c.jsonl --salida corridas/demo-v40
uv run python -m scripts.ver_caso corridas/demo-v40 --caso pev-017
```

Si usan el plan B, díganlo en el video: «esto es la traza guardada de la corrida v40, no una corrida en vivo».

### Qué señalar en pantalla (pev-017)

1. `leer_triage`: 76 años, signos neurológicos de inicio súbito, acompañante «su esposa Mariana», sin documento.
2. `buscar_normativa` trae NOR-ADM-02 (mayores van al box antes del registro, **P-ADM-2 = 72 años**) y NOR-TDE-02
   (Código ACV, **P-TDE-2 = 15 min**). **Este es el punto clave:** el número lo trae el buscador desde las normas c,
   el modelo no lo memorizó. Con las normas b el mismo caso da 20.
3. `preguntar` por la cobertura, porque no presentó credencial.
4. La ficha registrada y los **9 controles en 9/9**.

### Qué señalar en pantalla (pev-042)

1. `leer_triage`: 85 años, acompañante: ninguno, lesión en el trayecto al trabajo.
2. En la decisión, el modelo escribe primero `Edad: 85 años. Acompañante: ninguno.` y después la ficha con
   `aviso_al_voluntariado: true`. Ese prefijo es lo que agregamos en el último entrenamiento (v39) para que decida
   la costumbre mirando los dos datos.
3. Los 9 controles en 9/9.

## 3. El historial (3:00) — Yas

**Pantalla:** [Parte 1](../INFORME.md#parte-1-curado), y [costumbres.md](../costumbres.md) para las filas de ejemplo.

**Qué decir:**

- **Cómo separamos errores de ruido:** primero se calcula, fila por fila, si la ficha viola alguna norma escrita
  (registro, documento, box, circuito, plazo, cobertura, verificación, entidad). Eso es `error`, y tiene prioridad.
  Las filas limpias que se apartan de una costumbre son `ruido`.
- **Resultado:** 1.600 correctas, 240 errores y 160 ruido, en `historial_clasificado.csv`.
- **Costumbre 1:** se avisa al voluntariado **si y solo si** el paciente tiene 75 años o más **y** está sin
  acompañante. Ninguna norma menciona el voluntariado. Se encontró probando todas las reglas de edad, solas o
  combinadas con un dato del triage: la regla completa explica 1.609 de 1.760 filas limpias; solo por edad, 1.507.
- **Costumbre 2:** si el paciente tiene obra social y prepaga, se verifica **solo la prepaga** y no se recurre a la
  otra credencial. Pasa en 163 de 172 filas limpias con las dos.
- **Quién se aparta:** todos los desvíos son de tres operadores (OP-04, OP-11 y OP-17), en las dos direcciones y sin
  otra regla: es variación individual, por eso es `ruido`. Los otros 17 operadores cumplen el 100 %.
- **Filas de ejemplo** (abrir `historial/historial.csv` filtrado por id):
  - **H-0013:** OP-09, paciente de 80 años sin acompañante, avisó al voluntariado. Cumple la costumbre, así que es
    `correcta`.
  - **H-0005:** OP-04, paciente de 35 años, avisó al voluntariado. Ninguna norma lo prohíbe, pero se aparta de la
    costumbre, así que es `ruido`.
- **Conversaciones de entrenamiento:** 2.900 en total. Son 1.600 del historial (solo las correctas), 800 de la
  referencia de la cátedra y 500 sintéticas del generador con valores de normas variados. Las búsquedas se volvieron
  a correr con nuestro buscador, sobre normas con los valores de cada conversación. Después sumamos sets dirigidos
  al aviso (v4 a v6) para el último entrenamiento.

## 4. Buscador y prompt (3:30) — Eze y Martu

### Buscador (1:45) — Eze

**Pantalla:** [Parte 2](../INFORME.md#parte-2-buscador), la tabla de `evaluar_rag`.

- La referencia de la cátedra (BM25, 2 secciones) trae 51/60 consultas completas, recall 0,875, precisión 0,492 y
  580 tokens por consulta.
- Probamos 68 configuraciones. La mejor en la evaluación aparte fue `prefijos`: 55/60, recall 0,933, precisión
  0,686 y 549 tokens, mejor que la referencia en las cuatro métricas.
- **Pero con el agente no ayudó:** sin entrenar, `baseline` resuelve 10/50 y `prefijos` 6/50. Por eso la entrega usa
  `baseline`. Es el resultado negativo más importante del buscador: la consigna dice que lo que cuenta es cuánto le
  sirve al agente, no la evaluación aparte.
- **Contexto:** cada búsqueda trae ~618 tokens (2 secciones completas). Con la entrega, el agente hace ~3,2 búsquedas
  por caso.

### Prompt (1:45) — Martu

**Pantalla:** [Parte 3](../INFORME.md#parte-3-prompt), la tabla de candidatos.

- Con el modelo sin entrenar y un dev propio de 40 casos (nunca práctica ni test), probamos cinco mensajes de
  sistema.
- Ganó `procedimiento-v1`: le dice al modelo que busque cada tema por separado y verifique antes de decidir.
  Pasó de 8/40 a 9/40 en dev y de 27 a 42 sobre 150 en práctica.
- **El costo:** el prompt pasó de 615 a 857 tokens (+39 %) y el contexto por caso creció ~2,6 veces, porque el
  modelo busca más.
- **Lo que no funcionó:** acortar el prompt empeoró mucho (1/40 y 2/40), porque las descripciones de las
  herramientas son las que hacen que el modelo busque. Cambiar el formato de los mensajes (v37) dio 10/40, dentro
  del ruido, y no se adoptó.

## 5. Entrenamiento (3:30) — Martu y Fran

**Pantalla:** [Parte 4](../INFORME.md#parte-4-entrenamiento): la tabla de cuatro filas y la tabla de pasadas.

- **Estrategia:** fine-tuning supervisado con LoRA sobre las conversaciones curadas, entrenando solo las respuestas
  del asistente. Se entrenó con el prompt optimizado.
- **Las cuatro filas** (original / b / c):

  | Fila | Resultado |
  |---|---|
  | Sin entrenar | 10 / 10 / 7 |
  | Solo prompt | 14 / 16 / 12 |
  | Solo entrenamiento | 17 / 14 / 20 |
  | Las dos juntas (la entrega) | **36 / 28 / 36** |

  El entrenamiento y el prompt se potencian: juntos resuelven bastante más que la suma de lo que agrega cada uno.
- **Corridas que no funcionaron y por qué:**
  - v28: sin memoria en el primer paso (la pérdida calculaba todo el vocabulario en fp32). Se calculó solo donde se
    supervisa.
  - v29 y v31: error de dispositivo con 2 GPU. v30: timeout de W&B.
  - El primer adapter completo (v32 + v33) dio 26/40 en dev, pero **nunca avisaba al voluntariado** (0/8).
  - v36: más ejemplos de aviso. Llegó a 6/8, pero con 12 falsos positivos: 23/40, no adoptado.
  - v38: más peso al aviso en la pérdida. Casi no se movió: 26/40, no adoptado.
  - **v39:** el modelo escribe edad y acompañante antes de decidir. 5/8 positivos y 31/40: **adoptado**.
- **Lo que no llegamos a hacer:** barrido de hiperparámetros y DPO. Quedan como trabajo futuro.
- Todas las corridas, también las fallidas, están en W&B, con su checkpoint en Hugging Face.

## 6. Recursos y producción (2:30) — Fran

**Pantalla:** [Parte 5](../INFORME.md#parte-5-recursos) y [Trabajos futuros](../INFORME.md#trabajos-futuros).

- **Entrenamiento:** el último (v39) tardó 11.735 s (~3,3 h) con 20,36 GB de VRAM pico en 2 T4. Todo el desarrollo
  se hizo con la GPU gratuita de Kaggle: 0 USD.
- **Inferencia:** 28,8 s por caso con lote 4: ~125–129 casos por hora por proceso en T4. Con lote 8 o 16 la GPU
  falla (`CUDA error`). Lote 1 es 2,1 veces más lento.
- **Costo proyectado:** como orden de magnitud, con los tiempos de T4 y el precio de una A100 (2,10 USD/h), ~0,017 USD
  por caso. No está medido en A100.
- **Casos simultáneos en producción:** no lo pudimos medir sin A100; es la limitación declarada.
- **Cómo escalaría (diseño, no implementado):** réplicas con una A100 de 40 GB por contenedor y el modelo cargado
  una sola vez, con contenedores tibios porque el arranque en frío tarda entre 20 s y 2,5 min. La mayor ganancia
  esperable es un servidor con batching continuo (vLLM) en vez de lotes fijos, pero no se probó.
- **Trabajos futuros:** medir todo en A100, DPO con pares que difieran solo en el aviso, barrido de hiperparámetros,
  arreglar la falla con lotes de 8 o más, achicar el prompt y agregar sinónimos al buscador.

**Cierre (una frase):** «Pasamos de 27 a 100 casos de 150, con todo auditable en el repo: cada número tiene su
comando, su checkpoint y sus conversaciones guardadas».
