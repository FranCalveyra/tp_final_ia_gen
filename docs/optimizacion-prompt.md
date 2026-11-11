# Optimización de prompt

## Objetivo

El issue #13 implementa la Parte 3 de la misión: optimizar el mensaje de sistema y el
formato de los mensajes con Gemma 4 sin entrenar, antes del SFT. La comparación considera
los nueve controles y el largo del prompt, y conserva evidencia reproducible de cada candidato.

## Estado (2026-11-11)

- **Elegido:** B `procedimiento-v1` en el conjunto de desarrollo `datos/prompt-dev/v1`
  (9/40 contra 8/40 del baseline; 857 contra 615 tokens). Práctica solo se usó para reportar:
  14/16/12 contra 10/10/7. Detalle en [Resultados](#resultados-kaggle-v25v27) y [Decisión](#decisión).
- **Método:** búsqueda manual entre cinco candidatos, no DSPy/GEPA (la misión lo sugiere,
  no lo exige). Los candidatos se diseñaron mirando el perfil agregado de fallas de práctica
  v22; la selección se hizo solo en el conjunto de desarrollo (ver [Decisiones y límites](#decisiones-y-límites)).
- **Pendiente:** solo se optimizó el mensaje de sistema. El formato de los mensajes
  (`mensaje_inicial`, `formato_observacion`, `mensaje_json_invalido`) sigue igual al baseline
  en los cinco candidatos; optimizarlo requiere otra corrida en GPU.
- **Abierto:** `entrega.py` sigue usando el baseline. Pasarlo a B y elegir con qué conversaciones
  entrena #15 (`v2-baseline` o `v3-procedimiento`) se decide en equipo.

Desde el issue #7 (2026-11-10), `entrega.py` usa `AgentePrompt` con el `PromptSpec` de
`PROMPT_SPEC` o el baseline (ver [contrato de formato](#contrato-de-formato-7--13)). Las secciones
siguientes describen la infraestructura en el orden en que se construyó.

**Validado con Gemma real (2026-11-10, Kaggle v23):** `AgentePrompt` con el prompt `baseline`
reproduce exactamente al `AgenteJSON` del kit (50/50 conversaciones idénticas, 10/50 en
práctica original) y el runner corre de punta a punta con la config T4, `HF_TOKEN` desde
Kaggle Secrets y `--solo-plomeria`. Largo medido con el tokenizer real: system prompt 615
tokens, contexto inicial 650 (punto de partida de la optimización). Detalle en
[progreso](progreso.md).

## Problema del agente de referencia

`admision.ejemplo.AgenteJSON` construye internamente el mensaje de sistema, el mensaje
inicial, las observaciones y la corrección de JSON. Cambiar cualquiera de esos textos
requiere modificar el paquete `admision/`, que pertenece a la cátedra y será reemplazado
al evaluar la entrega.

La solución mantiene el protocolo JSON y separa los textos en una especificación propia:

```text
configuraciones/prompts/*.json
              │
              ▼
          PromptSpec ───────► identidad {nombre, version, sha256}
              │
              ▼
         AgentePrompt ──────► modelo_lote(conversaciones)
              │
              ▼
     VistaDelAgente.paso(herramienta, argumentos)
```

## Responsabilidades

- `admision_tp/prompts.py`: valida, renderiza, carga y calcula la identidad de un prompt.
- `configuraciones/prompts/baseline.json`: copia explícita y verificable de todos los
  textos del agente de referencia.
- `admision_tp/agente_prompt.py`: coordina partidas y lotes con un `PromptSpec`; no carga
  modelos ni conoce reglas de negocio.
- `tests/test_prompts.py`: integridad, placeholders, hash y equivalencia de textos.
- `tests/test_agente_prompt.py`: equivalencia de conversaciones, herramientas, formatos,
  reintentos y lotes mediante un modelo simulado.

Esta unidad no modificó `admision/`, el buscador, el modelo ni el entrenamiento; la conexión
con `entrega.py` llegó después, con el issue #7.

## `PromptSpec`

Una especificación contiene:

| Campo | Responsabilidad |
|---|---|
| `version` | Versión positiva del esquema y sus textos. |
| `nombre` | Etiqueta humana; no cambia el hash del contenido. |
| `system_prompt` | Mensaje de sistema completo. |
| `mensaje_inicial` | Plantilla con exactamente `{paciente_id}`. |
| `formato_observacion` | Plantilla con `{herramienta}` y `{observacion}`. |
| `mensaje_json_invalido` | Corrección literal luego de una respuesta sin JSON válido. |
| `sha256` | Hash canónico de la versión y los cuatro textos anteriores. |

Al cargar un JSON se rechazan campos desconocidos, placeholders incorrectos, textos
vacíos y cualquier contenido que no coincida con su `sha256`. El nombre queda fuera del
hash para que dos etiquetas del mismo contenido conserven la misma identidad semántica.

Ejemplo de carga y uso (es lo que hace `entrega.construir_agente()`):

```python
from pathlib import Path

from admision_tp.agente_prompt import AgentePrompt
from admision_tp.prompts import PromptSpec

prompt = PromptSpec.cargar(Path("configuraciones/prompts/baseline.json"))
agente = AgentePrompt(modelo_lote, prompt)
print(agente.identidad_prompt)
conversaciones = agente.jugar_lote(entornos)
```

`identidad_prompt` se expone para que el futuro runner la copie a cada manifiesto. No se
agrega a los mensajes del modelo porque eso cambiaría el baseline.

## Compatibilidad baseline

El archivo baseline preserva byte por byte el resultado actual de:

- `mensaje_de_sistema()`;
- `primer_mensaje(paciente_id)`;
- `Resultado de <herramienta>:\n<observación>`;
- la corrección usada luego de un JSON inválido.

Conserva tres respuestas consecutivas sin JSON. Desde el issue #7, el tope de turnos ya no es
16 fijo: sin `max_turnos` explícito se deriva del tope de 40 llamadas del kit (ver abajo). El
coordinador sigue usando `admision.ejemplo.extraer_json` y llama al entorno con los mismos
argumentos. El hash actual del baseline es:

```text
944ae7f6f52d2052b43be770f891593abfbefb86af088df672687590bfeaf474
```

Una actualización futura del agente de la cátedra que cambie esos textos hará fallar la
prueba de equivalencia; no actualizar el JSON ni el hash silenciosamente.

## Longitud y métrica de candidatos

`admision_tp.evaluacion_prompts.medir_longitud_prompt` recibe un `PromptSpec` y el
tokenizer de la revisión real del modelo. No aproxima caracteres: tokeniza por separado
los cuatro textos configurables sin tokens especiales y mide también el contexto inicial
con la plantilla de chat, `add_generation_prompt=True` y `enable_thinking=False`.
Registra el SHA-256 de esa plantilla y usa `P-METRICA` como id estable para que candidatos
distintos sean comparables. La función no descarga nada; el futuro runner le inyecta el
mismo tokenizer que usa el modelo.

`calcular_metricas_candidato` consume los resultados producidos por
`admision.correr.puntuar` y las trazas del mismo conjunto. Reutiliza `CONTROLES` del kit:
no replica reglas ni calcula respuestas esperadas. Informa casos 9/9, controles parciales,
desglose por control, ausencia de decisión, respuestas sin JSON reconocible, llamadas
inválidas, llamadas, búsquedas, contexto, segundos y longitud del prompt.

La comparación es lexicográfica y mayor-es-mejor: primero casos 9/9, luego controles
correctos y recién después errores y eficiencia. Por eso una reducción de tokens nunca
compensa perder un caso completo. Los candidatos solo son comparables si se midieron con
los mismos casos, esperadas, normas, modelo, revisión y tokenizer; el runner auditable
debe fijar y hashear esas entradas.

## Runner reproducible

`admision_tp.experimentos_prompt.ejecutar_evaluacion_prompt` recibe rutas externas de
prompt, configuración, casos, respuestas esperadas y normas. No genera ninguna de ellas.
Usa las fases públicas del kit de forma separada:

1. reserva una salida nueva, verifica el origen de los datos (ver abajo) y copia prompt,
   configuración, casos y normas;
2. juega con `AgentePrompt` sin leer ni copiar las respuestas esperadas;
3. recién después conserva una copia exacta de las esperadas y llama a
   `admision.correr.puntuar`;
4. calcula la métrica y escribe el manifiesto final.

Casos y esperadas deben tener exactamente los mismos ids. El runner no acepta adapters:
la configuración representa Gemma sin entrenar para esta etapa. `buscador_variante` puede
ser cualquier variante de `admision_tp.buscador.VARIANTES` (`baseline`, `sin_comunes`, `prefijos`, las
mismas que acepta `BUSCADOR_VARIANTE` en la entrega); el ejemplo mantiene `baseline`. El runner **no**
lee `BUSCADOR_VARIANTE`: la variante sale solo de la configuración, que queda copiada en la corrida.

La configuración cerrada vive en
`configuraciones/evaluacion-prompt-ejemplo.json` y fija modelo, revisión HF inmutable,
semilla, límites de generación/contexto, lote, dtype, device map, buscador y límites del
coordinador. No contiene credenciales. Para una corrida real, con entorno ya instalado y
GPU autorizada:

```bash
uv run --locked --no-sync python -m scripts.evaluar_prompts \
  --prompt configuraciones/prompts/baseline.json \
  --configuracion configuraciones/evaluacion-prompt-ejemplo.json \
  --casos <prompt-dev-casos.jsonl> \
  --esperadas <prompt-dev-esperadas.jsonl> \
  --normas <carpeta-normas> \
  --salida corridas/prompts/baseline-001
```

No usar `practica/`, test ni las 800 trazas de referencia como conjunto de selección.
La ruta debe ser nueva: una corrida exitosa o fallida nunca se sobrescribe.

En Kaggle (T4 x2) usar `configuraciones/evaluacion-prompt-kaggle-t4.json`: idéntica al
ejemplo A100 salvo `dtype=float16` y `device_map=auto`, como `infra/kaggle_baseline.py`
(T4 no tiene bf16 nativo). Las mediciones T4 son desarrollo y se reportan separadas de A100.
El runner toma `HF_TOKEN` del entorno y, si falta, del Kaggle Secret `HF_TOKEN`
(`kaggle_secrets` se importa solo en ese caso); sin token falla antes de cargar el modelo.
El token no se imprime ni se guarda en manifiestos.

### Guardas de datos y propósito

Antes de copiar nada, el runner:

- rechaza casos o esperadas cuya ruta resuelta (symlinks incluidos) cae dentro de
  `practica/` del repo;
- corre `admision.correr.revisar_que_no_haya_esperadas` del kit sobre las rutas
  **originales** de casos y normas. La copia en `salida/entradas/` nunca tiene
  `esperadas-*.jsonl` al lado, así que el chequeo que hace `jugar` sobre ella no alcanza.

Si una guarda falla, la corrida queda `fallida` sin copiar entradas ni esperadas.
Una prueba de plomería con práctica exige `--solo-plomeria` (o `proposito="plomeria"`):
omite esas dos guardas, `jugar` sigue revisando la copia y el manifiesto registra
`"proposito": "plomeria"`. Toda corrida registra `proposito` y `guardas`.

Casos o esperadas malformados terminan en `ValueError` con mensaje claro
(`casos con formato inválido (...)`), manifiesto `fallida` y código de salida 1, sin traceback.

### Regla de comparabilidad

Solo compite una corrida con `estado=completada` y `proposito=evaluacion`. El runner
marca `comparable` en `metricas.json` y `manifiesto.json`; si no lo es, deja
`clave_comparacion=null` y `motivo_no_comparable`. `clave_comparacion(metricas)` exige
`comparable=true` y `clave_de_manifiesto(manifiesto)` revisa además estado y propósito;
ambas levantan `ValueError` en otro caso. Una corrida `completada_con_error` (por ejemplo, un
fallo del modelo: el kit vacía todas las conversaciones) también sale con código 1. Desde el
issue #7, un caso que excede el contexto ya no hace fallar el lote: termina solo ese caso sin
decisión (guarda de `AgentePrompt` con `ModeloLocal.medir_contexto`).

### Artefactos

La salida contiene:

- `entradas/`: snapshots exactos de prompt, configuración, casos, esperadas y normas;
- `trazas.jsonl` y `tiempos.json`: fase sin respuestas esperadas;
- `resultados.jsonl` y `resumen.json`: nueve controles del evaluador del kit;
- `metricas.json`: calidad, errores, eficiencia y tokens del candidato;
- `manifiesto.json`: estado, commit/dirty, identidad semántica del prompt, configuración,
  hashes de todas las entradas y artefactos, y métricas.

Si una entrada, la carga o la ejecución falla, la carpeta queda reservada con
`estado=fallida`, tipo/mensaje de error y todos los snapshots que pudieron conservarse.
Si el modelo falla dentro del runner del kit, se preservan trazas y diagnóstico y el estado
es `completada_con_error`; esa corrida queda sin clave de comparación (ver regla arriba).

## Crear una variante

Crear otro JSON dentro de `configuraciones/prompts/`, cambiar `nombre` y solo los textos
que forman parte del experimento. Para construir el JSON y su hash sin duplicar la lógica:

```python
from dataclasses import replace
import json
from pathlib import Path

from admision_tp.prompts import cargar_prompt_baseline

variante = replace(
    cargar_prompt_baseline(),
    nombre="instrucciones-breves-v1",
    system_prompt="<nuevo mensaje de sistema>",
)
Path("configuraciones/prompts/instrucciones-breves-v1.json").write_text(
    json.dumps(variante.como_dict(), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
```

El script de optimización deberá automatizar esta escritura y agregar configuración,
semilla, commit, estado Git, datos y resultados. El ejemplo solo explica el formato; no es
todavía un workflow completo de experimentos.

## Tests CPU

Tests focalizados, sin modelo ni dependencias nuevas:

```bash
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_prompts.py' -v
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_agente_prompt.py' -v
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_evaluacion_prompts.py' -v
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_experimentos_prompt.py' -v
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_prompt_dev.py' -v
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_seleccion_prompts.py' -v
```

Suite completa:

```bash
uv run --locked --no-sync python -m unittest discover -s tests -v
```

Verificación local de esta unidad, con Python 3.12 del sistema porque este checkout no
tiene `uv` ni `.venv`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests \
  python3 -m unittest test_prompts test_agente_prompt \
  test_evaluacion_prompts test_experimentos_prompt -q
# 21 tests, todos pasan

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests \
  python3 -m unittest discover -s tests -q
# 137 tests, 136 pasan y 1 se omite (tokenizer real local opt-in)
```

Esta ejecución no reproduce el lock ni valida `transformers`; solo prueba contratos CPU
con dobles y las interfaces públicas del kit disponibles en el checkout.

Verificación realizada el 2026-11-05 con Python del sistema porque `.venv` no
estaba disponible en el entorno del agente:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Resultado: **119 tests ejecutados, 118 pasan y uno se omite** porque requiere un snapshot
local explícito del tokenizer real. Los diez tests nuevos son CPU con modelos simulados.
No se descargaron pesos, no se ejecutó GPU y no se midió calidad del modelo.

## Contrato de formato (#7 ↔ #13)

Qué ve el modelo depende de dos dueños y un único archivo:

| Parte | Dueño | Dónde vive |
|---|---|---|
| `system_prompt`, `mensaje_inicial`, `mensaje_json_invalido` (textos) | Eze (#13; antes Martu) | archivo `PromptSpec` |
| Ciclo: turnos, reintentos, tope de 40 llamadas, guarda de contexto | Eze (#7) | `admision_tp/agente_prompt.py` |
| Presentación de observaciones (`formato_observacion`) | Eze (#7) | archivo `PromptSpec` |
| Selección y texto de las secciones de búsqueda | #6 (cerrado) | `admision_tp/buscador.py` |

Reglas:

- Todo cambio de texto o formato es un **archivo nuevo** en `configuraciones/prompts/` con su
  `sha256`; `baseline.json` no se edita. `entrega.py` toma el archivo de `PROMPT_SPEC` (relativo
  a la raíz del repo) o el baseline, y falla antes de cargar el modelo si el hash no coincide.
- Cualquier cambio de `PromptSpec` obliga a **regenerar los datos de #8** con ese archivo:
  `python -m scripts.generar_conversaciones --prompt <spec>`. Agente y generador renderizan con
  el mismo código (`AgentePrompt` + `PromptSpec.observacion_para`); un test rejuega las
  conversaciones generadas con `AgentePrompt` y exige igualdad exacta.
- El agente corta sin decisión un caso que no entra en el contexto, sin recortar nada; un
  prompt más largo reduce el margen de los casos largos.
- Las configuraciones originales del runner fijan `max_turnos: 16`; la entrega ya no. Decisión de #13
  (2026-11-10): los candidatos se miden con `evaluacion-prompt-kaggle-t4-l4.json`, `max_turnos: 120`,
  el mismo tope que deriva la entrega (ver [fase 1](#fase-1-2026-11-10-eze-conjunto-de-desarrollo-candidatos-y-evaluación-preparada)).

**Dónde está el contexto.** `python -m scripts.medir_contexto --trazas <trazas.jsonl> --prompt <spec>`
rejuega trazas ya jugadas con otro `PromptSpec` (mismas respuestas y observaciones) y reparte
`caracteres_de_contexto` del kit por tipo de mensaje. Antes de comparar, exige que el prompt de
origen reproduzca exactamente cada conversación grabada. Sobre las 100 trazas v22 (Kaggle T4,
Gemma sin entrenar, 50 casos × buscadores `baseline` y `prefijos`; artefactos locales no versionados):

| Parte | `baseline` | `prefijos` |
|---|---:|---:|
| Mensaje de sistema | 55,0 % | 54,2 % |
| Búsquedas | 24,8 % | 25,7 % |
| Otras observaciones | 13,4 % | 13,3 % |
| Respuestas del modelo | 5,0 % | 5,1 % |
| Mensaje inicial | 1,8 % | 1,7 % |
| Caracteres por caso | 19.954 | 20.689 |

El mensaje de sistema (2.232 caracteres) se reenvía en cada turno: es la palanca principal. Con
los ~5 turnos medios de v22 (máximo 8), cada 100 caracteres menos de sistema ahorran ~2,5 % por caso.

Candidatos de formato que controla el agente (ahorro total sobre `caracteres_de_contexto`, mismas trazas):

| Candidato | `baseline` | `prefijos` | Decisión |
|---|---:|---:|---|
| `{herramienta}:\n{observacion}` | 0,67 % | 0,67 % | No adoptado: sin pérdida, pero no justifica regenerar #8 |
| `[{herramienta}]\n{observacion}` | 0,62 % | 0,62 % | No adoptado: ídem |
| `{observacion}` sin herramienta | 1,43 % | 1,45 % | No adoptado: exige relajar `PromptSpec` y debilita la validación herramienta/observación de #8 |
| Unir líneas en blanco de las observaciones | 0,10 % | 0,10 % | No adoptado: cambia la presentación del buscador (#6) |
| Reemplazar secciones ya mostradas por una referencia | 0,05 % | 0,79 % | No adoptado: cambia la presentación del buscador (#6) |

Supuesto: el modelo responde lo mismo con el formato nuevo; la medición es de formato, no de conducta.

## Fase 1 (2026-11-10, Eze): conjunto de desarrollo, candidatos y evaluación

Estado: **validado en GPU T4** (Kaggle v25–v27). Resultados, decisión y límites en
[Resultados](#resultados-kaggle-v25v27); esta sección describe el diseño.

### Diagnóstico de partida (baseline, práctica, Kaggle v22)

Gemma sin entrenar con el prompt de la cátedra resolvió 10/50 casos de práctica con normas
original. Controles fallados: `box_antes_del_registro` 14, `aviso_al_voluntariado` 14 (todos
"esperaba true, registró false"), `registro` 12 (9 esperaban `provisorio` y registró `definitivo`),
`verificacion` 12, `cobertura` 10, `entidad_a_notificar` 9, `plazo_circuito_min` 4. Causa común:
35/50 casos hicieron **una sola búsqueda** y decidieron de memoria; el prompt solo dice "Buscalas con
buscar_normativa antes de decidir". En contexto, el mensaje de sistema es el 55 % y las búsquedas el 25 %.

Ese diagnóstico es **agregado** (conteo de controles), sirve para formular hipótesis y no copia
nada de ningún caso al prompt. La selección se hace solo sobre el conjunto de desarrollo de abajo;
la corrida baseline sobre ese conjunto (fase 2) debe confirmar el mismo perfil de fallas antes de
atribuir una mejora a un candidato.

### Conjunto de desarrollo `datos/prompt-dev/v1`

```bash
uv run --locked --no-sync python -m scripts.generar_prompt_dev   # ~2 s, CPU
```

- **Casos:** `admision.generador` con semilla **20261110** (la de entrenamiento es 12012), 4.000
  candidatos. Fichas esperadas con el deductor fijo de Yas
  (`scripts.generar_sinteticos_entrenamiento.deducir_ficha`, reglas + las dos costumbres de
  `historial_costumbres.json`), con los valores de la carpeta de cada caso; el script solo agrega la
  parte `verificacion` del formato del kit (`requerida`: la afiliación que eligió el deductor;
  `prohibidas`: cualquier otra entidad que el paciente tenga). Un test juega cada caso en el `Entorno`
  del kit verificando lo requerido y registrando la esperada: los 40 pasan los 9 controles.
- **Normas:** K=2 carpetas con los textos de `normas/original` byte a byte y valores sorteados entre
  los observados en las 800 trazas de referencia (mismo criterio y código que #8,
  `normas_variables`). Nunca se leen `normas/b` ni `normas/c`.
  `n1`: P-ADM-1 1, P-ADM-2 70, P-ADM-3 1, P-COB-2 9, P-TDE-1 25, P-TDE-3 30.
  `n2`: P-ADM-3 3, P-TDE-1 25, P-TDE-2 20, P-TDE-3 30, P-TDE-4 45, P-TDE-5 6 (el resto, como original).
- **Por qué K=2 × N=20:** el test cambia los valores de las normas, así que un prompt que solo
  funciona con los valores originales tiene que perder; dos carpetas distintas del original lo
  muestran con el mismo costo que una de 40. 40 casos es del orden de práctica (1 caso = 2,5 %) y
  con cuotas de 4 por circuito; duplicarlo duplica el tiempo de GPU de las cinco evaluaciones
  (ver estimación). El runner toma una carpeta de normas por corrida: son dos corridas por prompt.
- **Balance (cuotas exactas por carpeta):** 4 por circuito del triage (ecg, ACV, sepsis probable,
  sepsis posible, ninguno); 3 ART, 3 dos credenciales, 3 acompañante responde, 3 sin nadie y 8
  generales; 4 mayores de 75 sin acompañante (aviso `true`) y 2 mayores con acompañante. Total:
  aviso `true` 8/40, box `true` 28/40, registro provisorio 21/40, sin_cobertura 21/40, con entidad
  prohibida 9/40. Detalle por carpeta en `manifiesto.json`.
- **Sin solapamiento:** se excluye todo paciente cuya nota de triage (sin el id), id + nombre y edad,
  o cualquier número de documento/afiliado/CUIL aparezca en `practica/casos.jsonl` (nunca sus
  esperadas), en las 800 trazas de referencia, en las 2.000 del historial, en
  `datos/entrenamiento/` y `datos/conversaciones/v2-baseline/`, o en los 5.000 candidatos de la
  semilla 12012. Resultado: 1 candidato excluido por número compartido.
- **Disposición:** `entradas/n<k>/{casos.jsonl,normas/}` y, aparte, `esperadas/esperadas-n<k>.jsonl`,
  fuera del alcance de `revisar_que_no_haya_esperadas` del kit. La salida nunca se sobrescribe; el
  manifiesto guarda semilla, valores, cuotas, exclusiones y hashes de entradas y archivos. Un test
  regenera el conjunto y exige bytes idénticos a los versionados. El manifiesto versionado registra
  `dirty=true` (se generó antes del commit); regenerar en un árbol limpio lo deja en `false`.

### Candidatos

Los cuatro conservan `mensaje_inicial`, `formato_observacion` y `mensaje_json_invalido` del baseline:
el formato es de #7 y cambiarlo obliga a regenerar #8 sin ganancia que lo justifique (≤1,4 % de
contexto, ver arriba). Solo cambia `system_prompt`.

| Archivo | Hipótesis | Qué cambia |
|---|---|---|
| `corto-v1.json` (A) | El largo cuesta sin aportar | Mismo contenido, descripciones de herramientas compactas; "las secciones más relevantes" sin fijar el k del buscador |
| `procedimiento-v1.json` (B) | Falla por buscar una vez y decidir de memoria | Baseline + procedimiento: triage; una búsqueda por tema (registro, box por nivel y edad, circuito + plazo + ambulancia, cobertura + ART + terminación); preguntar; verificar antes de decidir; registrar |
| `procedimiento-costumbres-v1.json` (C) | `aviso` y dos credenciales no están en las normas | B + las dos costumbres |
| `procedimiento-costumbres-corto-v1.json` (D) | Lo de C con menos tokens | C compactado |

B y C empiezan con el texto exacto del baseline (C con el de B): la comparación aísla cada agregado.
A y D aíslan la compactación. Costumbres de C/D: "aviso_al_voluntariado es true solo si el paciente
tiene 75 años o más y no tiene acompañante" y "con obra social y prepaga, verificá solo la prepaga; si
está inactiva, sin_cobertura; en accidente de trabajo rige la ART".

**Sin valores de las normas.** Ningún prompt menciona valores de parámetros (niveles, edades,
plazos, dígitos): el test de la cátedra los cambia, y B/C/D piden leerlos de las secciones recuperadas.
Los códigos de sección sí aparecen (el baseline ya los nombra y no cambian entre `original`, `b` y `c`).
**El 75 no es un parámetro de norma:** es el umbral de una costumbre que no está escrita en ninguna
norma y se infirió del historial (`historial_costumbres.json`, 1.609/1.760 filas limpias); el test
cambia `parametros.yaml`, no el historial. Un test exige que sea el único número libre de C y D y que
coincida con `umbral_edad` del informe de costumbres. Si la cátedra variara también las costumbres,
C y D fallarían en el aviso: es el riesgo aceptado de escribirlas.

### Medición CPU

Contexto (`caracteres_de_contexto` del kit) rejugando las 50 trazas v22 `baseline` con cada prompt
(`scripts.medir_contexto`). **Mide formato, no conducta:** supone las mismas respuestas del modelo.
Si B/C/D logran que el modelo busque más, cada caso tendrá más llamadas y más contexto que esto.
Tokens: solo el baseline está medido con el tokenizer real (615, v23); el resto es estimación
proporcional a caracteres (615/2.232). **Los tokens reales de A–D (453, 857, 969, 659) están
en [Resultados](#resultados-kaggle-v25v27);** esta tabla queda como estimación previa.

| Prompt | Caracteres sistema | vs baseline | Tokens sistema | Contexto/caso (v22 rejugado) | vs baseline |
|---|---:|---:|---:|---:|---:|
| `baseline` | 2.232 | — | 615 (real) | 19.954 | — |
| A `corto-v1` | 1.632 | −26,9 % | ~450 | 17.002 | −14,8 % |
| B `procedimiento-v1` | 3.224 | +44,4 % | ~888 | 24.835 | +24,5 % |
| C `procedimiento-costumbres-v1` | 3.639 | +63,0 % | ~1.003 | 26.877 | +34,7 % |
| D `procedimiento-costumbres-corto-v1` | 2.405 | +7,8 % | ~663 | 20.805 | +4,3 % |

```bash
uv run --locked --no-sync python -m scripts.medir_contexto \
  --trazas corridas/kaggle/v22/baseline-original-n50-l4/salida/trazas.jsonl \
  --prompt configuraciones/prompts/baseline.json --prompt configuraciones/prompts/corto-v1.json \
  --prompt configuraciones/prompts/procedimiento-v1.json \
  --prompt configuraciones/prompts/procedimiento-costumbres-v1.json \
  --prompt configuraciones/prompts/procedimiento-costumbres-corto-v1.json
```

### Configuración de evaluación

`configuraciones/evaluacion-prompt-kaggle-t4-l4.json`: como la de T4 (float16, `device_map=auto`)
con lote 4 y `max_turnos: 120` = `max_turnos_para(40, 3)`, el mismo tope que la entrega deriva de las
40 llamadas del kit (un test lo exige). El tope de 40 llamadas y la guarda de contexto son los de
`AgentePrompt`. `...-t4-l4-prefijos.json` es idéntica salvo `buscador_variante: prefijos` (fase 2).
Las cinco evaluaciones de fase 1 usan la misma configuración: si no, el ranking las rechaza.

### Kaggle: celda de fase 1 (ejecutada en v25 con el commit `f4d7300`)

Notebook con GPU T4 x2, Internet y Secrets `HF_TOKEN` (lectura) y `GITHUB_TOKEN` (como en
[kaggle.md](kaggle.md)). Una sola celda, ejecutada una vez, sin tocar el notebook hasta que termine.
Reemplazar `COMMIT` por el SHA completo pusheado de `feat/13-prompt-eze`. Escribe fuera del repo
(si no, el runner registraría `dirty=true`) y falla si la carpeta de la sesión ya existe.

```python
import base64, json, os, subprocess, time
from kaggle_secrets import UserSecretsClient

COMMIT = "<sha completo de feat/13-prompt-eze>"
RAMA, REPO = "feat/13-prompt-eze", "/kaggle/working/repo"
BASE = "/kaggle/working/corridas/prompt-dev-f1"
CONFIG = "configuraciones/evaluacion-prompt-kaggle-t4-l4.json"
PROMPTS = ["baseline", "corto-v1", "procedimiento-v1", "procedimiento-costumbres-v1",
           "procedimiento-costumbres-corto-v1"]
CARPETAS = ["n1", "n2"]

# 1. Repo en el commit exacto. El token va solo por entorno, nunca en argumentos ni en la salida.
t = UserSecretsClient().get_secret("GITHUB_TOKEN")
env_git = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraheader",
           "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic "
           + base64.b64encode(f"x-access-token:{t}".encode()).decode()}
del t
if not os.path.exists(REPO):
    r = subprocess.run(["git", "clone", "--branch", RAMA,
                        "https://github.com/FranCalveyra/tp_final_ia_gen.git", REPO],
                       env=env_git, capture_output=True, text=True)
    print("clone", r.returncode)
r = subprocess.run(["git", "-C", REPO, "fetch", "origin", RAMA], env=env_git, capture_output=True, text=True)
print("fetch", r.returncode)
del env_git
subprocess.run(["git", "-C", REPO, "checkout", "--detach", COMMIT], capture_output=True)
head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, f"HEAD {head} no es {COMMIT}"

# 2. Dependencias (ver kaggle.md: los paquetes torch* de la imagen rompen transformers).
subprocess.run(["nvidia-smi"])
assert subprocess.run(["pip", "install", "-q", "-r", f"{REPO}/requirements-inferencia.txt"]).returncode == 0
subprocess.run(["pip", "uninstall", "-y", "torchvision", "torchaudio", "torchcodec", "torchao", "torchdata"])

# 3. Cinco prompts × dos carpetas, propósito evaluacion (sin --solo-plomeria). Carpeta por carpeta:
#    si la sesión se corta, quedan todos los prompts de n1 completos.
os.makedirs(BASE)  # falla si existe: nunca sobrescribir
estado = []
for carpeta in CARPETAS:
    for prompt in PROMPTS:
        nombre = f"{prompt}-{carpeta}"
        cmd = ["python", "-m", "scripts.evaluar_prompts",
               "--prompt", f"configuraciones/prompts/{prompt}.json", "--configuracion", CONFIG,
               "--casos", f"datos/prompt-dev/v1/entradas/{carpeta}/casos.jsonl",
               "--esperadas", f"datos/prompt-dev/v1/esperadas/esperadas-{carpeta}.jsonl",
               "--normas", f"datos/prompt-dev/v1/entradas/{carpeta}/normas",
               "--salida", f"{BASE}/{nombre}"]
        inicio = time.time()
        with open(f"{BASE}/{nombre}.log", "w") as log:
            r = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        estado.append({"corrida": nombre, "returncode": r.returncode, "segundos": round(time.time() - inicio)})
        print(estado[-1], flush=True)
        with open(f"{BASE}/sesion.json", "w") as f:
            json.dump({"commit": COMMIT, "config": CONFIG, "corridas": estado}, f, indent=2)
subprocess.run(["zip", "-qr", "/kaggle/working/prompt-dev-f1.zip", BASE])
```

Descargar `prompt-dev-f1.zip` desde el panel de archivos (no ejecuta celdas), descomprimir en
`corridas/kaggle/<version>/` y rankear localmente:

```bash
uv run --locked --no-sync python -m scripts.rankear_prompts corridas/kaggle/<version>/prompt-dev-f1
```

`scripts.rankear_prompts` (lógica en `admision_tp/seleccion_prompts.py`) ordena por casos 9/9
sumados sobre las carpetas y, a igualdad, por tokens del sistema medidos por el runner con el
tokenizer real; muestra los nueve controles sumados. **Rechaza** (código 1, no omite) corridas que no
son `completada` + `evaluacion`, `resultados.jsonl` que no coinciden con el hash del manifiesto,
configuraciones o plantillas de chat distintas, y prompts sin exactamente una corrida por carpeta.
Este orden es el de la selección pedida (calidad, después largo); la clave del runner sigue
desempatando primero por controles parciales.

**Estimación de GPU (T4 x2, lote 4).** Base v22: ~15 s/caso con el baseline (~5 llamadas por caso),
~2,5 min de carga del modelo por corrida, instalación 4–20 min. Fase 1 = 10 corridas × 20 casos:
carga ~25 min; baseline y A ~20 min (80 casos × 15 s); B, C y D, si siguen el procedimiento
(~7–9 llamadas por caso), ~25 s/caso: ~50 min (120 casos). **Total ≈ 1 h 35 min a 2 h** de sesión.
No es una garantía: la cuota y el tiempo real de Kaggle no se verificaron.

### Fase 2 (plan original; lo que se corrió está en [Resultados](#resultados-kaggle-v25v27))

1. Elegir el ganador con `rankear_prompts`, mirando también ganancias y pérdidas caso por caso
   (`resultados.jsonl`) y que el baseline en el dev reproduzca el perfil de fallas de práctica.
2. Paso 6: el ganador con el buscador `prefijos` (que en v22 perdió con el baseline del prompt:
   6/50 contra 10/50). El runner **no** lee `BUSCADOR_VARIANTE`: la variante sale de
   `buscador_variante` en la configuración, por eso existe `...-t4-l4-prefijos.json`. Mismo bloque
   de repo y dependencias que la celda de fase 1 (`COMMIT`, pasos 1 y 2), y como paso 3:

```python
ELEGIDO = "<nombre del prompt elegido, p. ej. procedimiento-costumbres-corto-v1>"
BASE2 = "/kaggle/working/corridas/prompt-dev-f2-prefijos"
CONFIG2 = "configuraciones/evaluacion-prompt-kaggle-t4-l4-prefijos.json"
os.makedirs(BASE2)  # falla si existe
estado = []
for carpeta in ["n1", "n2"]:
    nombre = f"{ELEGIDO}-prefijos-{carpeta}"
    cmd = ["python", "-m", "scripts.evaluar_prompts",
           "--prompt", f"configuraciones/prompts/{ELEGIDO}.json", "--configuracion", CONFIG2,
           "--casos", f"datos/prompt-dev/v1/entradas/{carpeta}/casos.jsonl",
           "--esperadas", f"datos/prompt-dev/v1/esperadas/esperadas-{carpeta}.jsonl",
           "--normas", f"datos/prompt-dev/v1/entradas/{carpeta}/normas",
           "--salida", f"{BASE2}/{nombre}"]
    inicio = time.time()
    with open(f"{BASE2}/{nombre}.log", "w") as log:
        r = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    estado.append({"corrida": nombre, "returncode": r.returncode, "segundos": round(time.time() - inicio)})
    print(estado[-1], flush=True)
    with open(f"{BASE2}/sesion.json", "w") as f:
        json.dump({"commit": COMMIT, "config": CONFIG2, "corridas": estado}, f, indent=2)
subprocess.run(["zip", "-qr", "/kaggle/working/prompt-dev-f2-prefijos.zip", BASE2])
```

   ~25–45 min con instalación (2 cargas + 40 casos). `rankear_prompts` rechaza mezclar estas corridas
   con las de fase 1 (otra configuración): se compara el ganador con `baseline` y con `prefijos` como
   dos rankings y caso por caso.
3. Publicar el ganador como `PROMPT_SPEC`, regenerar #8 con él y registrar antes/después en práctica
   (solo evaluación, nunca selección).

### Decisiones y límites

- **Hipótesis desde práctica (decidido por Eze, 2026-11-10):** los candidatos se **diseñaron**
  a partir del perfil **agregado** de fallas en práctica (v22): el agente busca una sola vez en 35/50
  casos y decide de memoria, y nunca aplica la costumbre del voluntariado. El análisis independiente
  de #14 da el mismo perfil (ver [análisis de errores](analisis-errores.md)). Los prompts no copian
  contenido de ningún caso de práctica. La **selección** se hace solo sobre el conjunto de desarrollo,
  y la práctica se mide una sola vez por prompt, para reportar el antes y el después. Esta frontera
  (diseño desde patrones generales, selección fuera de práctica) se informa así en la Parte 3.
- **Búsqueda manual estructurada en lugar de DSPy/GEPA (decidido por Eze, 2026-11-10):** la
  consigna sugiere DSPy/GEPA, pero no lo exige. Con ~2 h de T4 por ronda de cinco candidatos, una
  búsqueda automática con Gemma local no entra en el calendario; queda como trabajo futuro.
- **Deductor:** las esperadas dependen de que el deductor de Yas sea correcto. Se valida que sean
  resolubles en el `Entorno` y que no violen normas (`razones_normativas`), no contra fichas de práctica.
- **Distribución:** el generador produce muchas `sin_cobertura` (21/40) y box `true` (28/40) con estos
  valores; se informa, no se reequilibra (cambiaría la distribución del generador de la cátedra).
- **Tokens de A–D:** la tabla de "Medición CPU" es la estimación previa; los reales (453, 857, 969,
  659) están en [Resultados](#resultados-kaggle-v25v27).

## Resultados (Kaggle v25–v27)

Gemma 4 E4B sin entrenar, 2× T4, float16, lote 4, `max_turnos` 120, semilla 20261109. Corridas en
`corridas/kaggle/v25/prompt-dev-f1/`, `v26/prompt-dev-f1b/` y `v27/evaluacion-solo-prompt/` (no versionadas).
Tokens del sistema medidos por el runner con el tokenizer real.

### Fase 1: conjunto de desarrollo (40 casos, buscador `baseline`)

```bash
uv run --locked --no-sync python -m scripts.rankear_prompts \
  corridas/kaggle/v25/prompt-dev-f1/{baseline,corto-v1,procedimiento-v1,procedimiento-costumbres-corto-v1}-n{1,2}
```

| Prompt | n1 | n2 | **Total /40** | Tokens sistema | Contexto/caso (caracteres) | Búsquedas/caso |
|---|---:|---:|---:|---:|---:|---:|
| B `procedimiento-v1` | 4 | 5 | **9** | 857 | 51.475 / 54.583 | 3,5 / 3,65 |
| `baseline` | 4 | 4 | **8** | 615 | 20.104 / 17.732 | 1,45 / 1,25 |
| D `procedimiento-costumbres-corto-v1` | 0 | 2 | **2** | 659 | 24.372 / 20.140 | 1,45 / 1,3 |
| A `corto-v1` | 0 | 1 | **1** | 453 | 13.655 / 14.202 | 0,95 / 0,95 |
| C `procedimiento-costumbres-v1` | falla | 6 | **no comparable** | 969 | — / 54.444 | — / 3,3 |

Controles sumados (B contra baseline): `registro` 33 vs 27, `box_antes_del_registro` 32 vs 24, `circuito`
40 vs 38, `plazo_circuito_min` 32 vs 30, `cobertura` 33 vs 32; pierde en `entidad_a_notificar` 32 vs 35
y `verificacion` 27 vs 33; `aviso_al_voluntariado` igual (32). El baseline en el dev reproduce el
perfil de práctica: ~1,3 búsquedas por caso y fallas en registro, box y plazo.

**Caso por caso hay mucho recambio.** B en n1 gana 3 casos y pierde otros 3; en n2 gana 2 y pierde 1.
Con 20 casos por carpeta, 1–2 casos de diferencia están dentro del ruido: B **no es una mejora
demostrada**, es el único candidato que no empeora. Lo que sí es claro:

1. **Acortar empeora.** A y D pierden casi todo: sin las descripciones detalladas de las herramientas,
   el modelo busca todavía menos (0,95 por caso con A) y falla `plazo_circuito_min` (18 y 13 de 40).
2. **El procedimiento cambia la conducta como se buscaba:** B y C buscan 3–4 veces por caso en vez de
   una, y `box_antes_del_registro` y `circuito` suben a 0,95–1,0 por carpeta. El costo es **~2,7× el
   contexto por caso** (~53.000 contra ~19.000 caracteres), y eso cuenta en la nota.
3. **C falla en GPU de forma reproducible.** `jugar_lote` termina con `AcceleratorError: CUDA error:
   unspecified launch failure` en n1 con el buscador `baseline` (3 de 3 intentos, v25 y v26) y en n2
   con `prefijos` (2 de 2), siempre en el mismo punto (mismo parcial 0,511 y 0,339: la generación es
   determinista). B, con el mismo procedimiento y contexto parecido, nunca falló (4 corridas). La
   causa no está diagnosticada (kernel de T4 en float16 con alguna secuencia concreta es la hipótesis);
   como el kit vacía las conversaciones del lote, la corrida no es comparable y el ranking la rechaza.
   Lo que corrió completo es prometedor (n2 6/20 con `baseline`; n1 5/20 con `prefijos`, contra 1/20
   de B), pero **no se puede elegir un prompt que no termina en el hardware de evaluación**.

### Paso 6: buscador `prefijos` (v26)

| Prompt | Buscador | n1 | n2 | Total /40 | Contexto/caso |
|---|---|---:|---:|---:|---:|
| B `procedimiento-v1` | `baseline` | 4 | 5 | **9** | ~53.000 |
| B `procedimiento-v1` | `prefijos` | 1 | 3 | **4** | ~47.000 |
| C `procedimiento-costumbres-v1` | `prefijos` | 5 | falla | no comparable | 51.684 (n1) |

`prefijos` vuelve a perder con el prompt elegido, como con el baseline en práctica (v22: 6/50 contra
10/50). **Se mantiene el buscador `baseline`.**

### Decisión

Con el criterio declarado antes de correr (más casos resueltos en el dev y, a igualdad, menos tokens),
**el prompt elegido es B `procedimiento-v1`** con el buscador `baseline`. En el dev la evidencia es débil
(+1 caso sobre 40); en práctica (solo reporte, abajo) la ganancia es clara: 42/150 contra 27/150. Cuesta
~2,6× de contexto por caso. Qué cambia y qué queda para decidir en equipo:

- **`entrega.py` sigue apuntando al baseline.** Recomendación: pasar el default a B (más casos resueltos
  en las tres versiones de práctica, a costa de contexto). Mientras tanto, para usar B:
  `PROMPT_SPEC=configuraciones/prompts/procedimiento-v1.json`.
- **Las conversaciones de #8 se regeneraron con B en una carpeta nueva**, `datos/conversaciones/v3-procedimiento`
  (la consigna pide entrenar con el prompt optimizado); `v2-baseline` queda igual. Solo cambia el mensaje de
  sistema ([detalle](conversaciones-entrenamiento.md)). Cuál se entrena lo decide #15 (Fran), porque define
  qué prompt usan las filas `solo_entrenamiento` y `ambos`. Lo que B no arregla (verificación, entidad,
  aviso) es justamente lo que el entrenamiento tiene que enseñar ([pasada 5](analisis-errores.md)).

### Fase 2: práctica, solo para reportar (v27)

`configuraciones/evaluacion/solo-prompt.json` (fila `solo_prompt`), puntuado con `scripts.tabla_evaluacion`.
La práctica no se usó para elegir.

```bash
uv run --locked --no-sync python -m scripts.tabla_evaluacion \
  --evaluacion corridas/kaggle/v25/evaluacion-actual --evaluacion corridas/kaggle/v27/evaluacion-solo-prompt \
  --salida corridas/tablas/2026-11-11-v25-v27
```

| Configuración | original | b | c | Total | Contexto/caso | Búsquedas/caso | s/caso (lote) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sin_entrenar` (baseline, v25) | 10/50 | 10/50 | 7/50 | 27/150 | ~19.800 | 1,33 | ~15,7 |
| **`solo_prompt` (B, v27)** | **14/50** | **16/50** | **12/50** | **42/150** | ~52.500 | 3,53 | ~25,3 |
| *Referencia cátedra, sin entrenar* | 12/50 | 15/50 | 12/50 | 39/150 | | | |

- **Caso por caso** (`scripts.analizar_errores comparar`, [pasada 6](analisis-errores.md)): original gana 7 y
  pierde 3, b gana 9 y pierde 3, c gana 7 y pierde 2. Juntas: 23 contra 8, signos p≈0,011; las tres versiones
  comparten pacientes (cambian las normas), así que no son independientes y el p es optimista. Por versión,
  p=0,15–0,34.
- **Por control**, igual que en el dev: `box_antes_del_registro` 0,72 → 0,90–0,96, `plazo` 0,92 → 0,96–1,00,
  `registro` sube en las tres; bajan `verificacion` (0,74–0,76 → 0,60–0,66) y `entidad_a_notificar`
  (→ 0,72). `aviso_al_voluntariado` queda en 0,72: B no tiene la costumbre.
- **Costo:** ~2,6× contexto y ~1,6× tiempo de lote por caso. No hay casos sin decisión.

La práctica **confirma la dirección** que el dev mostraba débil y supera a la referencia sin entrenar de
la cátedra en las tres versiones. No se usó para elegir: B ya era el elegido por el criterio declarado.

## Verificado y no verificado

**Probado con dobles CPU:** equivalencia completa de mensajes entre `AgenteJSON` y
`AgentePrompt` usando el baseline; orden y argumentos de las cinco herramientas; formato
de observaciones; corrección de JSON; cambio aislado del system prompt; orden del lote;
conversaciones completas; estabilidad e integridad del hash. Desde 2026-11-10, también
igualdad con `AgenteJSON` en los bordes: tres respuestas sin JSON seguidas, reinicio del
contador, `max_turnos`, límite de 40 llamadas del `Entorno` real del kit y lotes con casos
de distinto largo.

**Probado con dobles CPU:** además de la compatibilidad anterior, conteo exacto por IDs
del tokenizer inyectado, objeto mapping compatible con tokenizers reales, opciones de la
plantilla de chat, agregación de nueve controles, errores JSON, prioridad de casos completos,
separación temporal de esperadas, snapshots/hashes, salida exclusiva y fallos auditables.

**Validado en GPU T4 (2026-11-10/11):** los cinco prompts con Gemma real sobre el conjunto de
desarrollo, tokens reales de cada uno, `prefijos` con B y C, y B en práctica (v25–v27, ver
[Resultados](#resultados-kaggle-v25v27)). La integración con `entrega.py` se validó en v24.

**No verificado:** A100 (KI-A100-001: tiempos y resultados de T4 no son los de la entrega), la causa
del fallo CUDA de C, y si la ventaja de B sobre el baseline (+1 caso) se sostiene con más casos.

## Próximos pasos

1. ~~Conjunto prompt-dev independiente de práctica/test~~: `datos/prompt-dev/v1` (fase 1).
2. ~~Fase 1 en Kaggle y ranking~~ (v25–v26): B `procedimiento-v1`.
3. ~~`prefijos` con el elegido~~ (v26): pierde; se queda `baseline`.
4. ~~B en práctica para la fila `solo_prompt`~~ (v27).
5. ~~Regenerar #8 con el elegido~~: `datos/conversaciones/v3-procedimiento`.
6. **Decisión de equipo (Eze + Fran):** default de `entrega.py` y si #15 entrena con v3 (recomendado por la
   consigna) o v2, a la luz de la fila `solo_prompt` y del costo de contexto ×2,6.
7. Si sobra GPU: diagnosticar el fallo CUDA de C (correr el caso que lo dispara con lote 1) y probarlo
   en A100; C es el candidato con mejores parciales cuando termina.
8. DSPy/GEPA con versiones fijadas, solo si queda margen, sin servicios externos en el agente operativo.
