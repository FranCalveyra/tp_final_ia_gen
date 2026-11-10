# Optimización de prompt — infraestructura inicial

## Objetivo

El issue #13 implementa la Parte 3 de la misión: optimizar el mensaje de sistema y el
formato de los mensajes con Gemma 4 sin entrenar, antes del SFT. La comparación futura
debe considerar tanto los nueve controles como el largo del prompt y conservar evidencia
reproducible de cada candidato.

Esta primera unidad solo permite definir y ejecutar prompts configurables. No integra
DSPy/GEPA ni genera datos. Desde el issue #7 (2026-11-10), `entrega.py` usa `AgentePrompt`
con el `PromptSpec` de `PROMPT_SPEC` o el baseline (ver [contrato de formato](#contrato-de-formato-7--13)).

Las unidades siguientes agregan medición de tokens, una métrica común y un runner
auditable para candidatos. Todavía no se ejecutó Gemma ni se seleccionó un prompt: esas
operaciones quedan para corridas autorizadas sobre datos externos permitidos.

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
ser `baseline` o el candidato explícito `sin_comunes`; el ejemplo mantiene `baseline`.

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
| `system_prompt`, `mensaje_inicial`, `mensaje_json_invalido` (textos) | Martu (#13) | archivo `PromptSpec` |
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
- Las configuraciones del runner de #13 fijan `max_turnos: 16`; la entrega ya no. Si se quiere
  que los candidatos se midan como la entrega, conviene quitar ese tope (decisión de #13).

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

**No verificado:** comportamiento de una variante con Gemma real, cantidad real de tokens
del baseline, calidad sobre casos y ejecución en GPU. La integración con `entrega.py`
(issue #7) está probada con dobles CPU, no en GPU. Los
manifiestos se probaron con dobles y archivos temporales. Esta infraestructura no demuestra
ninguna mejora de prompt.

## Próximos pasos

1. Recibir un conjunto prompt-dev independiente de práctica/test, con respuestas esperadas
   auditables y splits fijados; esta unidad no lo genera.
2. Acordar con Eze/Yas el contrato y procedencia de esos archivos externos.
3. Medir el baseline con el tokenizer real fijado y Gemma sin entrenar, con autorización.
4. Incorporar DSPy/GEPA con versiones fijadas, sin llamar servicios externos desde el
   agente operativo.
5. Seleccionar el prompt con datos permitidos; publicarlo como archivo `PromptSpec`,
   regenerar las conversaciones de #8 con él y recién entonces apuntar `PROMPT_SPEC` (o el
   default de `entrega.py`) a ese archivo.
