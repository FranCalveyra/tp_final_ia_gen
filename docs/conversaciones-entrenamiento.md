# Conversaciones de entrenamiento — propuesta v1

Contrato mínimo propuesto para coordinar con Eze/Yas; todavía no fue acordado con ellos.
El conversor produce **JSONL UTF-8: un objeto por línea**, sin líneas vacías.

## Qué entregar

Cada objeto contiene exactamente:

- `version`: entero `1`.
- `id`: texto no vacío, único en el archivo.
- `origen`: `historial`, `referencia` (las 800 trazas de `entrenamiento/trazas.jsonl`, con las que la
  cátedra entrenó su modelo de referencia) o `sintetico`.
- `referencias`: lista no vacía de textos que identifican filas fuente o script/configuración/semilla sintéticos.
- `messages`: conversación completa en el formato actual del agente: objetos `role`, `content`, ambos textos.

Se conserva el contenido original; no se normalizan ni reescriben mensajes.
La conversación comienza `system`, `user`; luego alterna `assistant`, `user` y termina en `assistant`.
Cada respuesta del asistente es **JSON estricto** con `herramienta` y `argumentos`.
La última acción debe ser `registrar_decision`, sin observación posterior. Máximo 40 acciones.
Las observaciones siguen el `formato_observacion` del prompt usado; por defecto (y en el baseline)
`Resultado de <herramienta>:\n<texto no vacío>`, como en `AgenteJSON`. El agente de la entrega y el
generador renderizan con el mismo `PromptSpec`; un test exige que coincidan exactamente. `validar_registro` y
`validar_archivo` aceptan el formato como parámetro; la CLI, `--prompt <PromptSpec.json>`.

## Herramientas y tipos

Se admiten las cinco herramientas del kit con sus argumentos exactos (`admision/ejemplo.py`).
`fundamentos` es optativo. Los argumentos de consulta/pregunta/verificación son textos no vacíos.
Las enumeraciones usan los valores canónicos del kit. Los dos indicadores de la ficha son booleanos JSON;
plazo es entero no negativo o `null`; documento y entidad, texto no vacío o `null`.
Esto es deliberadamente más estricto que el entorno, que acepta algunas conversiones de tipos.

## Validación sin GPU

```bash
uv run --locked --no-sync python -m scripts.validar_conversaciones datos/conversaciones.jsonl
```

Imprime SHA-256 de los bytes originales, registros, válidos, origen y errores por línea.
Devuelve código 0 si todo cumple, 1 si hay errores y 2 si no se puede leer el archivo.
No imprime conversaciones ni datos de pacientes. No carga modelo ni modifica archivos.
Los mensajes de error de formato permiten localizar el problema por línea, no exponen el contenido.

## Límites IMPORTANTES

- Es un validador de **formato**, no prueba que las decisiones sean correctas ni que una observación venga del buscador real.
- La curación debe excluir errores de operadores. No generar respuestas desde fichas de práctica/test.
- Se rechazan orígenes distintos de los permitidos y referencias explícitas a `practica`/`test`.
  Esto NO detecta copias encubiertas: hay que auditar la procedencia por separado.
- Reintentos por JSON mal formado son válidos en operación, pero se rechazan como demostraciones curadas v1.
- El prompt final optimizado debe venir de Martu; el validador no lo inventa ni asegura que sea el correcto.
- Los ejemplos artificiales de `tests/test_conversaciones.py` solo prueban estructura: NO son dataset de entrenamiento.

## Conversor CPU de trazas (implementado)

`admision_tp.conversion_trazas` expone dos interfaces públicas:

- `convertir_traza(traza, *, origen, referencia, prompt=None, sistema=None, buscar=None)` devuelve una
  conversación v1. `prompt` es un `PromptSpec` y se usan sus textos (sistema, mensaje inicial y formato de
  observación); `sistema` es la forma anterior (solo reemplaza el sistema del baseline) y excluye a `prompt`.
  `buscar(consulta)`, si se pasa, reemplaza las observaciones de `buscar_normativa`; el resto se conserva.
- `convertir_archivo(entrada, salida, *, origen, sistema=None, prompt=None)` valida todo el JSONL antes de publicar.

Ambas requieren **origen y prompt explícitos**: no adivinan la procedencia ni el prompt final.
Se conservan ids, argumentos (incluidos fundamentos) y texto exacto de observaciones/valores normativos;
solo se omite la observación terminal. `ficha` se contrasta con la acción final, no se agrega como respuesta.
Las referencias son `<archivo fuente>#<id>`; no se inventan namespaces. IDs repetidos dentro de un archivo se
rechazan: combinar fuentes exige una decisión explícita de identidad, no concatenar salidas a ciegas.

Ejemplo **solo para una fuente ya curada y un prompt ya acordado** (no se ejecutó sobre los datos importados):

```bash
.venv/bin/python -m scripts.convertir_trazas \
  --entrada datos/trazas-curadas.jsonl --salida datos/conversaciones-curadas.jsonl \
  --origen historial --prompt configuraciones/prompts/baseline.json
```

`--prompt` (o el obsoleto `--sistema`, que lee UTF-8 sin modificar el texto) es obligatorio.
Esta CLI conserva todas las observaciones originales; el dataset de entrenamiento sale de
`scripts.generar_conversaciones` (abajo), que re-ejecuta búsquedas y revalida fichas. La CLI devuelve cantidad y SHA-256 de entrada/salida;
error de argumentos: código 2; datos/lectura/escritura: 1. No imprime conversaciones ni datos de pacientes.
Rechaza fuente vacía, JSON inválido, pasos inconsistentes, ficha distinta y conversaciones inválidas.
La publicación usa un temporal vecino validado y un enlace exclusivo: no reemplaza archivos ni symlinks,
no publica registros parciales. Requiere un filesystem local que soporte enlaces duros; no promete durabilidad
ante pérdida de energía. El conversor no clasifica historial, no regenera RAG ni prueba corrección normativa.

Validación por las interfaces públicas:

```bash
env PYTHONPATH=tests .venv/bin/python -m unittest test_conversion_trazas test_conversaciones test_preparacion test_tokenizacion -q
```

Evidencia CPU: 25 tests, 24 pasan y 1 opcional omitido. Suite completa:
`env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
116 tests, 115 pasan y 1 omitido. Red→green por unidad:
API ausente → conservación; inválidos aceptados → validación; API archivo ausente → determinismo;
archivo vacío aceptado → rechazo integral; `False == 0` aceptado → ficha con tipos exactos;
CLI ausente → contexto obligatorio y publicación sin overwrite. Sin mocks internos.

Las 800 + 2.000 trazas se recorrieron con `convertir_traza`, `json.loads` y el system actual, **solo en memoria**:
2.800 conversiones estructurales válidas. `origen=sintetico` para referencia fue una hipótesis exploratoria,
no una clasificación acordada; no se guardó un dataset final ni se entrenó. Ver [auditoría](auditoria-trazas.md).

## Dataset v2: `scripts.generar_conversaciones` (issue #8)

Un solo comando regenera el dataset de entrenamiento en una carpeta **nueva** (si existe, sale con código 1
sin tocarla). `datos/entrenamiento/` queda como la versión anterior de Yas, solo como historia.

```bash
uv run --locked --no-sync python -m scripts.generar_conversaciones
# equivalente explícito:
uv run --locked --no-sync python -m scripts.generar_conversaciones \
  --prompt configuraciones/prompts/baseline.json --semilla 12012 --candidatos 5000 \
  --salida datos/conversaciones/v2-baseline
# otra variante del buscador (la decisión vigente es baseline; ver abajo):
uv run --locked --no-sync python -m scripts.generar_conversaciones --buscador-variante prefijos
```

Sin `--buscador-variante` el buscador se arma con `entrega.construir_buscador`, igual que en inferencia
(`BUSCADOR_VARIANTE` o `baseline`), y la salida por defecto es `datos/conversaciones/v2-<variante>`.
Salidas: `conversaciones.jsonl` (contrato v1), `auditoria.jsonl` (por conversación: valores de parámetros,
supuestos, secciones rectoras/recuperadas y faltantes) y `manifiesto.json` (hashes de entradas y salida,
commit y `dirty`, prompt con sha256, variante y configuración del buscador, semilla, conteos y exclusiones
por fuente y motivo, cobertura de evidencia, distribución de parámetros, largos en **caracteres**).

**Fuentes.** `referencia`: las 800 trazas. `historial`: las filas que conserva `seleccionar_ids` de Yas
(solo `correcta`). `sintetico`: los 500 casos que elige `seleccionar_casos` de `admision.generador`
(semilla 12012, 5.000 candidatos), con valores de parámetros sorteados (misma semilla) entre los que
toman en la referencia; P-COB-1 y las secciones de otras áreas quedan como en `normas/original`.

**Búsquedas.** Cada `buscar_normativa` se re-ejecuta con el buscador real sobre una copia temporal de
`normas/original` cuyo `parametros.yaml` tiene los valores de esa conversación, y la observación sale de
`admision.entorno.Entorno.paso` (mismo texto, `k_entorno` y errores que en operación). Las consultas de
las trazas se conservan; leer_triage/preguntar/verificar_afiliacion conservan su observación original.

**Valores de cada traza.** Se leen de sus propias líneas `Valores: P-XXX-N = ...`; si una traza se
contradice, se excluye. Un parámetro no observado toma el valor original solo si la ficha registrada
sigue sin violar normas con **toda** combinación de valores conocidos (los de la referencia y el
original) de los no observados; si no, la traza se excluye (`parametro_relevante_no_observado`).
En historial se exige además que todo valor observado sea el original.

**Revalidación.** Cada traza se reconstruye como fila del historial (`admision_tp.trazas_normativas`;
coincide con `historial.csv` en las 2.000 filas) y se exige `razones_normativas(fila, valores) == []`
más las dos costumbres (`aviso_segun_costumbre`, `sigue_costumbre_credencial`). Los sintéticos usan
`deducir_ficha` con los valores de su carpeta y el entorno del kit valida la ficha.

**Práctica.** Se excluye toda conversación cuyo triage (sin la línea del id) sea idéntico al de un caso de
`practica/casos.jsonl`, o que comparta número de documento, afiliado o CUIL. No se leen esperadas.

**Variante del buscador: `baseline` (decisión del 2026-11-10).** En la comparación con el agente en Kaggle
(50 casos de práctica, normas originales) `baseline` resolvió 10/50 y `prefijos` 6/50, aunque `prefijos`
rinda mejor en `evaluar_rag`. La variante sigue siendo un argumento para poder regenerar con otra.

**Curado de trazas reales (transformaciones explícitas, registradas en `auditoria.jsonl` y contadas en
`estadisticas.<origen>.curado` del manifiesto).** Motivo: el modelo tiene que aprender a responder con la
evidencia que recuperó, no de memoria; una demostración que decide o cita una sección que no vio enseña
lo contrario.

1. *Búsqueda de ART agregada.* Si la ficha depende de la ART (NOR-COB-02 rectora) y ninguna búsqueda la
   recupera, por ejemplo con la consulta combinada «accidente de trabajo ART dolor torácico plazo», se
   inserta un paso `buscar_normativa` con «accidente de trabajo ART» justo después de esa consulta
   combinada. Las consultas originales no se modifican. La observación sale del buscador real con los
   valores de la conversación, y se verifica que traiga NOR-COB-02: si no la trae, o si no hay una consulta
   de ART después de la cual insertarla, la traza se excluye.
2. *Fundamentos filtrados.* De `fundamentos` de `registrar_decision` se quitan las secciones que ninguna
   búsqueda de la conversación recuperó. Si no queda ninguna, se quita la clave, que es optativa.
   Los ocho campos de la ficha no cambian.

**Cobertura de evidencia (se mide y se informa; no se filtra por ella).** Secciones rectoras de un caso: NOR-ADM-01 siempre;
NOR-ADM-02 si la edad alcanza P-ADM-2; NOR-ADM-03 con pasaporte o cédula; NOR-COB-02 si la cobertura es la
ART; NOR-COB-04 si el número verificado (no ART) termina en P-COB-2; la sección del circuito y NOR-TDE-04
si llegó en ambulancia con circuito. También se mide si cada sección citada en `fundamentos` aparece en
alguna búsqueda. El guion sintético busca cada sección rectora y cada fundamento que cita.

**Dataset v3 con el prompt elegido en #13 (2026-11-11).** La consigna pide entrenar con el prompt optimizado;
`procedimiento-v1` se eligió en el conjunto de desarrollo ([optimización de prompt](optimizacion-prompt.md#decisión)):

```bash
uv run --locked --no-sync python -m scripts.generar_conversaciones \
  --prompt configuraciones/prompts/procedimiento-v1.json --salida datos/conversaciones/v3-procedimiento
```

~20 s de CPU; 2.900/2.900 válidas, commit `b59239d` con `dirty=false`, regeneración byte a byte idéntica.
Comparado con `v2-baseline`: las 2.900 conversaciones cambian **solo el mensaje de sistema**; el resto de los
mensajes, los metadatos y `auditoria.jsonl` son idénticos. v2 queda sin tocar. **Qué dataset se entrena lo
decide #15 (Fran)**: con v3 la fila `ambos` usa el mismo prompt del entrenamiento y `solo_entrenamiento` debería
usar también `procedimiento-v1` o declararse como cambio de prompt. Las demostraciones no se reescribieron para
seguir el procedimiento al pie de la letra: conservan sus búsquedas (3,1 por conversación en referencia e
historial, 4,9 en sintéticas), cerca de las 3,5 que hace el modelo sin entrenar con este prompt.

**Límites.** No se tokenizó (requiere descargar el tokenizador real): los largos son caracteres.
La identidad de contaminación es exacta (triage/números), no detecta paráfrasis.
Resultados de la versión generada: [bitácora](progreso.md) e [INFORME](../INFORME.md#parte-1-conversaciones).
