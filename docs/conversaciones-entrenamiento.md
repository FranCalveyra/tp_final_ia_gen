# Conversaciones de entrenamiento — propuesta v1

Contrato mínimo propuesto para coordinar con Eze/Yas; todavía no fue acordado con ellos.
El conversor produce **JSONL UTF-8: un objeto por línea**, sin líneas vacías.

## Qué entregar

Cada objeto contiene exactamente:

- `version`: entero `1`.
- `id`: texto no vacío, único en el archivo.
- `origen`: `historial` o `sintetico`.
- `referencias`: lista no vacía de textos que identifican filas fuente o script/configuración/semilla sintéticos.
- `messages`: conversación completa en el formato actual del agente: objetos `role`, `content`, ambos textos.

Se conserva el contenido original; no se normalizan ni reescriben mensajes.
La conversación comienza `system`, `user`; luego alterna `assistant`, `user` y termina en `assistant`.
Cada respuesta del asistente es **JSON estricto** con `herramienta` y `argumentos`.
La última acción debe ser `registrar_decision`, sin observación posterior. Máximo 40 acciones.
Las observaciones son `Resultado de <herramienta>:\n<texto no vacío>`, como en `AgenteJSON`.

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

- `convertir_traza(traza, *, origen, sistema, referencia)` devuelve una conversación v1.
- `convertir_archivo(entrada, salida, *, origen, sistema)` valida todo el JSONL antes de publicar una salida nueva.

Ambas requieren **origen y prompt explícitos**: no adivinan la procedencia de las 800 trazas ni el prompt final.
Se conservan ids, argumentos (incluidos fundamentos) y texto exacto de observaciones/valores normativos;
solo se omite la observación terminal. `ficha` se contrasta con la acción final, no se agrega como respuesta.
Las referencias son `<archivo fuente>#<id>`; no se inventan namespaces. IDs repetidos dentro de un archivo se
rechazan: combinar fuentes exige una decisión explícita de identidad, no concatenar salidas a ciegas.

Ejemplo **solo para una fuente ya curada y un prompt ya acordado** (no se ejecutó sobre los datos importados):

```bash
rtk proxy .venv/bin/python -m scripts.convertir_trazas \
  --entrada datos/trazas-curadas.jsonl --salida datos/conversaciones-curadas.jsonl \
  --origen historial --sistema datos/prompt-final.txt
```

`--sistema` lee UTF-8 sin modificar el texto. La CLI devuelve cantidad y SHA-256 de entrada/salida;
error de argumentos: código 2; datos/lectura/escritura: 1. No imprime conversaciones ni datos de pacientes.
Rechaza fuente vacía, JSON inválido, pasos inconsistentes, ficha distinta y conversaciones inválidas.
La publicación usa un temporal vecino validado y un enlace exclusivo: no reemplaza archivos ni symlinks,
no publica registros parciales. Requiere un filesystem local que soporte enlaces duros; no promete durabilidad
ante pérdida de energía. El conversor no clasifica historial, no regenera RAG ni prueba corrección normativa.

Validación por las interfaces públicas:

```bash
rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest test_conversion_trazas test_conversaciones test_preparacion test_tokenizacion -q
```

Evidencia CPU: 25 tests, 24 pasan y 1 opcional omitido. Suite completa:
`rtk proxy env PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -q`:
116 tests, 115 pasan y 1 omitido. Red→green por unidad:
API ausente → conservación; inválidos aceptados → validación; API archivo ausente → determinismo;
archivo vacío aceptado → rechazo integral; `False == 0` aceptado → ficha con tipos exactos;
CLI ausente → contexto obligatorio y publicación sin overwrite. Sin mocks internos.

Las 800 + 2.000 trazas se recorrieron con `convertir_traza`, `json.loads` y el system actual, **solo en memoria**:
2.800 conversiones estructurales válidas. `origen=sintetico` para referencia fue una hipótesis exploratoria,
no una clasificación acordada; no se guardó un dataset final ni se entrenó. Ver [auditoría](auditoria-trazas.md).
