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
