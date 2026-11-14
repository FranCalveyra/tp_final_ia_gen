# Integración sin GPU

```bash
uv run --no-project --with pyyaml python -m unittest discover -s tests -v
```

El comando no construye el proyecto. Requiere Python y PyYAML; si ya están
disponibles, también sirve `python -m unittest discover -s tests -v`.

`tests/test_integracion_cpu.py` reemplaza **solo** la carga del modelo por
respuestas guionadas. Usa la fábrica de `entrega.py`, el coordinador, el entorno,
el buscador y las funciones de ejecución/puntuación reales del kit.

Comprueba dos pacientes sintéticos, las cinco herramientas, el orden del lote,
un reintento por respuesta sin JSON, los nueve controles y la escritura/lectura de
trazas, tiempos, resultados y resumen. Las fichas esperadas se preparan después
de jugar, separadas de las entradas al modelo. Los archivos se crean en carpetas
temporales y se borran al terminar; no son corridas para el informe del modelo.

También simula un fallo de generación: el runner guarda el log de herramientas
y el diagnóstico, aunque descarta las conversaciones de ese lote ante excepción.

**Alcance:** valida conexiones y artefactos, no inteligencia, velocidad, VRAM,
compatibilidad real de Gemma ni precisión en pacientes. No usa Modal, credenciales,
pesos del modelo ni GPU. No existe un modelo simulado de respaldo en producción.
