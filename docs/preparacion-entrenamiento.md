# Preparar una corrida SFT (sin entrenar)

Esta etapa congela qué datos y parámetros se pretende usar. No requiere GPU,
tokens ni conexión. El contrato de conversaciones sigue siendo una propuesta
que debe coordinarse con Eze/Yas; ver [formato v1](conversaciones-entrenamiento.md).

## Uso

1. Recibir el JSONL curado; no usar los casos de práctica o test para entrenar.
2. Copiar `configuraciones/sft-ejemplo.json` y revisar modelo, revisión, semilla,
   parámetros y destinos. Los valores son **ilustrativos**, no hiperparámetros
   validados ni una promesa de que el entrenamiento entre en memoria.
3. Crear la carpeta padre y elegir una salida que todavía no exista:

```bash
mkdir -p corridas
uv run --locked --no-sync python -m scripts.preparar_entrenamiento \
  --datos datos/conversaciones.jsonl \
  --configuracion configuraciones/sft-ejemplo.json \
  --salida corridas/sft-preparacion-001
```

El comando valida el dataset y crea:

| Archivo | Contenido |
|---|---|
| `conversaciones.jsonl` | Copia exacta de los bytes validados |
| `configuracion.json` | Configuración ordenada, sin credenciales |
| `manifiesto.json` | SHA-256 de ambas copias, conteos, configuración, commit Git y estado `dirty` |

El destino es exclusivo: si existe, el comando falla sin sobrescribirlo.
Los errores de validación no dejan una corrida parcial. El contenido del dataset
no aparece en el informe; la copia conserva las conversaciones y debe tratarse
con el mismo cuidado que los datos originales. `corridas/` está ignorada por Git.

## Qué falta para entrenar

Esto **no** carga el modelo, calcula tokens, aplica máscaras de pérdida,
entrena, retoma checkpoints ni sube archivos a HF/W&B. El máximo de secuencia es
una intención registrada: aquí no se verifica que las conversaciones entren.
Tampoco se verifica existencia/permisos de los destinos remotos o corrección de
las decisiones. La procedencia declarada no demuestra ausencia de contaminación.

Si `dirty=true`, el commit por sí solo no reconstruye el código usado: hay que
commitear los cambios y preparar otra corrida antes del entrenamiento auditado.
La semilla registrada tampoco garantiza por sí sola determinismo en GPU.

Pruebas CPU:

```bash
uv run --locked --no-sync python -m unittest discover -s tests -v
```
