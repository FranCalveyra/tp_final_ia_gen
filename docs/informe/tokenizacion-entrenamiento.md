# Tokenización SFT (sin entrenamiento)

Convierte una preparación validada a `input_ids`, `attention_mask` y `labels`.
Las etiquetas de contexto (system, user, resultados de herramientas y encabezados)
son `-100`: no contribuyen a la pérdida. Se supervisan todas las respuestas
assistant, incluido el cierre del turno. No hay padding ni empaquetado todavía;
el collator de entrenamiento implementado pone `-100` en las etiquetas del padding.

## Uso futuro con el tokenizador real

En macOS o Linux, el grupo `tokenizacion` contiene Transformers y Jinja2, sin
PyTorch ni dependencias de inferencia GPU. Jinja2 es explícita: Transformers solo
no alcanza para renderizar el formato de conversación.

```bash
uv sync --locked --no-build --group tokenizacion
```

Si también necesitás Modal o tracking en ese entorno, agregá sus `--group` al
sync: uv elimina grupos no solicitados. La validación aquí se hizo en un entorno
aisladamente temporal y no modificó `.venv`.

Después:

```bash
uv run --locked --no-sync python -m scripts.tokenizar_entrenamiento \
  --preparacion corridas/sft-preparacion-001 \
  --salida corridas/sft-tokens-001
```

Solo descarga archivos del tokenizador, nunca pesos del modelo; puede requerir
acceso HF. La salida debe ser nueva. No entrena ni publica datos. La aplicación
no carga `.env` automáticamente. No uses estos ejemplos artificiales para entrenar.

Primero comprueba hashes de conversaciones/configuración contra el manifiesto.
Guarda `tokens.jsonl` y `manifiesto-tokens.json`, con hashes de entrada/salida,
modelo/revisión, plantilla, estado Git y conteos. No modifica los snapshots.
Si el código está sucio, el commit no basta para reconstruirlo: commitear y volver
a preparar antes de una corrida auditable.

## Por qué no pedimos una máscara nativa

Inspeccionamos el [template de Gemma en la revisión fijada](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/chat_template.jinja):
no tiene bloques `generation`. Las máscaras nativas de Transformers requieren
esos bloques. Usamos la misma plantilla, `enable_thinking=False`, y prefijos
con encabezado de generación para ubicar el inicio de cada respuesta; el prefijo
con respuesta completa determina su final. Se comprueba igualdad exacta de tokens
contra la conversación completa. Si los prefijos cambian, falla: no adivina spans.
No se recortan conversaciones largas ni se buscan respuestas por substring.

[Formato de chat y entrenamiento en Transformers](https://huggingface.co/docs/transformers/main/chat_templating).

## Verificación y límites

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests CPU con tokenizador artificial: múltiples respuestas, cierre, contexto
ignorado, prefijos inseguros, padding inesperado, longitud y hashes alterados.
## Validación real (CPU, 2026-11-01)

Se ejecutó AutoTokenizer real en revisión fijada, con Transformers 5.22.1,
Tokenizers 0.23.3 y Hugging Face Hub 2.4.0. Solo se descargaron cuatro archivos:
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` y `config.json`:
32.196.422 bytes (~32,2 MB), sin pesos ni PyTorch. El template tiene SHA-256
`0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`.

La conversación artificial de dos respuestas produce 167 tokens, de los cuales
126 están supervisados. La decodificación de etiquetas coincide exactamente con
cada JSON assistant seguido de `<turn|>\n`, sin encabezados ni contexto. También
pasaron rechazo por longitud, preservación de snapshots, generación del artefacto
y CLI real con carga del tokenizador desde caché, offline, sin inyección.

Resultado local ignorado: `corridas/tokenizador-real-20261101/resultado.json`.
Para repetir el test real sin descargar nada, indicá el snapshot local:

```bash
TP_TOKENIZADOR_LOCAL=/ruta/al/snapshot uv run --locked --no-sync python -m unittest discover -s tests -v
```

Sin esa variable, el test real se omite y los tests artificiales siguen activos.
Esto valida el formato de ese ejemplo, no todas las conversaciones futuras.
No hay prueba de entrenamiento, VRAM ni calidad del modelo. La tokenización falla
cerrada si otra conversación o tokenizador produce prefijos incompatibles.
