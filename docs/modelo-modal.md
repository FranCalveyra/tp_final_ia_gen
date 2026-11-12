# Baseline: modelo local y Modal

Estado: implementación probada con dobles en CPU. En el intento autorizado del
01/11 se construyó la imagen (77,28 s), pero Modal rechazó la A100 por falta de
medio de pago. **No se descargaron pesos ni se ejecutó una GPU.** El diagnóstico
está en `corridas/smoke-original-20261101/intento.json`. La compatibilidad real y
el acceso de HF todavía no están verificados.

- `entrega.construir_agente()` valida el `PromptSpec` (`PROMPT_SPEC` o `procedimiento-v1` desde 2026-11-12), carga el modelo una vez
  y arma `AgentePrompt` (desde el issue #7; antes `AgenteJSON`, mismos textos con el baseline).
- `ModeloLocal` recibe conversaciones y devuelve textos en el mismo orden.
- Gemma 4 E4B sin entrenar, commit HF inmutable en `configuracion.py`.
- Transformers genera sin muestreo, bf16 y padding izquierdo. La atención no se fuerza:
  SDPA en el texto y eager en las torres que no lo soportan (forzar SDPA rompe el audio de Gemma 4). No se
  recorta contexto; al superar el presupuesto falla explícitamente.
- Micro-lotes de 1 por defecto para limitar VRAM. El adaptador admite un tamaño
  configurable con `MODELO_LOTE`. En Kaggle T4 (fp16, no comparable con A100) lote 4 fue 2,1x más rápido que 1
  sin cambiar resultados; 8 y 16 fallaron con error CUDA (ver [progreso](progreso.md), 2026-11-04). En A100 sin medir.
- El coordinador corta tras 3 respuestas sin JSON seguidas o al llegar al tope de 40 llamadas del kit
  (ya no a los 16 turnos). Antes de cada llamada mide cada conversación con `ModeloLocal.medir_contexto`
  y termina sin decisión solo el caso que no entra; los demás siguen. Las herramientas no se ejecutan
  dentro del adaptador. Ver [optimización de prompt](optimizacion-prompt.md#contrato-de-formato-7--13).

## Adapter LoRA (probado con dobles)

Si `adapter_repo` está definido, `ModeloLocal.cargar` aplica el LoRA sobre la base
y lo fusiona en memoria (`merge_and_unload`), así que la inferencia no suma capas.

| Campo | Entorno | Regla |
|---|---|---|
| `adapter_repo` | `ADAPTER_REPO` | Vacío = base sola (baseline). |
| `adapter_revision` | `ADAPTER_REVISION` | Obligatorio con repo: commit HF de 40 caracteres. |
| `adapter_subcarpeta` | `ADAPTER_SUBCARPETA` | Default `final`; el trainer publica cada checkpoint en su carpeta. |

- Antes de aplicarlo se valida que `base_model_name_or_path` y `revision` del
  `adapter_config.json` coincidan con la base configurada. Si no, falla.
- La cátedra solo define `HF_TOKEN`, así que el adapter final se fija en los
  defaults de `configuracion.py`. El entorno sirve solo para experimentar.
- PEFT forma parte del grupo `inferencia`. Todavía no se probó con pesos reales
  ni en GPU.

## Métricas por lote (probado con dobles)

Con `MODELO_METRICAS=<ruta>`, `ModeloLocal` agrega una línea JSONL por micro-lote:
`conversaciones`, `ancho_entrada` (con padding), `tokens_entrada` (sin padding),
`tokens_generados`, `segundos` (de `generate` a decodificar) y la VRAM pico
asignada y reservada de ese lote (`reset_peak_memory_stats` antes de cada uno).

- Sin la variable no mide ni toca `torch.cuda`. La cátedra no la define.
- `infra/modal_app.py` la define y devuelve `metricas.jsonl` junto a las trazas.
  Con `tiempos.json` del kit alcanza para el antes/después del #4 y los recursos del #5.
- La VRAM es la de PyTorch en ese proceso, no la de todo el dispositivo. No se midió todavía.

## Pruebas sin GPU ni build

```bash
uv run --no-project --with pyyaml==6.0.3 python -m unittest discover -s tests -v
```

## Próxima corrida (requiere nueva autorización)

Requiere autenticación Modal, secreto `huggingface` con `HF_TOKEN` y un medio de
pago habilitado en Modal para usar GPU. El intento previo no procesó pacientes.
**El comando siguiente construye la imagen y consume GPU: no ejecutarlo sin
autorizar ambas acciones.** Usa A100 de 40 GB y timeout de 15 minutos.

```bash
uvx --from modal==1.6.3 modal run infra/modal_app.py --cantidad 1 --variante original --salida corridas/smoke-original
```

Solo se envían los casos seleccionados y las normas a Modal, nunca las fichas
esperadas. Las trazas, tiempos y configuración se devuelven y guardan localmente.
No se sobreescriben carpetas existentes. Para puntuar después, sin GPU:

```bash
uv run --no-project --with pyyaml==6.0.3 python -m admision.correr puntuar --trazas corridas/smoke-original/trazas.jsonl --esperadas practica/esperadas-original.jsonl --salida corridas/smoke-original
```

Las dependencias directas están fijadas en `requirements-inferencia.txt`; todavía
falta un lock transitivo para la entrega final. No se instaló el proyecto como
paquete ni se modificó el kit de la cátedra.

Fuentes verificadas: [Gemma4 en Transformers](https://huggingface.co/docs/transformers/model_doc/gemma4)
(ejemplo text-only con AutoModelForCausalLM + AutoProcessor),
[GPU de Modal](https://modal.com/docs/guide/gpu),
[secrets](https://modal.com/docs/guide/secrets).
