# Entrenador SFT LoRA (implementado, todavía no validado en GPU)

Entrena únicamente las respuestas marcadas por la tokenización. Usa el modelo base
con revisión fija y LoRA `q_proj/v_proj`, rango 8, alpha 16, dropout 0. Son parámetros
iniciales, **no una demostración de que Gemma entre en A100 de 40 GB**.

## Qué está probado

Tests CPU con dobles comprueban snapshots, padding/etiquetas, diagnóstico, rechazo de
salidas existentes, reanudación y publicación privada. No ejecutan Trainer ni PEFT.
El tokenizador real se validó por separado. No hay resultados de entrenamiento aún.

## Ejecutar (requiere autorización y una máquina Linux con CUDA)

```bash
uv sync --locked --no-build --group entrenamiento
uv run --locked --no-sync python -m scripts.entrenar \
  --preparacion corridas/preparada --tokens corridas/tokens \
  --salida corridas/entrenamiento-001
```

**Este comando carga pesos y entrena. No es una prueba inocua.** Por defecto W&B
es offline y no publica en HF. Cada salida conserva las entradas exactas, hashes,
configuración, política de checkpoints, LoRA, revisión base y estado del código.
Si `dirty=true`, el commit no basta para reconstruir el código: commitear y volver
a preparar antes de una corrida que se reportará en el informe.

Con `--online` registra métricas en W&B. Con `--online --publicar` publica cada
checkpoint en el repositorio HF existente **privado**, guarda su commit inmutable
en `publicaciones.json` y lo registra en W&B. No crea ni cambia la privacidad del repo.
Un fallo de publicación falla la corrida; no se oculta como entrenamiento exitoso.

Necesita `WANDB_API_KEY` y, para publicar, un token HF con escritura en ese repo.
**El token Read usado para inferencia no sirve para subir checkpoints.** Preparar un
secreto de escritura separado del secreto `huggingface`; nunca pegarlo en código,
manifiestos ni chat. El secreto W&B existente se llama `wandb-secret`.

## Precisión y dispositivos (T4, desarrollo)

Los defaults no cambian: `--precision bf16 --dispositivos unico --guardar-cada 50` (A100 de 40 GB).
Para T4 x2: `--precision fp16 --dispositivos auto` (base en fp16, adapter fp32, `GradScaler`; el modelo se
reparte entre GPU con `device_map="auto"`). `--guardar-cada` fija los pasos entre checkpoints. Los tres valores
quedan en `corrida.json` (`ejecucion`) y en la política de la identidad: reanudar con otros valores falla, y en
fp16 el checkpoint debe traer `scaler.pt`. Una auditoría anterior sin `ejecucion` se lee como los defaults.
Solo probado con dobles. Ver [kaggle.md](kaggle.md#entrenamiento-sft-humo-en-kaggle).

## Reanudar

```bash
uv run --locked --no-sync python -m scripts.entrenar \
  --preparacion corridas/preparada --tokens corridas/tokens \
  --salida corridas/entrenamiento-001 \
  --reanudar corridas/entrenamiento-001/checkpoint-50
```

Se conserva el ID de corrida. Deben coincidir datos, configuración, LoRA, lock y
modo online/publicación. Cada checkpoint necesita adapter, optimizador, scheduler,
RNG, Trainer state y manifiestos íntegros. Un adapter `final/` no basta para reanudar.
En online W&B exige la corrida existente; offline produce segmentos locales que
todavía deben sincronizarse/conciliarse. Los checkpoints se guardan cada 50 pasos y
al cierre según Trainer. No se eliminan automáticamente.

## Pendiente para cerrar el issue #3

- Prueba real de carga, backward y reanudación con Gemma en A100 de 40 GB.
- Validar el [runner durable de Modal](entrenamiento-modal.md), ya implementado con
  commits por checkpoint pero todavía probado solo con dobles CPU.
- Datos curados reales y prompt optimizado del equipo antes de entrenar.
- Token HF de escritura y publicación real auditada; acceso de la cátedra.
- Cargar el adapter LoRA en inferencia o publicar un modelo fusionado. El agente
  actual **todavía no carga automáticamente** el adapter de esta salida.

Fuentes: [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer),
[LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).
