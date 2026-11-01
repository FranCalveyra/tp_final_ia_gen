# Instrumentación de experimentos

Destino: `francalveyra-ia-gen/tp-admision` (W&B, interfaz CoreWeave Forge).
El secreto de Modal se llama **`wandb-secret`**, con variable **`WANDB_API_KEY`**.
No compartas su valor, no lo pongas en argumentos ni lo subas al repositorio.
La aplicación no carga `.env` automáticamente.

## Prueba local, sin conexión

```bash
uv sync --locked --no-build --group tracking
uv run --locked --no-sync python -m scripts.probar_wandb
```

El modo predeterminado es `offline`: crea una corrida local bajo `corridas/`,
registra `setup_check=1` y la cierra. No entrena, no usa GPU ni verifica la clave
o permisos remotos. No ejecutar `wandb sync` sin decidir publicar esos archivos.
`WANDB_ENTITY` y `WANDB_PROJECT` permiten cambiar el destino.

## Prueba remota futura: requiere autorización

```bash
uv sync --locked --no-build --group modal
uv run --locked --no-sync modal run infra/wandb_app.py
```

**No ejecutada durante esta implementación.** Construye una imagen, consume CPU
de Modal y publica una corrida técnica real. Usa el secreto existente, sin copiar
la clave a tu terminal. No solicita GPU. Esta prueba verificará la credencial y
el acceso al proyecto; hasta entonces ambos siguen pendientes de validación.

No agregamos tracking al agente en operación: la consigna prohíbe servicios
externos. La instrumentación queda separada para desarrollo/entrenamiento.
La URL de Forge es de interfaz, no reemplaza la URL del API del SDK.

## Dependencias

El grupo `tracking` es opcional: el entorno CPU de tests/búsqueda sigue liviano.
El archivo para Modal se genera desde el mismo lock:

```bash
uv export --locked --no-default-groups --only-group tracking --no-emit-project \
  --no-hashes --output-file requirements-tracking.txt
```

Referencia: [SDK oficial de W&B](https://github.com/wandb/wandb).
