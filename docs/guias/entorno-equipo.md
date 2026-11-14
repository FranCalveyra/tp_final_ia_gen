# Preparar el entorno del equipo

> **Actualización 2026-11-05:** Modal descartado definitivamente. Sus comandos/Secrets de esta guía son
> legacy y no deben ejecutarse. Usar [Kaggle](kaggle.md) para desarrollo; validación final A100-40GB
> por vía alternativa todavía a determinar. No se borraron dependencias ni código histórico.


Desde la raíz del checkout, Python 3.12 o superior y uv instalado. Las pruebas y el buscador no necesitan cuentas, tokens ni GPU.

## Camino CPU (sin builds)

```bash
uv sync --locked --no-build
uv run --locked --no-sync python -m unittest discover -s tests -v
uv run --locked --no-sync python -m admision.evaluar_rag --entrega entrega.py --normas normas/original --consultas rag/consultas_dev.jsonl
```

`uv sync` incluye por defecto el grupo `inferencia` (contrato de la cátedra), pero sus paquetes
solo se instalan en Linux: en macOS/Windows queda PyYAML. Los tests simulan el modelo: no demuestran calidad de Gemma.
`tool.uv.package = false` evita construir/instalar el proyecto local; ejecutamos desde el checkout. Se conserva la configuración setuptools del kit, pero no se usa en este flujo.

## Configuración sin secretos en Git

`.env.example` documenta las variables. **No se carga automáticamente**: exportá únicamente las necesarias en tu terminal o configurá los Secrets propios de Kaggle. No pegues tokens en comandos que queden en el historial ni en el chat. `.env` y `.env.*` están ignorados salvo la plantilla.

| Variable | Uso |
|---|---|
| `HF_TOKEN` | Descargar modelos; en Modal, Secret `huggingface` |
| `WANDB_API_KEY` | Secret `wandb-secret`; prueba de instrumentación separada del agente |
| `WANDB_ENTITY`, `WANDB_PROJECT` | Destino de experimentos; ver [W&B](wandb.md) |
| `MODELO_ID`, `MODELO_REVISION` | Modelo y commit HF exacto |
| `MODELO_LOTE` | Micro-lote, por defecto 1 |
| `MODELO_METRICAS` | Opcional: JSONL de tokens, segundos y VRAM por lote; ver [modelo](../historico/modelo-modal.md#métricas-por-lote-probado-con-dobles) |
| `ADAPTER_REPO`, `ADAPTER_REVISION`, `ADAPTER_SUBCARPETA` | Adapter LoRA opcional; ver [modelo](../historico/modelo-modal.md#adapter-lora-probado-con-dobles) |
| `BUSCADOR_VARIANTE` | `baseline` o candidato `sin_comunes` |

## Modal e inferencia — LEGACY, NO USAR

Cada integrante usa su propia cuenta y créditos. Instalar el CLI no ejecuta GPU:

```bash
uv sync --locked --no-build --group modal
uv run --locked --no-sync modal setup
```

Creá en el dashboard un Secret `huggingface` con `HF_TOKEN`. No alcanza para comprobar permisos de descarga. Modal exige medio de pago para usar GPU; revisá saldo y presupuesto antes de correr.

El grupo `inferencia` solo instala sus dependencias en Linux. **En macOS/Windows no habilita inferencia**. Para un entorno Linux CUDA ya preparado:

```bash
uv sync --locked --no-build --group inferencia
```

No ejecutamos este último comando ni validamos CUDA en esta entrega. El runner Modal consume `requirements-inferencia.txt`, exportado desde el mismo lock. No editar ese archivo a mano.

## Actualizar dependencias deliberadamente

Editar los pines de `pyproject.toml`, luego:

```bash
uv lock --no-build
uv export --locked --no-dev --group inferencia --no-emit-project --no-hashes --output-file requirements-inferencia.txt
uv lock --check --no-build
```

`uv.lock` fija las versiones transitivas y hashes. El export fija versiones, pero omite hashes para el instalador de Modal. No comprueba por sí solo compatibilidad CUDA ni ausencia de vulnerabilidades.

## Pendientes del issue de infraestructura

- Repositorio privado HF compartido con la cátedra.
- Proyecto W&B compartido; integración de entrenamiento.
- Descarga real del modelo y prueba GPU (bloqueadas por medio de pago).

Esta guía no crea cuentas, repositorios ni proyectos externos. Ver [Kaggle](kaggle.md) para la ruta actual; [ejecución del modelo](../historico/modelo-modal.md) conserva referencias históricas a Modal.
