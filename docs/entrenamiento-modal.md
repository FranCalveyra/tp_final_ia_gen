# Entrenamiento durable en Modal — todavía sin ejecución GPU

El runner conserva entradas, manifiestos, checkpoints y salida final en el Volume v2
`tp-admision-entrenamiento`. No publica en HF: usa solamente los secretos existentes
`huggingface` (lectura del modelo base) y `wandb-secret` (W&B, únicamente si `--online`).

## Comando futuro (NO ejecutado)

```bash
uv run --locked --no-build --group modal modal run infra/entrenamiento_app.py \
  --preparacion corridas/preparada --tokens corridas/tokens --nombre corrida-001
```

**Construye imagen, crea recursos persistentes y consume A100-40GB. Requiere autorización.**
El timeout es 3600 s, sin reintentos automáticos, máximo un contenedor por función.
Se configura con `--limite-segundos` entre 60 y 14400; revisar costo antes de ampliarlo.
El flag `--online` autoriza registrar la corrida en W&B; por defecto queda offline.

Se validan localmente solo cinco archivos allowlisted (64 MiB por archivo, 128 MiB
en total), y nuevamente en remoto. No se transfieren rutas de usuario, `.git`, `.env`
ni datos de práctica. Es necesario disponer antes de conversaciones curadas y prompt
optimizado del equipo; los tests con ejemplos sintéticos no los reemplazan.

La imagen lleva los fuentes Python, `pyproject.toml`, `uv.lock` y el export congelado
de entrenamiento. Se comprueba su hash contra la procedencia enviada: el contenedor
no necesita `.git`. El commit y `dirty` se conservan; fuentes sin commitear siguen sin
ser reconstruibles desde el commit solo, aunque se registre su hash.

## Persistencia y reanudación

- Recarga el Volume antes de leer una corrida.
- Hace commit tras copiar entradas, tras cerrar/escribir integridad de cada checkpoint,
  después del adapter final y al registrar éxito o error.
- Un corte abrupto recupera como máximo el último checkpoint completamente escrito
  y confirmado. Los pasos posteriores pueden perderse; no hay promesa de recuperación
  de un checkpoint parcial ni de persistencia instantánea durante una terminación forzada.
- Usa `Dict.put(skip_if_exists=True)` en `tp-admision-entrenamiento-locks` para impedir
  dos ejecuciones del mismo nombre, incluso desde apps distintas.

```bash
uv run --locked --no-build --group modal modal run infra/entrenamiento_app.py \
  --preparacion corridas/preparada --tokens corridas/tokens --nombre corrida-001 \
  --checkpoint checkpoint-50
```

Requiere exactamente los mismos bytes, código, configuración y modo online. No hay
selección automática del checkpoint: inspeccionar y elegir uno completo. Las salidas
existentes no se sobreescriben. Un lock que queda por un cierre abrupto requiere
comprobar manualmente que no existe una ejecución activa antes de retirarlo; el programa
no lo limpia. Los Dict de Modal tienen expiración propia de 7 días sin actividad.

## Qué está probado y qué falta

Los tests CPU con dobles verifican copia exacta, hashes, exclusión, recarga, commits,
diagnóstico y reanudación. **No demuestran persistencia real de Modal ni entrenamiento.**
Faltan prueba GPU y corte/reanudación reales, token HF de escritura y runner de
publicación opt-in con secreto separado (`hf-write`, aún no creado), carga del adapter
en inferencia y acceso de la cátedra. No agregar el token de escritura al agente.

Regenerar dependencias sin construir:

```bash
uv export --locked --no-default-groups --only-group entrenamiento \
  --no-emit-project --no-hashes --output-file requirements-entrenamiento.txt
```

Fuentes: [Volumes](https://modal.com/docs/guide/volumes),
[Dict](https://modal.com/docs/sdk/py/latest/Dict).
