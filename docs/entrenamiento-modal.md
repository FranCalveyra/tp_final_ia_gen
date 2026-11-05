# Entrenamiento durable en Modal — todavía sin ejecución GPU

El runner conserva entradas, manifiestos, checkpoints y salida final en el Volume v2
`tp-admision-entrenamiento`. Por defecto no publica en HF: usa solamente los secretos
existentes `huggingface` (lectura del modelo base) y `wandb-secret` (W&B, únicamente si `--online`).
La publicación es opt-in (ver [Publicación HF](#publicación-hf-opt-in-probada-con-dobles)).

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

## Publicación HF (opt-in, probada con dobles)

```bash
uv run --locked --no-build --group modal modal run infra/entrenamiento_app.py \
  --preparacion corridas/preparada --tokens corridas/tokens --nombre corrida-001 \
  --online --publicar
```

- `--publicar` exige `--online` y usa `ejecutar_publicando`, que adjunta el secreto
  `hf-write` (**nombre propuesto, no creado ni verificado**; debe exponer `HF_TOKEN` de escritura)
  **en lugar de** `huggingface`. Así hay un solo `HF_TOKEN` y no dependemos del orden de los secretos.
  El mismo token descarga la base: su usuario debe haber aceptado la licencia de Gemma.
  Sin `--publicar` no se necesita.
- El repo sale de `hf_repo` de la configuración (obligatorio en el contrato) y debe existir y ser
  privado; el entrenador lo comprueba antes de cada subida y guarda `publicacion-hf.json` por
  checkpoint y `publicaciones.json` en la salida, con commit al Volume después de cada recibo.
- Al reanudar hay que repetir `--online --publicar`: cambiar el modo se rechaza. Los recibos
  previos se releen y no se vuelven a subir los checkpoints viejos.

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

El trainer compartido sí tiene humo y reanudación HF reales en Kaggle T4; ver la
[matriz de evidencia y handoff](entrenamiento-sft.md). Esa prueba no valida el Volume, locks ni GPU A100 de Modal.

El entrypoint usa checkpoints cada 50 pasos: el humo habitual de 8 pasos no garantiza
un checkpoint reanudable. Planificar suficientes pasos y verificar un checkpoint completo; `final/` no basta.

Los tests CPU con dobles verifican copia exacta, hashes, exclusión, recarga, commits,
diagnóstico y reanudación. **No demuestran persistencia real de Modal ni entrenamiento.**
La publicación quedó **probada con dobles** (el modo llega al entrenador y los recibos se
conservan al reanudar); la definición Modal de `ejecutar_publicando` no se importa en tests.
Faltan prueba GPU y corte/reanudación reales **en Modal/A100**, el secreto `hf-write` de Modal y una subida desde esa ruta,
y acceso de la cátedra. No agregar el token de escritura al agente.

Regenerar dependencias sin construir:

```bash
uv export --locked --no-default-groups --only-group entrenamiento \
  --no-emit-project --no-hashes --output-file requirements-entrenamiento.txt
```

Fuentes: [Volumes](https://modal.com/docs/guide/volumes),
[Dict](https://modal.com/docs/sdk/py/latest/Dict).
