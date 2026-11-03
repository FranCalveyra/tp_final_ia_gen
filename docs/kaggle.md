# Respaldo de desarrollo: Kaggle (T4 x2)

Estado: **probado con dobles** (tests del runner, config y métricas). No se ejecutó nada en Kaggle ni en GPU.
Es un respaldo temporal porque Modal exige una tarjeta que nadie del equipo puede cargar.
La entrega final sigue siendo A100 de 40 GB con los defaults (`bfloat16`, `cuda:0`), que no cambiaron.

**Límites:** resultados, tiempos y VRAM de Kaggle **no son comparables** con A100 y se reportan aparte.
La cuota es de unas 30 h/semana de GPU (aproximada, la fija Kaggle) y las sesiones son cortas.

## Variables nuevas (solo desarrollo)

| Variable | Default | Valores |
|---|---|---|
| `MODELO_DTYPE` | `bfloat16` | `bfloat16`, `float16`, `float32` |
| `MODELO_DEVICE_MAP` | `cuda:0` | `auto`, `cuda`, `cuda:N` |

En T4 (sm_75) no hay bf16 nativo: el runner usa `float16` y `auto` (reparte Gemma entre las 2 GPU).
fp16 puede dar NaN o salidas degradadas con Gemma; si pasa, no es un bug del agente. Se reportó
overflow en fp16 en la torre de visión de Gemma 4 grandes (no E4B) y en Gemma 3 en T4. Para texto
con E4B no está confirmado: se verifica en la primera corrida (mirar `stderr` y salidas vacías).
La VRAM de `metricas.jsonl` es la **suma** de las GPU visibles (cota superior) y trae `vram_pico_por_gpu_gb`.

## Cómo correr

1. Notebook de Kaggle: acelerador **GPU T4 x2**, Internet activado (no usar P100, ver abajo).
2. Add-ons > Secrets, con estos nombres: `HF_TOKEN` (token HF de lectura) y `GITHUB_TOKEN`
   (token fine-grained de GitHub, solo lectura de *Contents* y solo sobre este repo).
3. Celda 1, clonar el repo privado sin imprimir ni guardar el token en el remoto:

```python
import base64, subprocess
from kaggle_secrets import UserSecretsClient
t = UserSecretsClient().get_secret("GITHUB_TOKEN")
h = base64.b64encode(f"x-access-token:{t}".encode()).decode()
subprocess.run(["git", "-c", f"http.extraheader=AUTHORIZATION: basic {h}", "clone",
                "https://github.com/FranCalveyra/tp_final_ia_gen.git", "/kaggle/working/repo"], check=True)
```

4. Celda 2, mirar el driver y las dependencias: `!nvidia-smi` y luego `!pip install -q -r /kaggle/working/repo/requirements-inferencia.txt`.
5. Celda 3, correr (carpeta nueva por corrida; si existe, falla):

```
!cd /kaggle/working/repo && python infra/kaggle_baseline.py --variante original --cantidad 3 --salida /kaggle/working/corridas/t4-original-01
```

Resultado en la salida: `salida/` (trazas del kit), `metricas.jsonl`, `casos.jsonl`, `normas/` y `runner.json`
(config, commit, si el árbol está sucio, GPU, versiones de torch/CUDA, returncode, stdout/stderr con el token tapado).
Descargá la carpeta desde Output. Puntuar es una fase local aparte (ver [modelo-modal.md](modelo-modal.md)).

## Dependencias: riesgo conocido

`requirements-inferencia.txt` fija `torch==2.15.0` con runtime CUDA **13.2**. Según las notas de PyTorch 2.11,
los wheels de PyPI con CUDA 13 solo soportan Turing (sm_75) o más nuevo: **T4 sí, P100 (sm_60) no**.
CUDA 13 pide un driver reciente; si `nvidia-smi` muestra un driver de la serie 5xx viejo, `torch.cuda.is_available()`
puede dar `False`. No se verificó el driver actual de Kaggle. Si pasa: instalar un torch con CUDA 12.x compatible
con ese driver (o usar el torch de la imagen) y las demás versiones fijadas sin reinstalar torch, y anotarlo en `runner.json`/bitácora.
`transformers==5.22.1` podría exigir un torch mínimo: tampoco verificado.
