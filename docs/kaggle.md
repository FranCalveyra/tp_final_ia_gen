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
3. Celda 1, clonar o actualizar el repo privado. El token va por variables de entorno de git, **nunca como
   argumento**: si el comando falla, Python imprime los argumentos y el header en base64 expone el token
   (pasó el 2026-11-03 y hubo que revocarlo). No se usa `check=True` ni se muestra stderr.

```python
import base64, os, subprocess
from kaggle_secrets import UserSecretsClient
t = UserSecretsClient().get_secret("GITHUB_TOKEN")
env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraheader",
       "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{t}".encode()).decode()}
del t
repo = "/kaggle/working/repo"
cmd = ["git", "-C", repo, "pull"] if os.path.exists(repo) else \
      ["git", "clone", "https://github.com/FranCalveyra/tp_final_ia_gen.git", repo]
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("ok" if r.returncode == 0 else f"git falló (código {r.returncode})")
subprocess.run(["git", "-C", repo, "log", "--oneline", "-1"])
```

4. Celda 2, mirar el driver y las dependencias: `!nvidia-smi`, luego
   `!pip install -q -r /kaggle/working/repo/requirements-inferencia.txt` y luego
   **`!pip uninstall -y torchvision torchaudio torchcodec torchao torchdata`**.
   Esos paquetes de la imagen están compilados para otro torch/CUDA, y transformers los importa solo
   si están instalados (`operator torchvision::nms does not exist`, CUDA 13.0 contra 13.2). No los usamos.
   Los demás avisos de conflicto de pip son paquetes de la imagen que tampoco usamos.
5. Celda 3, correr. Para varias variantes, **una sola celda** con un loop, ejecutada una vez y sin
   tocar el notebook hasta que termine: volver a ejecutarla o interrumpirla corta la corrida en curso
   (pasó el 2026-11-03). La carpeta debe ser nueva; si existe, falla.

```
!cd /kaggle/working/repo && for v in original b c; do MODELO_LOTE=4 python infra/kaggle_baseline.py --variante $v --cantidad 50 --salida /kaggle/working/corridas/base-$v-l4; done
```

Resultado en la salida: `salida/` (trazas del kit), `metricas.jsonl`, `casos.jsonl`, `normas/`, `runner.log` y
`runner.json`. `runner.json` se escribe **al arrancar** con `estado: en_curso` (config, commit, si el árbol está
sucio, GPU, versiones de torch/CUDA) y se completa al terminar con `estado: terminada`, `returncode` y las
últimas 200 líneas en `log_final`. `runner.log` se escribe línea a línea. Los dos tapan el token.
Si una corrida quedó `en_curso` sin proceso vivo, se cortó: mirar el final de `runner.log`.

```
!python -c "import json;d=json.load(open('/kaggle/working/corridas/<carpeta>/runner.json'));print(d['estado'],d.get('returncode'));print(d.get('log_final','')[-3000:])"
```

Para analizar: `zip -qr` de las carpetas, descargar desde el panel de archivos (no ejecuta celdas) y
puntuar local con `admision.correr puntuar` (ver [modelo-modal.md](modelo-modal.md)).

## Dependencias: riesgo conocido

`requirements-inferencia.txt` fija `torch==2.15.0` con runtime CUDA **13.2**. Según las notas de PyTorch 2.11,
los wheels de PyPI con CUDA 13 solo soportan Turing (sm_75) o más nuevo: **T4 sí, P100 (sm_60) no**.
CUDA 13 pide un driver reciente; si `nvidia-smi` muestra un driver de la serie 5xx viejo, `torch.cuda.is_available()`
puede dar `False`. No se verificó el driver actual de Kaggle. Si pasa: instalar un torch con CUDA 12.x compatible
con ese driver (o usar el torch de la imagen) y las demás versiones fijadas sin reinstalar torch, y anotarlo en `runner.json`/bitácora.
`transformers==5.22.1` podría exigir un torch mínimo: tampoco verificado.
