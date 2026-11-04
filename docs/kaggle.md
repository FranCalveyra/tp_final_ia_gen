# Respaldo de desarrollo: Kaggle (T4 x2)

Estado: **probado con dobles** (tests del runner, config y métricas; el entrenamiento humo se describe al final). No se ejecutó nada en Kaggle ni en GPU.
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
sucio, GPU, versiones de torch/CUDA) y se completa al terminar con `estado` (`terminada` o `fallida`),
`returncode`, `error_agente` y las últimas 200 líneas en `log_final`. `runner.log` se escribe línea a línea.
Los dos tapan el token. Si una corrida quedó `en_curso` sin proceso vivo, se cortó: mirar el final de `runner.log`.
**El kit atrapa los errores de `jugar_lote` y sale con 0**: el runner lee `salida/tiempos.json` y, si hay
`error`, marca la corrida `fallida` con `returncode 1`. Pasó con los lotes 8 y 16 (error de CUDA a mitad de camino).

```
!python -c "import json;d=json.load(open('/kaggle/working/corridas/<carpeta>/runner.json'));print(d['estado'],d.get('returncode'));print(d.get('log_final','')[-3000:])"
```

Para analizar: `zip -qr` de las carpetas, descargar desde el panel de archivos (no ejecuta celdas) y
puntuar local con `admision.correr puntuar` (ver [modelo-modal.md](modelo-modal.md)).

## Entrenamiento SFT (humo) en Kaggle

Estado: **implementado y probado con dobles**. No se ejecutó en GPU. fp16 en T4 con Gemma no está probado
para entrenar (puede dar NaN/overflow: el `GradScaler` lo atenúa, no lo descarta). Modal queda como está.
Es una **prueba de pipeline** con datos sintéticos: no dice nada de la calidad del modelo.

**Qué hace.** `infra/kaggle_entrenamiento.py` llama a `scripts.entrenar` (el mismo `entrenar` de siempre) en
un subproceso con `--precision fp16 --dispositivos auto` y **siempre** `--online --publicar`: en Kaggle los
archivos de una versión cortada se pueden perder, así que cada checkpoint sube a HF (repo privado `hf_repo` de
la configuración, `FranCalveyra/tp-admision`) y W&B registra la corrida. Con `auto` el modelo se reparte entre
las 2 T4 en un solo proceso: Transformers lo detecta por `hf_device_map` y no usa DataParallel
(`trainer.py:459-465,507`). El adapter LoRA queda en fp32 y la base en fp16 (PEFT castea con
`autocast_adapter_dtype=True`). bf16 en T4 falla en Transformers (`training_args.py:1741-1746`): por eso fp16.
Cada checkpoint de fp16 incluye `scaler.pt` y la reanudación lo exige.

**Secrets de Kaggle** (Add-ons > Secrets): `GITHUB_TOKEN` (igual que arriba), `HF_WRITE_TOKEN` (escritura en el
repo `hf_repo`; el mismo token baja el modelo base si la cuenta aceptó la licencia de Gemma) y `WANDB_API_KEY`.
El runner también acepta esas variables de entorno. Los tokens viajan **solo por entorno** al subproceso: nunca
por argumentos, ni en `runner.json`, ni en `runner.log` (se tapan con `***`). Falla antes de gastar GPU si falta
`HF_WRITE_TOKEN` o `WANDB_API_KEY` (publicar exige W&B online). `--sin-publicar` existe solo para ensayos sin
durabilidad; ahí W&B queda offline y `runner.json` lo dice.

**Evidencia.** `runner.json` (al arrancar `en_curso`; al terminar `terminada`/`fallida`, `returncode`,
`log_final`, `publicaciones` con repo y revisión inmutable de cada checkpoint), `runner.log` línea a línea,
y `entrenamiento/` con `corrida.json` (incluye `ejecucion`: precisión, dispositivos, `guardar_cada`), `estado.json`
y checkpoints. Registra commit, `sucio`, GPU, torch/CUDA. La salida debe ser nueva; si existe, falla.
**No escribas salidas dentro del repo clonado:** un archivo sin trackear marca `dirty=true` y la reanudación
exige el mismo código.

**Datos de humo.** `scripts.datos_humo` genera 4 conversaciones **sintéticas** (de la forma de
`tests/test_conversaciones.py`, sin práctica ni esperadas) y una configuración copiada de
`configuraciones/sft-ejemplo.json` con `batch_size=1`, `gradient_accumulation_steps=1`, `epochs=2`: son 8 pasos.
Con `--guardar-cada 2` salen checkpoints en 2, 4, 6 y 8, más `final`. Luego los CLI de siempre preparan y tokenizan.
Son deterministas: en otra sesión, con el mismo commit, dan los mismos hashes (hace falta para reanudar).

**Celdas (versión guardada, T4 x2, Internet activado).** Celda 1: la misma de clonado de arriba. Celda 2:

```
!nvidia-smi
!pip install -q -r /kaggle/working/repo/requirements-entrenamiento.txt
!pip uninstall -y torchvision torchaudio torchcodec torchao torchdata
```

Celda 3, entradas (el tokenizador de Gemma está gateado: el token va por entorno, y solo se descargan archivos del tokenizador):

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
repo, base = "/kaggle/working/repo", "/kaggle/working/corridas"
env = {**os.environ, "HF_TOKEN": UserSecretsClient().get_secret("HF_WRITE_TOKEN")}
pasos = [["python", "-m", "scripts.datos_humo", "--salida", f"{base}/humo-datos"],
         ["python", "-m", "scripts.preparar_entrenamiento", "--datos", f"{base}/humo-datos/conversaciones.jsonl",
          "--configuracion", f"{base}/humo-datos/configuracion.json", "--salida", f"{base}/humo-preparada"],
         ["python", "-m", "scripts.tokenizar_entrenamiento", "--preparacion", f"{base}/humo-preparada",
          "--salida", f"{base}/humo-tokens"]]
os.makedirs(base, exist_ok=True)
for cmd in pasos:
    r = subprocess.run(cmd, cwd=repo, env=env)
    print(cmd[2], "returncode", r.returncode)
    if r.returncode: break
```

Celda 4, entrenar (una sola celda, sin tocar el notebook hasta que termine):

```
!cd /kaggle/working/repo && python infra/kaggle_entrenamiento.py --preparacion /kaggle/working/corridas/humo-preparada --tokens /kaggle/working/corridas/humo-tokens --salida /kaggle/working/corridas/humo-entrena-1 --guardar-cada 2
```

Celda 5, evidencia: `!python -c "import json;d=json.load(open('/kaggle/working/corridas/humo-entrena-1/runner.json'));print(d['estado'],d.get('returncode'));print(d.get('publicaciones'));print(d.get('log_final','')[-2000:])"`

**Prueba de reanudación** (otra versión guardada, simula el corte). Celdas 1 a 3 iguales (mismo commit, árbol limpio,
mismas carpetas de entradas). La revisión sale de `publicaciones` en el `runner.json` anterior (o del historial de commits
del repo HF; nunca `main`). Celda 4, con la revisión real en lugar de `<revision>`:

```
!cd /kaggle/working/repo && python infra/kaggle_entrenamiento.py --preparacion /kaggle/working/corridas/humo-preparada --tokens /kaggle/working/corridas/humo-tokens --salida /kaggle/working/corridas/humo-entrena-2 --guardar-cada 2 --reanudar-checkpoint checkpoint-2 --reanudar-revision <revision>
```

El runner baja solo `checkpoint-2/*` en esa revisión (`snapshot_download` con `revision` y `allow_patterns`), copia su
`corrida.json` a la raíz y valida hashes e identidad (`validar_reanudacion`) **antes** de cargar el modelo; después
`entrenar` retoma desde el checkpoint (mismo run de W&B, `resume="must"`). No puede cambiar nada de modo, precisión,
dispositivos, `--guardar-cada`, datos ni código. Éxito esperado: pasos 3 a 8 y checkpoints 4, 6, 8 nuevos en HF.

**GPU estimada** (sin medir): primera corrida unos 5 a 8 min (bajar y cargar el modelo domina; 8 pasos de ~300 tokens y
subidas de adapter chicas) y la reanudación otros 4 a 6 min: ~15 min de la cuota en total.

## Dependencias: riesgo conocido

`requirements-inferencia.txt` fija `torch==2.15.0` con runtime CUDA **13.2**. Según las notas de PyTorch 2.11,
los wheels de PyPI con CUDA 13 solo soportan Turing (sm_75) o más nuevo: **T4 sí, P100 (sm_60) no**.
CUDA 13 pide un driver reciente; si `nvidia-smi` muestra un driver de la serie 5xx viejo, `torch.cuda.is_available()`
puede dar `False`. No se verificó el driver actual de Kaggle. Si pasa: instalar un torch con CUDA 12.x compatible
con ese driver (o usar el torch de la imagen) y las demás versiones fijadas sin reinstalar torch, y anotarlo en `runner.json`/bitácora.
`transformers==5.22.1` podría exigir un torch mínimo: tampoco verificado.
