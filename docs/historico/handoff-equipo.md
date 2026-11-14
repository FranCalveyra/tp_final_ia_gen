# Handoff del equipo: avanzar sin esperar al entrenamiento final

**Se puede repartir trabajo ahora, pero no eliminar sus dependencias.** El tracker está en
[`FranCalveyra/tp_final_ia_gen`](https://github.com/FranCalveyra/tp_final_ia_gen/issues), no en upstream.
Asignaciones realizadas y verificadas por la sesión principal el 2026-11-05:
Eze `ezequielrickert` #6–9; Yas `Yasmindesivo` #10–12; Martu `MartuGiordanelli` #13–15; los cuatro #16.

> **Rama vigente: `main`.** El paquete se integró localmente por fast-forward desde `feat/training-handoff`;
> la publicación de esta integración debe confirmarse con el hash comunicado por chat.
> El commit inicial `3d7da60` y la rama feature quedan como historia, no como ruta operativa actual.
> Publicar código no confirma accesos externos. Nunca compartir `.env`, tokens, cachés, pesos ni `corridas/`.

### Obtener el paquete en main tras confirmarse la publicación

Primero `git status --short`: si muestra cambios, **detenete** y preservalos antes de cambiar de rama;
no hagas reset, limpieza ni stash automático. Con árbol limpio:

```bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

Compará el hash con el commit comunicado por chat. Si no existe `main` local, primero hacé `git fetch origin`
y `git switch --track origin/main`; luego actualizá con `pull --ff-only`. Si falla por divergencia, detenete:
no forzar ni reescribir trabajo local.

**Ruta actual (decisión 2026-11-05): Kaggle para desarrollo. Modal descartado definitivamente.**
A100-40GB se exige **para entrenamiento y evaluación**, no solo para la entrega final; T4 no lo reemplaza.
La deuda está registrada en [KI-A100-001](../../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb).
T4 queda autorizado en principio para desarrollo, pero el alcance concreto de la próxima corrida GPU requiere confirmación.
No hay excepción de hardware acordada con la cátedra.
Menú de `franciscocalveyra/tp-admision-baseline` verificado el 2026-11-05: solo `None`, `GPU T4 x2` y
`TPU v5e-8`, sin A100. Consulta read-only, Draft Session apagada; no se inició GPU ni se inspeccionó VRAM.
No describe los permisos de otras cuentas/notebooks: ver [evidencia y alcance](../guias/kaggle.md).

## Qué compartir y qué debe devolver cada uno

| Persona | Entradas que necesita | Salida esperada / contrato | Dependencia inevitable |
|---|---|---|---|
| **Yas** #10–12 | `historial/historial.csv`, `historial/trazas.jsonl`, normas y parámetros recibidos | Clasificación reproducible `historial_clasificado.csv` (`id,tipo`, 2.000 filas), `costumbres.md` con ids/evidencia y criterio de selección/sintéticos | Las fichas de operadores no son verdad certificada; separar error de ruido antes de seleccionar |
| **Eze** #6–9 | Normas/carpeta real, clasificación/costumbres de Yas, trazas y auditoría | Buscador/evaluación RAG; conversaciones v1 seleccionadas y trazables; ciclo/formato coordinado; evaluación con checkpoint/prompt identificados | #8 depende de selección Yas; formato/prompt se acuerdan con Martu; no regenerar búsquedas con números incompatibles |
| **Martu** #13–15 | Contrato del agente/buscador, costumbres disponibles y generador de casos; luego conversaciones preparadas y trainer | Prompt final explícito con evidencia en casos generados, análisis por controles; después experimentos y selección de checkpoint | Puede empezar #13/#14 sin SFT final; #15 requiere datos y ruta GPU habilitada, no bloquea todo su trabajo |
| **Fran** #3–5 | Conversaciones/prompt acordados, configuración/semilla, permisos de sus cuentas | Pipeline/handoff, prueba A100 pendiente, evidencia de checkpoints/reanudación y recursos | No puede certificar curación/prompt ajenos ni A100 a partir del humo T4 |

## Paquete mínimo y fuente de verdad

- **Datos originales:** `entrenamiento/trazas.jsonl` (800 de referencia) y `historial/trazas.jsonl` (2.000).
  Hashes, compatibilidad y límites en [auditoría](../informe/auditoria-trazas.md). Referencia varía parámetros entre trazas;
  conservar sus observaciones. No asumir que las 2.000 sirven enteras para SFT.
- **Contrato v1:** [conversaciones-entrenamiento.md](../informe/conversaciones-entrenamiento.md), validado por
  `scripts.validar_conversaciones`. **Versionado propuesto, aún por acordar** con Eze/Yas/Martu:
  `version,id,origen,referencias,messages`; JSON de herramientas y observaciones del agente, registro final.
  Acordar además procedencia de las 800: el conversor exige origen explícito, no lo inventa.
- **Conversor incluido en la rama publicada:** `admision_tp/conversion_trazas.py`, `scripts/convertir_trazas.py` y tests.
  Convierte estructura sin curar; exige prompt elegido, conserva llamadas/observaciones y omite resultado terminal.
- **Config de experimento:** `configuraciones/sft-ejemplo.json`, con valores ilustrativos. No es la receta final.
  Preparación congela conversaciones/config/semilla; tokenización valida límite sin recortes silenciosos.

## Guías operativas

| Trabajo | Guía |
|---|---|
| Entorno/dependencias del equipo | [entorno-equipo.md](../guias/entorno-equipo.md) |
| RAG y medición | [buscador-baseline.md](../informe/buscador-baseline.md) |
| Conversión y validación CPU | [conversaciones-entrenamiento.md](../informe/conversaciones-entrenamiento.md#conversor-cpu-de-trazas-implementado) |
| Snapshots y tokenización | [preparacion-entrenamiento.md](../informe/preparacion-entrenamiento.md), [tokenizacion-entrenamiento.md](../informe/tokenizacion-entrenamiento.md) |
| SFT: comandos, permisos, criterios #3 | [entrenamiento-sft.md](../informe/entrenamiento-sft.md) |
| GPU/recuperación actual | [Kaggle](../guias/kaggle.md#entrenamiento-sft-humo-en-kaggle); [Modal legacy, no usar](entrenamiento-modal.md) |
| Estado/evidencia, no etiquetas solas | [progreso.md](../progreso.md) |

## Checklist de accesos: hecho versus pendiente de comprobar

**Hecho en GitHub** (no implica acceso a otros servicios):

- [x] Paquete inicial publicado en `feat/training-handoff`, commit `3d7da60` (histórico).
- [x] Integración local fast-forward en `main`; verificar publicación del commit indicado por chat antes de obtenerlo.
- [x] Issues asignados a los usuarios de la tabla inicial.
- [ ] Cada integrante confirma que puede obtener esa rama con su propia cuenta/permisos de lectura del repo.

**Kaggle/HF/W&B: no verificado para las cuentas del equipo.** Compartir recursos y comprobar acceso con cada
persona; no marcar estos ítems por tener acceso desde la cuenta de Fran.

- [ ] **Kaggle inferencia → Eze y Martu:** compartir notebook y versión reproducible, o copia separada por persona;
  registrar enlace/version cuando estén confirmados. Enlaces no disponibles aquí: no se inventan.
- [ ] **Kaggle entrenamiento → Martu:** compartir notebook/versión o copia propia; salidas y corridas con nombres
  separados para no interrumpirse. Confirmar lectura/edición de la copia antes de ejecutar celdas GPU.
- [ ] **HF `FranCalveyra/tp-admision` → Eze/Martu:** comprobar lectura de la revisión necesaria. Martu requiere
  además escritura si publica; dar lectura no habilita uploads ni crea automáticamente permisos de escritura.
- [ ] **Gemma gateado → cada cuenta HF:** aceptar licencia/obtener acceso individual al modelo base y verificarlo;
  compartir el adapter privado no concede acceso al modelo base.
- [ ] **W&B `francalveyra-ia-gen/tp-admision` → Martu:** permiso para registrar corridas (y Eze si sus experimentos
  también registran allí); confirmar membresía efectiva, no solo que se envió invitación.
- [ ] **Secrets propios de Kaggle:** cada persona crea `GITHUB_TOKEN` fine-grained para este repo, *Contents: read*;
  inferencia usa `HF_TOKEN` de lectura. Entrenamiento usa `HF_WRITE_TOKEN` con acceso base y escritura al repo
  privado, más `WANDB_API_KEY` de su cuenta. Activar en su copia del notebook solo los necesarios.

**No compartir tokens ni exportar Secrets del notebook de Fran.** Registrar solo persona/recurso/permiso y
confirmación de acceso, nunca valores. Obtener/validar credenciales y ejecutar GPU son alcances separados:
una invitación o copia de notebook no autoriza consumir cuota. Las celdas de clonado deben seleccionar la rama
vigente `main` y verificar el commit publicado indicado por chat; no usar una copia antigua del notebook/repo.

## Pendientes por persona y aceptación operativa

| Persona | Confirmación que falta | Datos de cuenta aún necesarios |
|---|---|---|
| **Eze** | Obtener rama/commit; acceso a copia/versión de notebook inferencia; HF lectura del adapter y acceso propio a Gemma. W&B solo si registrará experimentos | Usuario Kaggle y cuenta HF; identidad W&B si corresponde, no tokens |
| **Martu** | Obtener rama/commit; copias inferencia/entrenamiento; HF lectura/escritura si publica y Gemma; W&B con permiso registrar; Secrets propios configurados | Usuario Kaggle, cuenta HF e identidad W&B; confirmar membresías efectivas |
| **Yas** | Obtener rama/datos por hash; acordar clasificación/selección y contrato con Eze | Lectura GitHub efectiva con `Yasmindesivo`. Kaggle/HF/W&B no son requisitos del curado CPU salvo que ella elija un experimento que los necesite |

Los usernames de GitHub conocidos no identifican automáticamente las cuentas de Kaggle/HF/W&B.
No se completaron ni verificaron esos permisos externos desde esta sesión.

**Aceptación inicial de Martu, sin pesos ni consumo GPU:**

1. Con árbol limpio, obtener la rama según instrucciones iniciales y confirmar el hash comunicado por chat.
2. En un entorno ya instalado, ejecutar solo la ayuda CPU:

```bash
.venv/bin/python -m scripts.entrenar --help
.venv/bin/python -m scripts.convertir_trazas --help
```

3. Revisar desde sus cuentas la copia del notebook, membresía HF/W&B y licencia/acceso Gemma; confirmar
   permisos/secrets configurados **sin mostrar valores**. No iniciar Draft Session ni correr celdas,
   tokenización, entrenamiento, descargas o llamadas API que creen corridas.
4. Responder qué verificó y qué acceso falta. Esto confirma recepción/preflight, **no** el criterio
   “Martu lo usa para sus experimentos”: su uso efectivo sigue pendiente y requiere alcance GPU propio.

## Accesos y frontera de costo

Cada integrante usa **su cuenta y sus permisos**, no tokens de otra persona. Para bajar Gemma gateado,
su cuenta HF debe tener acceso/licencia aceptada; para publicar, escritura explícita en el repo privado elegido.
W&B requiere acceso a la entidad/proyecto configurados. Compartir acceso al repo/corrida no comparte credenciales.
`HF_TOKEN`, `HF_WRITE_TOKEN` (runner Kaggle) y `WANDB_API_KEY` se configuran por entorno/Secrets; nunca por chat,
argumentos ni commits. Los Secrets/comandos de Modal son legacy: no crearlos ni ejecutar esa ruta.

**CPU sin cloud:** conversión/validación, preparación y tests existentes. Tokenización no usa pesos/GPU, pero
puede descargar tokenizer si falta caché. **Con autorización propia:** instalaciones/descargas, pesos,
GPU/entrenamiento y publicaciones HF/W&B. No ejecutar comandos GPU como un check inocuo.
La ruta T4 tiene humo y recuperación HF reales; hardware final A100-40GB sigue pendiente por vía alternativa.
La infraestructura genérica de #3 tiene cuatro criterios técnicos respaldados; falta aceptación/accesos y uso
efectivo por Martu. La ayuda CLI no es un experimento. No implica calidad ni validación A100.

**Primer acuerdo compartido:** fijar versión/formato, procedencia y archivos de entrada por hash; luego cada uno
puede avanzar en su frente sin esperar al checkpoint final. Esto no declara terminados #3 ni los tickets del equipo.
