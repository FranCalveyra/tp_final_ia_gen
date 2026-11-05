# Handoff del equipo: avanzar sin esperar al entrenamiento final

**Se puede repartir trabajo ahora, pero no eliminar sus dependencias.** El tracker está en
[`FranCalveyra/tp_final_ia_gen`](https://github.com/FranCalveyra/tp_final_ia_gen/issues), no en upstream.
Asignaciones realizadas y verificadas por la sesión principal el 2026-11-05:
Eze `ezequielrickert` #6–9; Yas `Yasmindesivo` #10–12; Martu `MartuGiordanelli` #13–15; los cuatro #16.

> **Rama de entrega:** paquete preparado para `feat/training-handoff`, todavía no integrado en `main`.
> Commit/push autorizados; verificar su publicación y el hash exacto informado por chat antes de usarlo.
> No alcanza con actualizar `main`. Nunca compartir `.env`, tokens, cachés, pesos ni `corridas/`.

### Obtener el paquete una vez confirmado el push

Primero `rtk git status --short`: si muestra cambios, **detenete** y preservalos antes de cambiar de rama;
no hagas reset, limpieza ni stash automático. Con árbol limpio, para crear la rama local por primera vez:

```bash
rtk git fetch origin
rtk git switch --track origin/feat/training-handoff
rtk git rev-parse HEAD
```

Compará el hash con el commit comunicado por chat. Si la rama local ya existe, el comando `--track` falla sin
reemplazarla: revisar su estado antes de actualizarla. Estos pasos no integran el paquete en `main`.

## Qué compartir y qué debe devolver cada uno

| Persona | Entradas que necesita | Salida esperada / contrato | Dependencia inevitable |
|---|---|---|---|
| **Yas** #10–12 | `historial/historial.csv`, `historial/trazas.jsonl`, normas y parámetros recibidos | Clasificación reproducible `historial_clasificado.csv` (`id,tipo`, 2.000 filas), `costumbres.md` con ids/evidencia y criterio de selección/sintéticos | Las fichas de operadores no son verdad certificada; separar error de ruido antes de seleccionar |
| **Eze** #6–9 | Normas/carpeta real, clasificación/costumbres de Yas, trazas y auditoría | Buscador/evaluación RAG; conversaciones v1 seleccionadas y trazables; ciclo/formato coordinado; evaluación con checkpoint/prompt identificados | #8 depende de selección Yas; formato/prompt se acuerdan con Martu; no regenerar búsquedas con números incompatibles |
| **Martu** #13–15 | Contrato del agente/buscador, costumbres disponibles y generador de casos; luego conversaciones preparadas y trainer | Prompt final explícito con evidencia en casos generados, análisis por controles; después experimentos y selección de checkpoint | Puede empezar #13/#14 sin SFT final; #15 requiere datos y ruta GPU habilitada, no bloquea todo su trabajo |
| **Fran** #3–5 | Conversaciones/prompt acordados, configuración/semilla, permisos de sus cuentas | Pipeline/handoff, prueba A100 pendiente, evidencia de checkpoints/reanudación y recursos | No puede certificar curación/prompt ajenos ni A100 a partir del humo T4 |

## Paquete mínimo y fuente de verdad

- **Datos originales:** `entrenamiento/trazas.jsonl` (800 de referencia) y `historial/trazas.jsonl` (2.000).
  Hashes, compatibilidad y límites en [auditoría](auditoria-trazas.md). Referencia varía parámetros entre trazas;
  conservar sus observaciones. No asumir que las 2.000 sirven enteras para SFT.
- **Contrato v1:** [conversaciones-entrenamiento.md](conversaciones-entrenamiento.md), validado por
  `scripts.validar_conversaciones`. **Versionado propuesto, aún por acordar** con Eze/Yas/Martu:
  `version,id,origen,referencias,messages`; JSON de herramientas y observaciones del agente, registro final.
  Acordar además procedencia de las 800: el conversor exige origen explícito, no lo inventa.
- **Conversor disponible localmente:** `admision_tp/conversion_trazas.py`, `scripts/convertir_trazas.py` y tests.
  Convierte estructura sin curar; exige prompt elegido, conserva llamadas/observaciones y omite resultado terminal.
- **Config de experimento:** `configuraciones/sft-ejemplo.json`, con valores ilustrativos. No es la receta final.
  Preparación congela conversaciones/config/semilla; tokenización valida límite sin recortes silenciosos.

## Guías operativas

| Trabajo | Guía |
|---|---|
| Entorno/dependencias del equipo | [entorno-equipo.md](entorno-equipo.md) |
| RAG y medición | [buscador-baseline.md](buscador-baseline.md) |
| Conversión y validación CPU | [conversaciones-entrenamiento.md](conversaciones-entrenamiento.md#conversor-cpu-de-trazas-implementado) |
| Snapshots y tokenización | [preparacion-entrenamiento.md](preparacion-entrenamiento.md), [tokenizacion-entrenamiento.md](tokenizacion-entrenamiento.md) |
| SFT: comandos, permisos, criterios #3 | [entrenamiento-sft.md](entrenamiento-sft.md) |
| GPU/recuperación por plataforma | [entrenamiento-modal.md](entrenamiento-modal.md), [kaggle.md](kaggle.md#entrenamiento-sft-humo-en-kaggle) |
| Estado/evidencia, no etiquetas solas | [progreso.md](progreso.md) |

## Accesos y frontera de costo

Cada integrante usa **su cuenta y sus permisos**, no tokens de otra persona. Para bajar Gemma gateado,
su cuenta HF debe tener acceso/licencia aceptada; para publicar, escritura explícita en el repo privado elegido.
W&B requiere acceso a la entidad/proyecto configurados. Compartir acceso al repo/corrida no comparte credenciales.
`HF_TOKEN`, `HF_WRITE_TOKEN` (runner Kaggle) y `WANDB_API_KEY` se configuran por entorno/Secrets; nunca por chat,
argumentos ni commits. En Modal, `huggingface` es lectura; `hf-write` sigue siendo un nombre propuesto no verificado.

**CPU sin cloud:** conversión/validación, preparación y tests existentes. Tokenización no usa pesos/GPU, pero
puede descargar tokenizer si falta caché. **Con autorización propia:** instalaciones/descargas, pesos,
GPU/entrenamiento, publicaciones HF/W&B y recursos Modal. No ejecutar comandos GPU como un check inocuo.
La ruta T4 está validada solo con humo; A100-40GB/Modal y su corte/reanudación reales siguen pendientes.

**Primer acuerdo compartido:** fijar versión/formato, procedencia y archivos de entrada por hash; luego cada uno
puede avanzar en su frente sin esperar al checkpoint final. Esto no declara terminados #3 ni los tickets del equipo.
