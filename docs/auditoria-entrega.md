# Auditoría de la entrega contra mission.md (2026-11-12)

Recorrido de la consigna punto por punto contra el repo (rama `fix/3-perdida-supervisada`). Sin GPU. INFORME y
Loom quedan fuera de este alcance (los arma el equipo). Estado: ✅ cumple · ⚠️ cumple a medias · ❌ falta.

## Huecos que bajan la nota o rompen la corrida

| # | Requisito (mission.md) | Estado | Qué pasa | Arreglo |
|---|---|---|---|---|
| 1 | Auditabilidad: «cada número reportado deja todas las conversaciones de la evaluación… el resultado caso por caso» (:162) | ✅ resuelto | Estaban solo en `corridas/` (ignorado) y en Kaggle. | Versionadas en [`evaluaciones/`](../evaluaciones/README.md) (v25, v27, v34, v35; 18 MB) y tabla regenerada desde ahí con `sucio=False`. Cada evaluación nueva que se reporte (v36, fase 3) se agrega igual. |
| 2 | La entrega corre con `MODELO_LOTE` por defecto | ✅ resuelto: default 4 (`LOTE_ENTREGA`) | El default de `entrega.py` es **lote 1**, pero todos los números de la entrega salieron con **lote 4**. En T4, lote 1 tarda ~2× por caso; y «evaluar en lote puede mover algún caso» (:165). | Fijar lote 4 como default de la entrega (en T4, sin entrenar: mismos 10 resueltos que lote 1 y conversación idéntica en 49/50; [Parte 5](informe-parte5.md#2-casos-a-la-vez-al-evaluar-lote)) o reportar con lote 1. Decisión pendiente; idealmente medida en A100. |
| 3 | Acceso de la cátedra a HF y W&B (:239–240, :193) | ❌ bloqueado | Repo HF y proyecto W&B privados sin invitar a la cátedra: `construir_agente()` falla con 401. | Invitar a sus usuarios (#16); falta saber cuáles. |
| 4 | A100 de 40 GB para entrenar y correr (:98, :220, :239) | ❌ bloqueado | Todo se midió en Kaggle T4 x2 (KI-A100-001). No hay prueba de que la entrega cargue y corra en una A100 con `cuda:0` y bf16. | Conseguir A100 o acuerdo explícito con la cátedra. |
| 5 | README punto 1: `.env.example` con las variables (:228) | ✅ resuelto (Fran, 2026-11-12) | Tenía `ADAPTER_REPO=` vacío: exportarlo tal cual corría **sin adapter**. | `ADAPTER_*` comentadas, comentario de `PROMPT_SPEC` corregido y `MODELO_LOTE=4`. |

## Lo que ya cumple

| Requisito | Estado | Evidencia |
|---|---|---|
| `entrega.py` con `construir_buscador` y `construir_agente` | ✅ | `entrega.py`; tests `tests/test_entrega.py` |
| `pyproject.toml` y `uv.lock` fijados; `uv sync` instala la inferencia | ✅ | `default-groups = ["inferencia"]`; `uv lock --check` resuelve |
| `admision/` sin modificar | ✅ | `git diff 601192c HEAD -- admision/` vacío |
| `historial_clasificado.csv`: 2.000 filas `id,tipo` | ✅ | 1.600 correcta, 240 error, 160 ruido |
| `costumbres.md` con ids que las respaldan | ✅ | 2 costumbres + desvíos por operador |
| Modelo ≤ 8B, sin servicios externos en operación | ✅ | Gemma 4 E4B; el modelo se baja de HF al construir el agente |
| Tope de 40 llamadas, 5 herramientas, 8 campos | ✅ | `AgentePrompt`; tests |
| Prompt optimizado antes de entrenar, sin usar práctica para elegir | ✅ | [optimización de prompt](optimizacion-prompt.md), conjunto `datos/prompt-dev/v1` |
| Práctica solo para reportar | ⚠️ acotado | La falla del aviso se **vio** en el análisis de errores de práctica (pasada 7). Lo que se eligió después (hipótesis, remedio `v4-aviso`, y si la v36 reemplaza al adapter) se decide **solo con el conjunto de desarrollo** (v35, v36, v38, v39: la v39 se adoptó con 31/40 en el dev); la práctica de la v39 (Kaggle v40) está pendiente, es solo para reportar y no cambia la decisión. Queda declarado como límite en el [análisis de errores](analisis-errores.md). |
| Entrenamiento con el prompt optimizado | ✅ | `v3-procedimiento` (`dc41674a`) + continuación `v6-aviso-datos`; adapter de la entrega `770c9fac` (v39) |
| Cada entrenamiento: conversaciones, config, semilla, W&B, checkpoints HF con revisión | ✅ | `runner.json`, `estado.json`, `publicaciones`; W&B `3add2742ddfd4b42` |
| Tabla de la Parte 4 (4 filas × 3 versiones + referencia de la cátedra) | ✅ | [`evaluaciones/practica/tabla`](../evaluaciones/practica/tabla/tabla.md), regenerada desde las corridas versionadas (`sucio=False`) |
| README con los 6 puntos | ✅ | `README.md` (2026-11-12), salvo el aviso de `.env.example` |
| Todo archivo generado sale de un script | ✅ | Tabla del punto 5 del README |
| Formato de los mensajes optimizado sin entrenar (:135) | ✅ | Tamaño (fase 2) y conducta (fase 3, v37: 10/40 contra 9/40, dentro del ruido; no adoptado) en [optimización de prompt](optimizacion-prompt.md) |

## Para revisar más tarde

- **W&B, corridas fallidas (:240):** v28–v31 cayeron antes de entrenar; confirmar que cada una dejó su corrida en
  W&B con conversaciones, configuración y semilla (sin acceso local a W&B no se pudo verificar).
- **Commits repartidos parejo (:166):** hoy la mayoría son de Fran; no se arregla con código.
- **Mail de envío (:257):** necesita commit o tag final, revisión del adapter y links; armarlo al final.
