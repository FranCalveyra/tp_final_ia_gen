# Auditoría de la entrega contra mission.md (2026-11-12)

Recorrido de la consigna punto por punto contra el repo (rama `fix/3-perdida-supervisada`). Sin GPU. INFORME y
Loom quedan fuera de este alcance (los arma el equipo). Estado: ✅ cumple · ⚠️ cumple a medias · ❌ falta.

## Huecos que bajan la nota o rompen la corrida

| # | Requisito (mission.md) | Estado | Qué pasa | Arreglo |
|---|---|---|---|---|
| 1 | Auditabilidad: «cada número reportado deja todas las conversaciones de la evaluación… el resultado caso por caso» (:162) | ✅ resuelto | Estaban solo en `corridas/` (ignorado) y en Kaggle. | Versionadas en [`evaluaciones/`](../evaluaciones/README.md) (v25, v27, v34, v35; 18 MB) y tabla regenerada desde ahí con `sucio=False`. Cada evaluación nueva que se reporte (v36, fase 3) se agrega igual. |
| 2 | La entrega corre con `MODELO_LOTE` por defecto | ⚠️ | El default de `entrega.py` es **lote 1**, pero todos los números de la entrega salieron con **lote 4**. En T4, lote 1 tarda ~2× por caso; y «evaluar en lote puede mover algún caso» (:165). | Fijar lote 4 como default de la entrega (en T4, sin entrenar: mismos 10 resueltos que lote 1 y conversación idéntica en 49/50; [Parte 5](informe-parte5.md#2-casos-a-la-vez-al-evaluar-lote)) o reportar con lote 1. Decisión pendiente; idealmente medida en A100. |
| 3 | Acceso de la cátedra a HF y W&B (:239–240, :193) | ❌ bloqueado | Repo HF y proyecto W&B privados sin invitar a la cátedra: `construir_agente()` falla con 401. | Invitar a sus usuarios (#16); falta saber cuáles. |
| 4 | A100 de 40 GB para entrenar y correr (:98, :220, :239) | ❌ bloqueado | Todo se midió en Kaggle T4 x2 (KI-A100-001). No hay prueba de que la entrega cargue y corra en una A100 con `cuda:0` y bf16. | Conseguir A100 o acuerdo explícito con la cátedra. |
| 5 | README punto 1: `.env.example` con las variables (:228) | ⚠️ | Tiene `ADAPTER_REPO=` vacío: exportarlo tal cual corre **sin adapter**. El comentario de `PROMPT_SPEC` dice baseline. | Comentar las líneas `ADAPTER_*` y corregir el comentario (lo hace Fran: el archivo está protegido para el agente). |

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
| Entrenamiento con el prompt optimizado | ✅ | `v3-procedimiento`; adapter `dc41674a` |
| Cada entrenamiento: conversaciones, config, semilla, W&B, checkpoints HF con revisión | ✅ | `runner.json`, `estado.json`, `publicaciones`; W&B `3add2742ddfd4b42` |
| Tabla de la Parte 4 (4 filas × 3 versiones + referencia de la cátedra) | ⚠️ | Números en la bitácora; la tabla con procedencia está en `corridas/tablas/` (ver hueco 1) |
| README con los 6 puntos | ✅ | `README.md` (2026-11-12), salvo el aviso de `.env.example` |
| Todo archivo generado sale de un script | ✅ | Tabla del punto 5 del README |
| Formato de los mensajes optimizado sin entrenar (:135) | ⚠️ | Solo tamaño; la conducta se mide en la fase 3 (después de v36). Respaldo documentado. |

## Para revisar más tarde

- **W&B, corridas fallidas (:240):** v28–v31 cayeron antes de entrenar; confirmar que cada una dejó su corrida en
  W&B con conversaciones, configuración y semilla (sin acceso local a W&B no se pudo verificar).
- **Commits repartidos parejo (:166):** hoy la mayoría son de Fran; no se arregla con código.
- **Mail de envío (:257):** necesita commit o tag final, revisión del adapter y links; armarlo al final.
