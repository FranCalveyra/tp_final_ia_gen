# Diferencias consecutivas v1–v41

La comparación normaliza outputs, `execution_count`, metadata de celdas, orden de claves JSON y otros campos
transitorios. Cada fila separa lo visible en el notebook de lo que solo puede acreditarse inspeccionando el
commit ejecutado.

Tipos: **C** código/circuito; **K** configuración; **P** parámetros; **I** información o metadata.

| Transición | Tipo | Cambio verificable en notebook/metadata | Código del repositorio y alcance de la afirmación |
|---|---|---|---|
| v1→v2 | C | Clonado siempre limpio; elimina prueba de 3 casos, zips y rerun b/c; conserva baseline original/b/c. | Sin commit fijado: el contenido exacto ejecutado sale de los runners, no del notebook. |
| v2→v3 | C/P/I | Variantes original/b/c → lotes 1/8/16 sobre original; agrega diagnóstico CUDA y resumen por lote. | Sin pin; runner acredita `11bf3d6`. |
| v3→v4 | C/K | Inferencia → datos de humo, preparación, tokenización y entrenamiento; cambia requirements. | Sin pin; runner acredita `6f9abc0`. |
| v4→v5 | — | Ningún cambio: notebooks idénticos. | Los runners acreditan `6f9abc0`→`5e1837e`; cambia el traceback del CLI. El notebook solo clonaba una rama móvil. |
| v5→v6 | — | Ningún cambio: notebooks idénticos. | Runners `5e1837e`→`acc95b2`; LoRA queda limitado al language model. No es atribuible al notebook. |
| v6→v7 | K/P | Agrega checkout explícito `acc95b2`; salida nueva y reanudación checkpoint-2/revisión HF. | Primer pin reproducible del tramo. |
| v7→v8 | C/K/P | Entrenamiento → inferencia diagnóstica; lote 8; `CUDA_LAUNCH_BLOCKING` y stacktraces. | El runner acredita `522d540`; el notebook ya no fija commit. |
| v8→v9 | C/K | Diagnóstico → inferencia lote 4 solo texto + humo SFT; vuelve a requirements de entrenamiento. | El runner acredita `a828467`, que omite torres de visión/audio. |
| v9→v10 | C/K/P | SFT/solo texto → humo `prefijos` de 3 casos; fija rama y commit `5208810`. | El árbol fijado agrega registro de variante de buscador en el runner. |
| v10→v11 | C/K/I | Ejecución GPU → diagnóstico de integraciones; GPU desactivada. | No hay resultado conservado; no se infiere motivo adicional. |
| v11→v12 | C/I | Amplía el diagnóstico de lectura por mecanismos alternativos. | Sin artefactos de resultado. |
| v12→v13 | C/I | Reemplaza el diagnóstico completo por una inspección simplificada del entorno. | Sin artefactos de resultado. |
| v13→v14 | I | Código semánticamente igual; cambia orden JSON. Metadata cambia imagen de kernel privada por imagen estándar. | No hay cambio experimental acreditado. |
| v14→v15 | C/I | Vuelve al diagnóstico de v12; restaura metadata anterior. | Sin resultado. |
| v15→v16 | — | Idénticos byte a byte. | No hay diferencia acreditable. |
| v16→v17 | C/K | Diagnóstico → humo `prefijos`; GPU reactivada; rama/commit fijados. | Repite exactamente v10. |
| v17→v18 | I | Solo orden de claves JSON. | Sin diferencia semántica. |
| v18→v19 | I | Solo orden de claves; v19 vuelve a ser idéntica a v10/v17. | La bitácora solo conserva para v19 una sesión de 642 s. |
| v19→v20 | P | Prefijos 3 casos → baseline y prefijos, 50 casos cada uno. | Sigue en `5208810`. No hay resultados conservados. |
| v20→v21 | — | Idénticos byte a byte. | Sin diferencia acreditable. |
| v21→v22 | K | Solo cambia commit `5208810`→`4e3613a`. | El diff Git integra infraestructura de prompts, `AgentePrompt`, configuraciones y datos; el objetivo del notebook sigue siendo comparar buscadores. |
| v22→v23 | C/K/P | Comparación de buscadores → plomería `AgentePrompt`, 50 casos, prompt baseline; rama `main@5d4b657`. | El commit agrega/genera conversaciones v2 y utilidades normativas. La equivalencia del agente se acredita por resultados, no por el diff. |
| v23→v24 | C/K/P | Runner de prompts, 50 casos → `entrega.py`, 10 casos; `feat/7-agente-eze@2c16014`. | El árbol cambia `entrega.py`, contexto, guarda y medición. |
| v24→v25 | C/K/P | Una prueba de entrega → matriz de cinco prompts en dev y plan de evaluación baseline en práctica; `f4d7300`. | El diff agrega prompt-dev, candidatos, ranking y runner de planes. |
| v25→v26 | C/K/P | Fase completa → reintento de C y finalistas B/C con prefijos; `275c15b`. | Entre commits solo cambia documentación; las diferencias experimentales están en la celda y configuraciones ya presentes. |
| v26→v27 | C/K | Dev de prompts → plan `solo-prompt` original/b/c; `a5745f0`. | El diff agrega `configuraciones/evaluacion/solo-prompt.json`. |
| v27→v28 | C/K | Evaluación → primer SFT real; requirements de entrenamiento; `main@8843004`. | El árbol incorpora datos `v3-procedimiento` y análisis de errores. |
| v28→v29 | K | Rama pasa a `fix/3-perdida-supervisada@10e117c`; agrega allocator expandible. | El diff modifica la pérdida para calcular solo posiciones supervisadas. |
| v29→v30 | K | Solo commit `10e117c`→`f5a7196`. | El diff cambia colocación de posiciones/máscara y documenta el fallo. |
| v30→v31 | K/P | Agrega `WANDB_INIT_TIMEOUT=300`; mantiene `f5a7196`. | Aísla timeout de W&B del fallo de dispositivo. |
| v31→v32 | K | Solo commit `f5a7196`→`6169241`. | El diff pasa posiciones supervisadas como lista para que Accelerate no las mueva. |
| v32→v33 | P | Corrida nueva → salida `sft-v3-2`, reanudar checkpoint-300/revisión `83dd1d…`. | Mismo commit; los cambios son de ejecución, no de código. |
| v33→v34 | C/K | Entrenamiento → evaluación del adapter `dc41674a`; commit `3adf12e`. | El diff agrega plan `entrenado.json`. |
| v34→v35 | C/K | Práctica completa → dev del adapter con `procedimiento-v1`; commit `bea206b`. | El árbol adopta provisionalmente adapter/prompt y agrega configuración de evaluación entrenada. |
| v35→v36 | C/K/P | Evaluación → continuación SFT con `v4-aviso`, 80 pasos, y evaluación dev; commit `fb353c0`. | El diff agrega `adapter_inicial`, dataset balanceado y configuración `sft-aviso`. |
| v36→v37 | C/K | Elimina entrenamiento/adapter; evalúa `procedimiento-formato-v1` sin adapter; commit `7b92583`. | El árbol agrega el candidato de formato, documentación y artefactos previos. |
| v37→v38 | C/K/P | Formato sin adapter → SFT ponderado + dev; dataset v5, peso 20, 142 pasos; commit `984fc18`. | El diff agrega pérdida ponderada, dataset y configuración. |
| v38→v39 | K | Rutas v5/peso → v6/datos; conserva peso e hiperparámetros; commit `5a36ecf`. | El diff agrega el prefijo edad/acompañante antes de la decisión. |
| v39→v40 | C/K | Entrenamiento+dev → plan de práctica `entrenado-v39`; commit `1405aab`. | El diff adopta `770c9fac` en configuración y agrega el plan final. |
| v40→v41 | C/K | Práctica del adapter → evaluación base con buscador prefijos; `docs/informe-completo@6f6a4f5`. | El diff agrega el plan `buscador-prefijos` y documentación; v41 rehace v22. |

## Cambios de rama acreditados

| Versiones | Rama declarada |
|---|---|
| v10, v17–v22 | `feat/6-buscador-eze` |
| v23, v28 | `main` |
| v24 | `feat/7-agente-eze` |
| v25–v27 | `feat/13-prompt-eze` |
| v29–v37 | `fix/3-perdida-supervisada` |
| v38 | `feat/15-peso-aviso` |
| v39–v40 | `feat/15-aviso-datos` |
| v41 | `docs/informe-completo` |

v1–v9 y v11–v16 no fijan una rama/commit reproducible dentro del notebook. Cuando existe, el commit registrado
por el runner se conserva en [`versiones.csv`](versiones.csv); no convierte retroactivamente la celda de clonado
en reproducible, pero sí acredita el árbol de una ejecución concreta.

## Duplicados y diferencias no semánticas

- v4 = v5 = v6 byte a byte.
- v10 = v17 = v19 byte a byte.
- v12 = v15 = v16 byte a byte.
- v20 = v21 byte a byte.
- v13→v14 y v17→v18→v19 solo reordenan claves del notebook; v14 además cambia la imagen Docker del metadata.

Estos casos son especialmente importantes: una “versión nueva de Kaggle” no implica necesariamente una versión
nueva del código. En v4–v6 los resultados cambian porque el clonado tomó distintos HEAD de una rama móvil.
