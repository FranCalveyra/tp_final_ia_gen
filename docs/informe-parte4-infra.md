# Parte 4: infraestructura SFT aportada por Fran (borrador integrable)

**Aporte listo para integrar; no es el informe de experimentos ni el `INFORME.md` final.**
La estrategia implementada es SFT LoRA sobre Gemma 4 E4B, supervisando únicamente respuestas assistant.
La evidencia real disponible prueba funcionamiento técnico con datos sintéticos cortos en T4, no mejora del agente.

## Diseño y reproducibilidad

| Responsabilidad | Implementación y contrato | Fuente sin duplicar detalle |
|---|---|---|
| Entrada trazable | Conversión fiel de trazas a v1, origen/prompt explícitos y referencias fuente/id; no cura datos | [Conversaciones](conversaciones-entrenamiento.md), [auditoría](auditoria-trazas.md) |
| Congelar experimento | Bytes/hash de conversaciones y config, revisión base, semilla y código; salidas exclusivas | [Preparación](preparacion-entrenamiento.md) |
| Pérdida correcta | Máscara assistant-only, contexto/padding `-100`, prefijos verificados y rechazo por longitud sin truncar | [Tokenización](tokenizacion-entrenamiento.md) |
| Entrenar/retomar | LoRA atención del lenguaje r=8/alpha=16/dropout=0, optimizer/scheduler/RNG/Trainer y scaler fp16 en checkpoints completos | [SFT: contratos y evidencia](entrenamiento-sft.md) |
| Durabilidad/linaje | HF privado con revisión por checkpoint, W&B con config/semilla/hashes y recuperación desde HF fijado | [Kaggle operativo](kaggle.md#entrenamiento-sft-humo-en-kaggle) |

Los hiperparámetros son una configuración inicial técnica, **no una selección experimental**.
El proceso no consulta fichas esperadas para entrenar. La curación/procedencia y prompt deben acordarse con el equipo;
validar formato no certifica decisiones ni ausencia de contaminación. El código/estado Git sucio no se reconstruye
solo desde un hash de commit, por eso se registra y no se presenta como reproducibilidad completa.

## Evidencia y límites para el informe

- **Implementado y probado CPU:** contratos, máscaras, integridad/rechazo de cambios, publicación y reanudación
  con dobles. El tokenizador real también tiene prueba CPU, separada de las pruebas con dobles.
- **Validado en GPU T4:** humo de ocho pasos, checkpoints 2/4/6/8/final en HF; otra sesión retomó checkpoint 2
  publicado y completó con mismo run_id W&B. Revisión, comandos y artefactos están en la matriz SFT enlazada.
- **Intentos fallidos incluidos:** incompatibilidad LoRA con torres visión/audio y errores previos de carga
  quedaron registrados en la [bitácora](progreso.md); no se borran ni cuentan como corridas exitosas.
- **No demostrado:** calidad SFT con conversaciones curadas, barrido de hiperparámetros, DPO/refuerzo o selección
  de checkpoint final (experimentos de Martu); tampoco ajuste A100-40GB, deuda [KI-A100-001](../README.md#ki-a100-001--entrenamiento-evaluación-y-recursos-sin-validar-en-a100-40gb).

## Para completar la Parte 4 conjunta

Martu aporta estrategia/config final, datos/prompt identificados y resultados de sin entrenar / solo prompt /
solo entrenamiento / ambos en original/b/c, como pide la [consigna](../mission.md#parte-4-entrenamiento).
Fran aporta esta infraestructura y las mediciones de [Parte 5](informe-parte5.md).
**Aceptación de este aporte:** integrar texto y enlaces en el informe compartido, manteniendo cada nivel de
validación y sus artefactos. No usar la curva de pérdida del humo como evidencia de calidad ni cerrar #5/#16 por este borrador.
