# Auditoría histórica de Kaggle: versiones 1 a 41

Esta carpeta documenta la evolución del notebook privado `tp-admision-baseline` entre sus versiones de Kaggle
v1 y v41. El objetivo es conservar una cronología verificable de qué se intentó, qué código se ejecutó, qué
resultado produjo y por qué una alternativa continuó, se descartó o terminó en la entrega.

La auditoría no reemplaza la [bitácora](../progreso.md), las [guías operativas](../guias/kaggle.md), los
[informes técnicos](../informe/) ni los [artefactos experimentales](../../evaluaciones/). Este directorio los
indexa desde la perspectiva histórica de las versiones de Kaggle y explicita los huecos que esas fuentes no
permiten resolver.

## Alcance

- 41 versiones consecutivas, v1 a v41.
- 40 transiciones entre notebooks.
- Ejecuciones exitosas, fallidas, interrumpidas, parciales y descartadas.
- Commits Git fijados por el notebook o acreditados por `runner.json`.
- Duraciones de sesión, runner y entrenamiento, siempre separadas.
- Hardware, VRAM, revisiones inmutables de Hugging Face y runs de W&B cuando existe un recibo local.
- Resultados de desarrollo y práctica separados. No existe evidencia del test secreto.

La evidencia GPU corresponde a **2 Tesla T4, fp16 y `device_map=auto`**. No valida funcionamiento, memoria,
rendimiento ni calidad en la A100 de 40 GB exigida para la entrega.

## Fuentes

La reconstrucción usa exclusivamente:

1. La extracción histórica externa `kaggle-auditoria/v1` a `v41`, con un notebook y un
   `kernel-metadata.json` por versión.
2. Los resultados externos disponibles bajo `Downloads/kaggle/`.
3. La referencia local `origin/main` en `2c4485abda80898c3173e0aa5eb73fba8882573c` y los commits alcanzables
   desde el repositorio. No se hizo `fetch`: el hash identifica exactamente la referencia local auditada.
4. Los artefactos versionados bajo [`evaluaciones/`](../../evaluaciones/).
5. La [bitácora](../progreso.md), la [guía de Kaggle](../guias/kaggle.md) y los documentos de
   [`docs/informe/`](../informe/).

Las raíces externas se describen por nombre lógico y no se copian al repositorio. Los hashes permiten comprobar
una futura reextracción sin incorporar notebooks privados, logs completos, pesos ni datos sensibles.

## Método

1. Se comprobó que existieran exactamente las carpetas v1–v41 y que cada una contuviera los dos JSON esperados.
2. Se calculó SHA-256 sobre los notebooks originales, antes de cualquier normalización.
3. Para comparar versiones se ignoraron `outputs`, `execution_count`, metadata de celdas, orden de claves JSON y
   metadata transitoria.
4. Se separaron cambios en código, configuración, parámetros e información.
5. Cuando el notebook solo clona y ejecuta el repositorio, se distinguió entre:
   - lo acreditado directamente por la celda, como rama, commit, comando y argumentos;
   - lo que exige inspeccionar el árbol Git fijado;
   - lo que no puede reconstruirse porque la rama era móvil y no se fijó un commit.
6. Los resultados se cruzaron con runners, manifiestos, sesiones, recibos HF, logs y documentación.
7. Las razones se registraron solamente cuando estaban documentadas. El objetivo aparente de una celda no se
   promovió a justificación histórica sin otra evidencia.

## Clasificación de evidencia

| Nivel | Significado |
|---|---|
| A — alta | Notebook más artefacto directo, o notebook más artefacto versionado y documentación concordante. |
| M — media | Notebook y documentación concordante, pero falta parte del artefacto original o su procedencia completa. |
| B — baja | Solo queda el notebook, metadata genérica o una inferencia de secuencia sin resultado conservado. |

Además se usan tres clases de afirmación:

- **Hecho verificado:** sale directamente de un archivo, hash, commit o recibo identificado.
- **Inferencia:** interpretación explícita y limitada de varias fuentes concordantes.
- **Pendiente:** no hay evidencia suficiente para completar el campo.

La metadata de Kaggle identifica el kernel, el archivo, la imagen y el tipo de máquina, pero **no contiene el
número histórico de versión**. Por eso acredita que las carpetas pertenecen al mismo kernel, no que `vN` sea por
sí sola la revisión N. Esa asociación depende de la extracción histórica.

## Estructura

- [`versiones.md`](versiones.md): cronología explicada v1–v41.
- [`diferencias.md`](diferencias.md): las 40 transiciones consecutivas.
- [`contradicciones.md`](contradicciones.md): faltantes, duplicados y límites de interpretación.
- [`versiones.csv`](versiones.csv): una fila normalizada por versión.
- [`recursos.csv`](recursos.csv): ejecuciones y sesiones con tiempos y recursos comprobables.
- [`evidencia.csv`](evidencia.csv): índice de notebooks, commits, artefactos, HF, W&B y documentos.

## Convenciones importantes

- `v32` y `v32b` son dos extracciones parciales de **la misma ejecución**, no dos entrenamientos.
- El adapter `dc41674a` producido por v32+v33 fue el primer adapter completo, pero v39 lo reemplazó por
  `770c9fac` como adapter definitivo.
- v40 es evaluación de práctica posterior a la adopción de v39. No se usó para elegir prompt, checkpoint ni
  hiperparámetros.
- Los resultados de práctica se reportan como medición, no como evidencia de generalización. El conjunto de
  desarrollo se usó para seleccionar; el test secreto no fue inspeccionado.
- `gpu_horas_fisicas` es, cuando aparece, una derivación de reloj de pared por cantidad de GPU. Nunca representa
  consumo real de cuota de Kaggle. El campo `cuota_real` queda vacío salvo que una fuente la registre para esa
  ejecución.
- La fecha de modificación de los archivos externos corresponde a su descarga, no a la ejecución histórica.

## Resultado histórico resumido

La secuencia pasó por cuatro etapas:

1. v1–v9 estabilizaron inferencia, tamaño de lote, LoRA, reanudación y carga solo texto.
2. v10–v24 compararon el buscador e integraron `AgentePrompt` y la nueva entrega; es el tramo con más evidencia
   faltante.
3. v25–v35 seleccionaron `procedimiento-v1`, corrigieron la pérdida supervisada y produjeron el adapter
   `dc41674a`.
4. v36–v41 probaron tres remedios para el aviso, adoptaron v39 (`770c9fac`), reportaron su práctica en v40 y
   rehicieron en v41 la comparación de buscadores perdida de v22.

La configuración final documentada es adapter v39 + `procedimiento-v1` + buscador `baseline`, lote 4. Fue
validada en T4, no en A100-40GB.
