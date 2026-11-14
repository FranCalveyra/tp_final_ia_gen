# Auditoría CPU de trazas: requieren conversión antes de SFT

**Conclusión:** ninguno de los dos archivos crudos puede entrenarse directamente con el pipeline actual.
Hace falta convertir `pasos` a `messages` v1 y acordar procedencia/prompt. Las 800 trazas de referencia
son un candidato documentado por la cátedra; las 2.000 del historial **no son demostraciones curadas**.
Esta auditoría no certifica decisiones, contaminación ausente, calidad del modelo ni memoria GPU.

## Alcance y fuentes

Auditoría local del 2026-11-05, importación upstream `876d889`. Solo Python stdlib y módulos CPU del repo;
no se descargó tokenizador, no se cargaron pesos ni se ejecutó entrenamiento. Fuentes:
[README original, trazas y conversión](../referencias/kit-876d889/README.md),
[consigna original](../referencias/kit-876d889/mission.md),
[contrato v1 propuesto](conversaciones-entrenamiento.md),
`admision_tp/{conversaciones,preparacion,tokenizacion}.py` y `admision/ejemplo.py`.

| Archivo | Registros | SHA-256 exacto |
|---|---:|---|
| `entrenamiento/trazas.jsonl` | 800 | `adc42d2769cdb90e8509ab4fc71daf02f8c8934137315d06e9f87bd402ff7290` |
| `historial/trazas.jsonl` | 2.000 | `02a7a25bb73bfa199fe0a8a74d5a8332815b786e37056692a58167c9ec69cda8` |

## Compatibilidad real con el pipeline

| Control | 800 referencia | 2.000 historial |
|---|---:|---:|
| Aceptados crudos por `validar_archivo` | 0 | 0 |
| Error crudo: campos faltantes/desconocidos | 800 | 2.000 |
| Conversión exploratoria en memoria aceptada por `validar_registro` | 800 | 2.000 |
| Pasos mínimo / máximo | 4 / 9 | 4 / 9 |
| IDs repetidos / objetos completos duplicados | 0 / 0 | 0 / 0 |
| Orden consecutivo desde 1, triage primero, un único registro final | todos | todos |
| Acción final sin `fundamentos` igual a `ficha` | todos | todos |
| Acciones aceptadas por tipos/campos/enums de `_accion` | todas | todas |
| Argumentos `paciente_id` distintos del de la traza | 0 | 0 |
| Observaciones que empiezan `Error:` | 0 | 0 |

Los registros crudos tienen `id,paciente_id,pasos,ficha` (historial agrega `fecha,operador`),
no `version,id,origen,referencias,messages`. `preparar_corrida` valida antes de crear su salida;
`tokenizar_preparacion` vuelve a validar. No alcanza con pasar este JSONL al entrenador.

**Experimento, no conversor implementado:** para verificar la viabilidad estructural se construyó en memoria
el system actual de `mensaje_de_sistema()`, `primer_mensaje(paciente_id)`, un assistant JSON estricto por paso
y el user `Resultado de <herramienta>:\n<resultado>` correspondiente. Se omitió solo la observación posterior
a `registrar_decision`, conservando su acción. No se incluyó `ficha` como respuesta adicional.
Se envolvió en v1 con referencias a archivo/id; `origen=sintetico` para las 800 fue **un valor exploratorio**, no
una conclusión sobre cómo se generaron. Su procedencia debe acordarse/documentarse, no inventarse.
No se guardaron conversaciones ni se eligió un prompt final.

### Herramientas presentes

| Herramienta | Referencia | Historial |
|---|---:|---:|
| leer_triage | 800 | 2.000 |
| buscar_normativa | 2.457 | 6.197 |
| preguntar | 565 | 1.405 |
| verificar_afiliacion | 647 | 1.587 |
| registrar_decision | 800 | 2.000 |

Todas las búsquedas tienen dos encabezados `## `. Esto es compatible con el formato observado, **no prueba**
que las haya producido nuestro buscador ni que seleccionen evidencia suficiente. Las observaciones de
preguntas/verificaciones no se reprodujeron contra estados originales de pacientes, que no están completos
como objetos `Caso` en estos archivos. La consistencia de secuencia no demuestra corrección del oficio.

## Relación con historial y contaminación

- `historial.csv`: 2.000 filas; mismos ids y mismo orden en 2.000/2.000. Fecha, operador y los ocho campos de ficha
  coinciden íntegramente al interpretar `sí/no` como booleanos, vacío como null y plazo como entero.
  No se reconstruyeron todos los campos narrativos del CSV ni se clasificaron errores/ruido.
- No hay ids de traza compartidos entre referencia/historial ni con `practica/casos.jsonl`.
- Sí se repiten identificadores de paciente: 157 distintos entre referencia e historial; 2 entre referencia y
  práctica, y 10 entre historial y práctica. **Una colisión de identificador no demuestra el mismo caso.**
- Comparando el texto completo del primer triage: cero duplicados dentro de cada archivo, cero coincidencias
  entre ambos y cero con los 50 triages generados por `nota_de_triage(cargar_casos(practica/casos.jsonl))`.
- Estos controles exactos **no descartan** variantes/paráfrasis, duplicados semánticos ni contaminación indirecta.
  El test secreto no está disponible. No se abrieron las fichas esperadas ni se usaron para curar/entrenar.

Las 2.000 trazas reflejan fichas de operadores, no correcciones certificadas: conservar coincidencia con CSV
es fidelidad de importación, no garantía de que sean buenos ejemplos. Sigue pendiente curación con Eze/Yas.

## Parámetros variables y tamaño

En referencia cambian nueve parámetros observados: P-ADM-1/2/3, P-COB-2 y P-TDE-1/2/3/4/5.
P-COB-1 aparece fijo en 3 días. En historial todos los valores observados están fijos. No se detectaron
valores distintos del mismo parámetro dentro de una traza (comparación textual de asignaciones `P-… = …`).
Ejemplos de variación agregada: P-ADM-1 y P-ADM-3 toman niveles 1/2/3; P-COB-2 toma dígitos 0–9;
P-TDE-3 toma 10/15/20/30 min. La variación coincide con lo anunciado por el README original.

**No regenerar indiscriminadamente búsquedas contra `normas/original`:** podría cambiar los valores que
sustentan la ficha de referencia. Un conversor mínimo conserva observaciones; adaptar a otro buscador requiere
preservar la configuración normativa de cada traza y volver a validar decisiones, no solo reemplazar texto.

| Caracteres (no tokens), mínimo / mediana / máximo | Referencia | Historial |
|---|---:|---:|
| Conversación exploratoria, suma de `content` | 7.474 / 11.026 / 15.186 | 7.483 / 11.044 / 15.156 |
| Una observación de búsqueda | 1.182 / 2.242 / 3.303 | 1.182 / 2.242 / 3.303 |

Tamaños con el system actual, JSON de acción `ensure_ascii=False` y espacios por defecto; un prompt distinto
los cambia. No son tokens, no garantizan entrar en 8.192 tokens ni estiman VRAM. La tokenización real sigue
pendiente; el pipeline rechaza excedidos sin recortarlos y mantiene supervisión assistant-only.

## Evidencia de ejecución y límites

Se ejecutaron scripts inline con `.venv/bin/python -` usando `json,csv,hashlib,collections,re,statistics`,
`validar_archivo`, `validar_registro`, `_accion`, `mensaje_de_sistema`, `primer_mensaje`, `cargar_casos` y
`nota_de_triage`. Solo se imprimieron métricas agregadas en el análisis final. Los scripts no dejaron datasets.

Chequeo público reproducible de rechazo crudo (produce un informe por línea, salida esperada 1):

```bash
.venv/bin/python -m scripts.validar_conversaciones entrenamiento/trazas.jsonl
.venv/bin/python -m scripts.validar_conversaciones historial/trazas.jsonl
```

Tests CPU pertinentes:

```bash
env PYTHONPATH=tests .venv/bin/python -m unittest test_conversaciones test_preparacion test_tokenizacion -q
```

Resultado: **18 ejecutados, 17 pasan, 1 opcional omitido** (0,253 s). Son tests de contrato/pipeline, no calidad.
Primer intento usando `tests.test_preparacion`/`tests.test_tokenizacion` falló por import interno
`test_conversaciones` fuera de `sys.path`; se corrigió la invocación con `PYTHONPATH=tests`, no código.
Un primer control CSV también falló al asumir booleanos numéricos; se repitió con el formato real `sí/no`.

## Próximo cambio mínimo

1. Acordar con Eze/Yas la procedencia del conjunto de 800 y la selección del historial; confirmar con Martu el
   prompt final (el contrato v1 todavía figura como propuesto).
2. Implementar conversor determinista por interfaz pública: preservar llamadas/observaciones, quitar únicamente
   la observación terminal, emitir referencias trazables y validar el archivo completo. Agregar tests junto al cambio.
3. Congelar datos/config/semilla, tokenizar sin truncar y recién entonces solicitar el alcance GPU para humo real
   y entrenamiento. A100-40GB, publicaciones y tiempos/costos requieren autorización propia.

No se implementó conversor, no se resolvió la curación y no se cerró #3 con esta auditoría.
