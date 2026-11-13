# Análisis de errores por corrida (issue #14)

Después de cada corrida: qué controles fallan, en qué tipos de caso y **por qué** (formato, costumbre, buscador o
modelo), para decidir la pasada siguiente. Abajo está el registro de pasadas, incluidas las que no mejoraron.

Analizar la práctica es **diagnóstico y reporte**: sus conclusiones solo sirven para formular candidatos genéricos.
La selección de prompt (#13) se hace en el conjunto prompt-dev y la de checkpoint (#15) con datos permitidos;
nunca con práctica ni test ([AGENTS.md](../AGENTS.md)).

## Comandos

```bash
# Una corrida -> carpeta nueva con analisis.json (cada número con procedencia) y analisis.md
.venv/bin/python -m scripts.analizar_errores analizar \
    --corrida corridas/kaggle/v22/baseline-original-n50-l4 --salida corridas/analisis/<fecha>-<nombre>

# Dos corridas sobre los mismos casos -> comparacion.json y comparacion.md
.venv/bin/python -m scripts.analizar_errores comparar \
    --antes corridas/kaggle/v22/baseline-original-n50-l4 \
    --despues corridas/kaggle/v22/prefijos-original-n50-l4 --salida corridas/analisis/<fecha>-cmp-<nombre>
```

- **Carpetas aceptadas:** `infra/kaggle_baseline.py` y cada corrida de `infra/kaggle_evaluacion.py` (`runner.json`
  + `salida/trazas.jsonl`); el runner de prompts del #13 (`manifiesto.json` + `trazas.jsonl` + `entradas/`, de donde
  salen casos, esperadas y normas); o solo trazas, con `--version` obligatoria (sin procedencia, queda avisado).
  `--casos/--esperadas/--normas` cambian las entradas; en `comparar`, `--version-antes/--version-despues`.
- **Puntuación:** siempre con `admision.correr.puntuar` del kit. Si la carpeta ya trae `resultados.jsonl` o
  `resumen.json`, el análisis verifica que coincidan y lo informa en «Cruce». Las tasas por control se verifican
  contra `resumen.por_control` del kit.
- **Salidas:** se niega a escribir en una carpeta existente. `corridas/` está ignorada por Git.
- **Comparación:** se niega si los conjuntos de casos difieren, salvo `--permitir-interseccion` (compara los
  comunes y lo dice). Informa ganados/perdidos con prueba de signos, controles arreglados/rotos, cambios de fuente,
  conversaciones distintas y costo (caracteres de contexto, búsquedas, llamadas, segundos).
- **Tests:** `tests/test_analizar_errores.py` (corridas armadas con el `Entorno` real del kit y un buscador falso;
  sin GPU).

## Reglas de fuente

Cada control fallado recibe **una** fuente, con esta precedencia:

1. **formato**: el caso terminó sin ficha (el kit pone los 9 controles en falso). Subtipo según la conversación:
   `sin_json` (última respuesta sin JSON: tope de respuestas sin JSON), `tope_llamadas` (40 del kit),
   `guarda_contexto` (mensaje `[agente] Caso terminado sin registrar decisión` de `AgentePrompt`),
   `registrar_decision_invalida`, `tope_turnos` o `sin_conversacion`. También es formato un control de cobertura
   cuando todas las llamadas a `preguntar` o `verificar_afiliacion` del caso fueron inválidas
   (`argumentos_invalidos:<herramienta>`).
2. **costumbre**: `aviso_al_voluntariado` (costumbre no escrita, #11) siempre; y cobertura/entidad/verificación
   cuando, con dos credenciales y sin accidente laboral, el agente verificó la credencial que la costumbre no elige
   (`eleccion_de_credencial`). Ninguna búsqueda puede arreglarlas.
3. **buscador**: falta en todas las observaciones de búsqueda del caso alguna sección rectora del control **y** el
   agente hizo al menos una búsqueda relevante para esa familia (la pidió y no llegó).
4. **modelo**: lo demás. `busco_y_decidio_mal` si hubo búsqueda relevante (las rectoras estaban o el control no
   tiene sección); `no_busco` si no hubo ninguna búsqueda relevante para la familia del control (aunque falte la
   rectora: el buscador no tuvo oportunidad).

**Secciones rectoras por control.** Reutiliza `secciones_rectoras` del #8 (`scripts/generar_conversaciones.py`),
repartida por familia: registro → NOR-ADM-01 (+ ADM-03 con pasaporte/cédula); número de documento → ADM-01; box →
ADM-01 (+ ADM-02 por edad ≥ P-ADM-2); circuito → la sección del circuito (TDE-01/02/03; ninguna si no hay
circuito); plazo → esa sección + TDE-04 si llegó en ambulancia; cobertura, entidad y verificación → COB-02 por ART
o COB-04 por terminación reservada (como el #8) y, si el #8 no lista ninguna, la que decide el caso común: COB-02
si hay cobertura que verificar, COB-01 si no presentó credencial y COB-03 si el resultado es sin cobertura. La
credencial elegida con dos credenciales y la actividad de la afiliación salen del caso y de la verificación
esperada: se usan solo para diagnosticar, después de jugar.

**Búsqueda relevante** para una familia (ADM, TDE, COB): la observación trajo alguna sección de esa familia o la
consulta, sin tildes, tiene alguna raíz de la lista `PALABRAS` del script (p. ej. `registr`, `box`, `document`;
`plazo`, `torac`, `sepsis`, `ambulancia`; `cobertura`, `afilia`, `art`, `trabaj`).

**Tipos de caso** (desde el archivo de casos que usó la corrida): circuito esperado según el triage, llegada,
credenciales (ninguna/obra social/prepaga/dos), accidente laboral, edad ≥ P-ADM-2, acompañante, puede responder,
documento, nivel y número con terminación reservada (P-COB-2). El reporte ordena los grupos de ≥ 3 casos por
proporción sin resolver.

### Límites

- Las fuentes son **heurísticas**, no causalidad: «buscador» dice que la evidencia no llegó, no que con ella el
  modelo habría acertado; «modelo/busco_y_decidio_mal» no distingue entre no leer la sección y leerla mal.
- La relevancia por palabras puede dar falsos positivos (una consulta mixta «dolor torácico y registro» cuenta
  para ADM y TDE) y falsos negativos (sinónimos fuera de la lista). No se midió su precisión contra etiquetas
  humanas.
- Las rectoras de cobertura en el caso común son una ampliación propia del #8, documentada arriba.
- Un control mal decidido por arrastre (p. ej. entidad a notificar después de verificar mal) se cuenta en su propio
  control; no hay atribución de causa raíz entre controles.
- `segundos` por caso es el momento de la última llamada desde el inicio del lote (los casos corren a la vez);
  el costo por caso comparable es `jugar_lote_por_caso` de `tiempos.json`. T4 no es comparable con A100.
- Supone el protocolo JSON (`AgenteJSON`/`AgentePrompt`) para contar respuestas sin JSON.

## Registro de pasadas

Todas: Gemma 4 E4B **sin entrenar**, Kaggle T4, normas original, 50 casos de práctica (salvo v24). Salidas en
`corridas/analisis/` (local, ignorado por Git); comandos exactos en cada `analisis.json`/`comparacion.json`.

### 2026-11-10 · Pasada 1: v22 baseline (el «antes» de #13 y #15)

- **Corrida:** `corridas/kaggle/v22/baseline-original-n50-l4`, commit `4e3613a` limpio, `AgenteJSON` (el prompt no
  quedó registrado en `runner.json`), buscador `baseline`, lote 4. Salida:
  `corridas/analisis/2026-11-10-v22-baseline/`.
- **Resultado:** 10/50, parcial 0,831, 0 sin decisión, 19.954 caracteres/caso, 1,34 búsquedas/caso.
  Cruce: tasas iguales a `resumen.por_control` del kit; `resultados.jsonl` y `resumen.json` previos coinciden.

| Control | Fallas | Tasa | Formato | Costumbre | Buscador | Modelo: no buscó | Modelo: buscó y decidió mal |
|---|---|---|---|---|---|---|---|
| registro | 12 | 24 % | 0 | 0 | 8 | 3 | 1 |
| numero_documento | 0 | 0 % | 0 | 0 | 0 | 0 | 0 |
| box_antes_del_registro | 14 | 28 % | 0 | 0 | 10 | 3 | 1 |
| circuito | 1 | 2 % | 0 | 0 | 0 | 1 | 0 |
| plazo_circuito_min | 4 | 8 % | 0 | 0 | 2 | 2 | 0 |
| cobertura | 10 | 20 % | 0 | 1 | 2 | 6 | 1 |
| entidad_a_notificar | 9 | 18 % | 0 | 0 | 3 | 2 | 4 |
| aviso_al_voluntariado | 14 | 28 % | 0 | 14 | 0 | 0 | 0 |
| verificacion | 12 | 24 % | 0 | 1 | 2 | 4 | 5 |
| **total** | 76 | | 0 | 16 | 27 | 21 | 12 |

- Rectoras no recuperadas en fallas del buscador: NOR-ADM-01 16, ADM-02 7, COB-02 6, ADM-03 5, TDE-04 2, COB-04 1.
- Tipos de caso con más fallas (grupos ≥ 3): accidente en el trayecto 0/5 resueltos (3,4 fallas/caso), accidente
  trabajando 0/5 (2,0), pasaporte 0/3, nivel 4 0/6, terminación reservada 0/5, credencial de obra social 0/6, sin
  acompañante 2/23 (1,87), cédula 1/11.
- **Hallazgos (patrones genéricos):** (1) no hay fallas de formato; (2) la mayor fuente es el buscador: en 23 de
  sus 27 fallas el caso tuvo una consulta que mezcla temas de dos familias («circuito y registro») y las 2 secciones
  devueltas son de una sola, así que el registro y el box quedan sin evidencia; (3) le sigue el modelo que no busca la familia del control (cobertura y
  verificación sobre todo; 1,34 búsquedas por caso); (4) el aviso al voluntariado es un 18 % de las fallas y es pura
  costumbre; (5) los accidentes laborales (ART) no se resuelven nunca.
- **Decisión:** es la línea de base del análisis; v23/v24 y las corridas de #13/#15 se comparan contra ella.

### 2026-11-10 · Pasada 2: v22 `prefijos` contra `baseline` (decisión del #6)

- **Corrida:** `corridas/kaggle/v22/prefijos-original-n50-l4` (mismo commit, solo cambia el buscador). Salidas:
  `corridas/analisis/2026-11-10-v22-prefijos/` y `2026-11-10-cmp-v22-prefijos-vs-baseline/`.
- **Resultado:** 6/50 (parcial 0,796). Gana 3 casos, pierde 7 (signos p=0,344). 43 conversaciones distintas.
  Fuentes totales (formato/costumbre/buscador/modelo): 0/16/27/33 → 2/16/30/44. Modelo «no buscó» 21 → 33.
  NOR-ADM-01 no recuperada 16 → 22.

| Control | Arreglados | Rotos | Fuentes antes (F/C/B/M) | Fuentes después (F/C/B/M) |
|---|---|---|---|---|
| registro | 1 | 8 | 0/0/8/4 | 0/0/12/7 |
| box_antes_del_registro | 8 | 10 | 0/0/10/4 | 0/0/10/6 |
| circuito | 0 | 2 | 0/0/0/1 | 0/0/0/3 |
| plazo_circuito_min | 0 | 4 | 0/0/2/2 | 0/0/4/4 |
| cobertura | 1 | 2 | 0/1/2/7 | 0/1/1/9 |
| entidad_a_notificar | 3 | 3 | 0/0/3/6 | 1/0/2/6 |
| verificacion | 2 | 2 | 0/1/2/9 | 1/1/1/9 |

- Costo: 19.954 → 20.689 caracteres/caso, 1,34 → 1,40 búsquedas, 15,16 → 15,73 s/caso de lote.
- Las 2 fallas de formato son un único caso con `preguntar` de argumentos inválidos (entidad y verificación).
- **Hallazgo:** `prefijos` mejora el RAG aislado pero con el agente pierde evidencia de registro (ADM-01) y de
  plazos; el cambio de conversaciones también cambia qué busca el modelo. Resultado negativo, no concluyente.
- **Decisión (ya tomada en el #6, ahora con desglose):** default `baseline`; `prefijos` queda documentado como
  intento fallido.

### 2026-11-10 · Pasada 3: v23 `AgentePrompt` contra v22 `AgenteJSON` (equivalencia)

- **Corrida:** `corridas/kaggle/v23/plomeria-agenteprompt-baseline-original-n50-l4` (runner del #13, commit
  `5d4b657` limpio, prompt `baseline` v1 sha256 `944ae7f6f52d…`, propósito plomería). Salidas:
  `corridas/analisis/2026-11-10-v23-agenteprompt/` y `2026-11-10-cmp-v23-vs-v22-baseline/`.
- **Resultado:** 10 → 10, **0 diferencias**: ningún caso ganado/perdido, ningún control ni fuente cambia y las 50
  conversaciones, logs y fichas son idénticos. Cruce contra sus propios `resultados.jsonl`/`resumen.json`: coincide.
  Solo cambia el tiempo de lote (15,16 → 14,86 s/caso).
- **Decisión:** `AgentePrompt` con el prompt `baseline` reproduce `AgenteJSON`; v22 vale como «antes» para corridas
  hechas con `AgentePrompt`.

### 2026-11-10 · Pasada 4: v24 entrega nueva (parcial)

- **Corrida:** `corridas/kaggle/v24/salida/trazas.jsonl`, 10 casos (pev-001 a pev-010), **sin `runner.json`**: sin
  procedencia de código, prompt ni modelo (se analizó con `--version original`). Salidas:
  `corridas/analisis/2026-11-10-v24-parcial/` y `2026-11-10-cmp-v24-vs-v22-baseline/` (con
  `--permitir-interseccion`).
- **Resultado:** 1/10; contra los mismos 10 casos de v22: 0 diferencias (conversaciones idénticas).
- **Decisión:** solo confirma que la entrega nueva no cambia decisiones en esos 10 casos; no es una medición
  completa.

### 2026-11-11 · Pasada 5: #13 `procedimiento-v1` contra `baseline` en el conjunto de desarrollo

- **Corridas:** `corridas/kaggle/v25/prompt-dev-f1/{baseline,procedimiento-v1}-{n1,n2}` (runner del #13, commit
  `f4d7300` limpio, buscador `baseline`, `datos/prompt-dev/v1`, **no práctica**). Salidas:
  `corridas/analisis/2026-11-11-cmp-v25-procedimiento-vs-baseline-{n1,n2}/`. Las corridas descargadas no traían
  `entradas/normas/`; se copió `datos/prompt-dev/v1/entradas/n<k>/normas` tras verificar que cada archivo coincide
  con el sha256 del manifiesto de la corrida.
- **Resultado:** n1 4 → 4 (gana 3, pierde 3), n2 4 → 5 (gana 2, pierde 1); signos p=1,0 en ambas. El baseline en el
  dev repite el perfil de la pasada 1: buscador y modelo «no buscó» como fuentes principales, aviso solo costumbre.

| Control | Arreglados n1/n2 | Rotos n1/n2 | Buscador antes → después (n1+n2) | Modelo antes → después (n1+n2) |
|---|---|---|---|---|
| registro | 4/4 | 0/2 | 7 → 0 | 6 → 7 |
| box_antes_del_registro | 4/6 | 2/0 | 6 → 0 | 10 → 8 |
| circuito | 0/2 | 0/0 | 0 → 0 | 2 → 0 |
| plazo_circuito_min | 2/4 | 3/1 | 5 → 0 | 5 → 8 |
| cobertura | 0/3 | 1/1 | 1 → 2 | 7 → 4 |
| entidad_a_notificar | 0/1 | 3/1 | 1 → 1 | 4 → 6 |
| aviso_al_voluntariado | 0/0 | 0/0 | 0 → 0 | 0 → 0 (8 → 8 costumbre) |
| verificacion | 1/0 | 4/3 | 1 → 1 | 6 → 11 |

- Costo: ~19.000 → ~53.000 caracteres/caso, 1,35 → 3,6 búsquedas, 4,8 → 6,75 llamadas.
- **Hallazgo:** el procedimiento hace lo que se buscaba en el RAG: las fallas atribuibles al buscador pasan de 21 a
  4 (registro, box y plazo quedan en 0), porque cada tema se busca por separado. Pero el modelo **decide mal con la
  evidencia delante**: suben las fallas «buscó y decidió mal» en `verificacion` (6 → 11), `plazo` y `entidad`. El
  aviso no se mueve (es costumbre; B no la menciona). La ganancia neta en casos es ruido.
- **Para #15 (Fran):** el prompt solo ya no puede arreglar lo que queda: con la evidencia correcta en contexto, el
  modelo base no aplica bien verificación, plazo y entidad, ni la costumbre del aviso. Eso es lo que tienen que
  enseñar las conversaciones de #8 (decidir y verificar con la sección recuperada, y la costumbre). Comparar el
  checkpoint contra estas corridas con `comparar`.
- **C (`procedimiento-costumbres-v1`) no se analizó:** falla en GPU de forma reproducible (ver
  [optimizacion-prompt.md](optimizacion-prompt.md#resultados-kaggle-v25v27)).

### 2026-11-11 · Pasada 6: fila `solo_prompt` contra `sin_entrenar` en práctica (solo reporte)

- **Corridas:** `corridas/kaggle/v25/evaluacion-actual/sin_entrenar-<v>` (prompt `baseline`, commit `f4d7300`) contra
  `corridas/kaggle/v27/evaluacion-solo-prompt/solo_prompt-<v>` (prompt `procedimiento-v1`, commit `a5745f0`), v =
  original, b, c; buscador `baseline`, lote 4, T4. Salidas: `corridas/analisis/2026-11-11-cmp-v27-solo-prompt-vs-sin-entrenar-<v>/`.
  El prompt ya estaba elegido en el dev (pasada 5): esto no se usó para elegir.
- **Resultado:** original 10 → 14 (gana 7, pierde 3), b 10 → 16 (9/3), c 7 → 12 (7/2); signos p=0,34 / 0,15 / 0,18
  por versión. Las tres versiones comparten pacientes, así que no se suman como independientes.

| Control (original) | Arreglados | Rotos | Fuentes antes (F/C/B/M) | Fuentes después (F/C/B/M) |
|---|---|---|---|---|
| registro | 4 | 1 | 0/0/8/4 | 0/0/0/9 |
| box_antes_del_registro | 11 | 2 | 0/0/10/4 | 0/0/0/5 |
| plazo_circuito_min | 3 | 0 | 0/0/2/2 | 0/0/0/1 |
| cobertura | 4 | 5 | 0/1/2/7 | 0/3/0/8 |
| entidad_a_notificar | 3 | 8 | 0/0/3/6 | 0/2/0/12 |
| aviso_al_voluntariado | 0 | 0 | 0/14/0/0 | 0/14/0/0 |
| verificacion | 3 | 11 | 0/1/2/9 | 0/3/0/17 |

- **Hallazgo:** el mismo patrón que en el dev, más marcado. Las fallas del buscador pasan a 0 en registro, box y
  plazo (en las tres versiones), y box gana 11–12 casos por versión. Lo que empeora es decisión del modelo con la
  evidencia correcta: verificación (7–11 rotos por versión) y entidad a notificar (5–8). El aviso queda en 14
  fallas de costumbre en las tres. **Línea de base para #15:** comparar `solo_entrenamiento` y `ambos` contra
  estas corridas con `comparar`; el entrenamiento tiene que bajar «modelo» en verificación/entidad y «costumbre»
  en el aviso sin devolver las fallas del buscador.

### 2026-11-12 · Pasada 7: fila `ambos` contra `solo_prompt` en práctica (solo reporte)

- **Corridas:** `corridas/kaggle/v27/corridas/evaluacion-solo-prompt/solo_prompt-<v>` (base + `procedimiento-v1`,
  commit `a5745f0`) contra `corridas/kaggle/v34/x/evaluacion-entrenado/ambos-<v>` (adapter
  `FranCalveyra/tp-admision@dc41674a/final` + `procedimiento-v1`, commit `3adf12e`); buscador `baseline`, lote 4,
  T4. Salidas: `corridas/analisis/2026-11-12-cmp-v34-ambos-vs-solo-prompt-<v>/`. El adapter no se eligió con esto.
- **Resultado:** original 14 → 30 (gana 16, **pierde 0**), b 16 → 29 (13/0), c 12 → 33 (21/0); signos p < 0,001
  en las tres (versiones no independientes).

| Control (original) | Arreglados | Rotos | Fuentes antes (F/C/B/M) | Fuentes después (F/C/B/M) |
|---|---|---|---|---|
| registro | 9 | 0 | 0/0/0/9 | 0/0/0/0 |
| box_antes_del_registro | 5 | 1 | 0/0/0/5 | 0/0/0/1 |
| cobertura | 8 | 1 | 0/3/0/8 | 0/0/1/3 |
| entidad_a_notificar | 12 | 1 | 0/2/0/12 | 0/0/0/3 |
| aviso_al_voluntariado | 0 | 0 | 0/14/0/0 | 0/14/0/0 |
| verificacion | 20 | 0 | 0/3/0/17 | 0/0/0/0 |

- **Hallazgo:** el entrenamiento hace lo que la pasada 6 pedía: verificación pasa de 17–20 fallas «modelo» a 0 y
  entidad a notificar de 11–14 a 0–4, sin devolver las fallas del buscador (1 en cobertura). Las 3 versiones de
  normas mejoran igual: aprendió a aplicar normas, no a memorizarlas.
- **El aviso no se mueve, y la causa es concreta:** las 14 fallas por versión son **todos** los pacientes de 75 años
  o más sin acompañante (esperada `true`), y el modelo responde `false`. Con o sin adapter, nunca responde `true`
  en esos casos. Los datos de entrenamiento no tienen el problema: en `v3-procedimiento` la regla se cumple sin
  excepciones (302 `true` = todos los 75+/sin acompañante; 2.598 `false`). La nota de triage tiene el mismo formato
  en ambos lados («N años», «Acompañante: ninguno»). Hipótesis: con 10 % de positivos y un solo token en juego, 1
  época con LoRA r=8 solo en q/v no aprende la regla condicional y queda en la clase mayoritaria; la pérdida media
  (~0,005) no lo muestra. **Sin verificar:** falta ver si el adapter también responde `false` en conversaciones de
  entrenamiento o del conjunto de desarrollo (requiere GPU).
- **Para #15:** confirmar la hipótesis en el conjunto de desarrollo antes de cambiar el entrenamiento; la práctica
  no puede elegir el remedio.
- **Confirmado en el conjunto de desarrollo (Kaggle v35, commit `bea206b`):** `procedimiento-v1` + adapter resuelve
  26/40 (11 + 15; sin adapter, v25: 9/40). El aviso queda en 32/40: el modelo **no responde `true` en ninguno de los
  40 casos**, y los 8 que fallan son los 8 mayores de 75 sin acompañante. Esos 8 fallan **solo** por el aviso; los
  otros controles: verificación y entidad 40/40, box 38, cobertura 37, registro 39, plazo 39. Es colapso a la clase
  mayoritaria, no un problema de formato ni del dato. Salidas en `evaluaciones/dev/v35-entrenado`.
- **Límite declarado:** el problema se **vio** en práctica (esta pasada es un reporte sobre práctica). Para no
  usar la práctica como selector, la hipótesis se confirmó en el conjunto de desarrollo, el remedio (`v4-aviso`)
  se arma solo con datos de entrenamiento, y la decisión de adoptar el adapter de la v36 se toma solo con el
  dev; la práctica de la v36 se corre después y no cambia la decisión. Aun así, saber que existía la falla vino
  de mirar práctica: lo declaramos en vez de ocultarlo.

### 2026-11-12 · Pasada 8: refuerzo del aviso (v36) contra el adapter de la entrega (v35), conjunto de desarrollo

**No funcionó; el adapter de la entrega sigue siendo `dc41674a`.** La v36 siguió entrenando ese adapter
(`adapter_inicial`) 1 época sobre `datos/conversaciones/v4-aviso` (633 conversaciones: 302 con aviso `true`, 231
mayores con acompañante, 100 del resto; lr 1e-4) y se midió en el dev con la misma configuración que v35.
Decisión tomada solo con el dev; la práctica de la v36 no se corrió.

| Dev, 40 casos | v35 (`dc41674a`) | v36 (`22ee5596`) |
|---|---:|---:|
| Resueltos | **26** | 23 (ganados 7, perdidos 10) |
| Aviso correcto | **32** | 26 |
| 75+ sin acompañante (aviso `true`) | 0/8 | **6/8** |
| Resto (aviso `false`) | **32/32** | 20/32 (10 menores de 75 y 2 mayores con acompañante con `true`) |
| Otros controles que cambiaron | — | registro 1 arreglado y 1 roto; plazo 1 roto |

- **Causa:** con 47,7 % de positivos (302/633) el modelo pasó de «siempre `false`» a decir `true` seguido, sin aprender la
  condición (edad **y** sin acompañante). La caída de 26 a 23 es casi toda del aviso (6 arreglados, 12 rotos).
- **Por qué cuesta:** el aviso es **un token** entre los ~350 supervisados de cada conversación (1.015.072 / 2.900). La pérdida media lo
  diluye (v36 arrancó en 0,0069 y terminó en 0,0033: el adapter ya «sabía» todo lo demás), así que el gradiente
  sobre esa decisión es chico y cambiar la proporción de positivos mueve el sesgo, no la regla.
- **Confirma en GPU** que `adapter_inicial` funciona: la pérdida del paso 1 fue 0,0069 (un LoRA nuevo arranca ~0,44).
- **Qué probaría con más GPU** (no se hizo): una proporción de positivos cercana a la real con más épocas,
  peso extra en la pérdida del token del aviso, o DPO con pares que difieran solo en el aviso. Cada intento son
  ~3 h de T4; la cuota alcanza, pero sin garantía de que el siguiente gane al actual.
- Corridas en `evaluaciones/dev/v36-aviso`; recibos del entrenamiento en `evaluaciones/entrenamiento/v36-aviso`
  (W&B `6d8edb168beb4e9b`, checkpoints 20/40/60/80 y `final` en HF).

### 2026-11-13 · Pasada 9: peso 20 en el token del aviso (v38) contra el adapter de la entrega (v35), conjunto de desarrollo

**No se adopta: empata en resueltos y rompe más controles; el adapter de la entrega sigue siendo `dc41674a`.**
La v38 siguió entrenando ese adapter 1 época (142 pasos) sobre `datos/conversaciones/v5-aviso-peso` (1.133
conversaciones, 27 % con aviso `true`) con `peso_aviso` 20 y lr 1e-4, y se midió en el dev con la misma
configuración que v35 y v36 (`procedimiento-v1`, n1 + n2). La regla era adoptar solo si superaba 26/40;
la práctica de la v38 no se corrió.

| Dev, 40 casos | v35 (`dc41674a`) | v36 (`22ee5596`) | v38 (`17e9fe3d`) |
|---|---:|---:|---:|
| Resueltos | 26 | 23 | 26 (ganados 3, perdidos 3 contra v35) |
| Aviso correcto | **32** | 26 | 30 |
| 75+ sin acompañante (aviso `true`) | 0/8 | **6/8** | 1/8 |
| Falsos `true` (aviso `false` esperado) | 0 | 12 | 3 |
| Otros controles que fallan | — | — | cobertura 3, registro 1, `box_antes_del_registro` 1 |

- **Casos:** ganó `pd-n2-002`, `pd-n1-008` y `pd-n1-012`; perdió `pd-n2-006`, `pd-n2-009` y `pd-n2-017`.
  Sumar y restar tres casos queda dentro del ruido de una sola corrida de 40 casos.
- **Lectura:** el peso movió el sesgo menos que la proporción de positivos (v36: 6/8 y 12 falsos `true`; v38: 1/8 y
  3 falsos `true`) y tampoco enseñó la regla (edad **y** sin acompañante). El aviso empeora de 32 a 30 y aparecen
  fallas fuera del aviso (cobertura, registro, orden del box), que v35 no tenía.
- **Entrenamiento:** 11.750 s (~83 s/paso en T4 x2), 20,29 GB de VRAM pico, pérdida final ponderada 0,043 y media
  0,030 (no comparable con v36: el peso 20 cambia la escala de la pérdida). W&B `f9326b10342b4be4`.
- **Qué quedaría sin probar** (no se hizo): DPO con pares que difieran solo en el aviso, o un peso mucho más
  alto con la proporción real de positivos. Ya van dos intentos de ~3 h de T4 sin ganarle al adapter actual.
- Corridas en `evaluaciones/dev/v38-peso`; recibos del entrenamiento en `evaluaciones/entrenamiento/v38-peso`
  (checkpoints 20 a 140, 142 y `final` en HF).

### 2026-11-13 · Pasada 10: edad y acompañante antes de la decisión (v39) contra el adapter de la entrega (v35), conjunto de desarrollo

**Se adopta: 31/40 contra 26/40, por encima de la barra acordada antes (≥30/40, con las ganancias en los casos de 75+
sin acompañante).** La v39 siguió entrenando `dc41674a` 1 época (142 pasos) sobre `datos/conversaciones/v6-aviso-datos`
(1.133 conversaciones; el último mensaje lleva `Edad: N años. Acompañante: X.` antes del JSON), `peso_aviso` 20,
lr 1e-4. Se midió en el dev con la misma configuración que v35, v36 y v38 (`procedimiento-v1`, n1 + n2). Comando y
datos: `datos/prompt-dev/v1`, 40 casos, `evaluaciones/dev/v39-datos`. La práctica de la v39 se corrió después
(Kaggle v40) solo para reportar; ver la nota al final de esta pasada.

| Dev, 40 casos | v35 (`dc41674a`) | v38 (`17e9fe3d`) | v39 (`770c9fac`) |
|---|---:|---:|---:|
| Resueltos | 26 | 26 | **31** |
| 75+ sin acompañante (aviso `true`) | 0/8 | 1/8 | **5/8** |
| Falsos `true` (aviso `false` esperado) | 0 | 3 | 2 |

- **Controles que fallan en v39:** aviso 6, cobertura 4, `box_antes_del_registro` 3, registro 2, y 1 cada uno
  `numero_documento`, circuito, `plazo_circuito_min`, `entidad_a_notificar` y verificación. En el aviso, 1 caso
  negativo quedó sin decisión. El prefijo apareció en 38 de 40 mensajes finales.
- **Contra v35, caso por caso:** ganó `pd-n1-001`, `pd-n1-002`, `pd-n1-008`, `pd-n1-012`, `pd-n2-001`, `pd-n2-002` y
  `pd-n2-003`; perdió `pd-n1-010` (registro provisorio en vez de definitivo y box `true`; no es del aviso) y
  `pd-n2-013` (sin decisión: terminó en plena búsqueda tras 13 llamadas).
- **Lectura:** copiar los dos hechos antes de decidir sí movió la decisión (de 0/8 y 1/8 a 5/8 positivos), a diferencia
  de la proporción de positivos (v36) y del peso (v38). Quedan 3 positivos sin atrapar, 2 falsos `true` y fallas fuera
  del aviso (cobertura, orden del box). Una sola corrida de 40 casos: 5 casos de diferencia no prueban generalización.
- **Compatibilidad:** `admision_tp/agente_prompt.py` parsea con `extraer_json` de `admision/ejemplo.py`, que acepta
  texto previo sin llaves; el dev usa ese mismo camino.
- **Entrenamiento:** 11.735 s, 20,36 GB de VRAM pico, pérdida final ponderada 0,038, W&B `41ec01f292e64306`.
- Corridas en `evaluaciones/dev/v39-datos`; recibos en `evaluaciones/entrenamiento/v39-datos` (checkpoints 20 a 140,
  142 y `final` en HF).
- **Práctica (Kaggle v40), reportada y no usada para elegir:** la v39 ya estaba adoptada con el dev. `ambos` dio
  36/28/36 = 100/150 contra 30/29/33 = 92/150 del adapter anterior; el aviso pasó de 0,72 a 0,82/0,70/0,78. En la
  versión b perdió un caso neto (29 a 28). Tabla en `evaluaciones/practica/tabla-v39`.

## Uso para #13 y #15

- **Qué conclusiones se usaron para diseñar candidatos de #13** (solo patrones genéricos, sin contenido de casos
  de práctica): buscar cada tema por separado (registro, circuito, cobertura) en vez de consultas mixtas; buscar la
  cobertura y verificar antes de decidir; preguntar por la ART cuando el triage marca lesión laboral; el aviso al
  voluntariado solo se puede mejorar con la costumbre inferida del historial (#11), no con búsquedas. Ningún
  candidato copia ids, fichas, umbrales ni textos de la práctica.
- **Dónde se selecciona:** los candidatos de #13 se eligen en el conjunto prompt-dev
  ([optimizacion-prompt.md](optimizacion-prompt.md)) y los checkpoints de #15 con datos permitidos. La práctica se
  usa después, una vez, para reportar.
- **Próximas pasadas:** las corridas de #13 sobre prompt-dev y las de checkpoints de #15 se analizan con este mismo
  comando (`analizar` y `comparar` contra la pasada 1 o contra el candidato anterior), y se agregan acá aunque no
  mejoren.
