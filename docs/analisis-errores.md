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
