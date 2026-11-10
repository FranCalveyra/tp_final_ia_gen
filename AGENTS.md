# Estándares compartidos del proyecto

Este archivo es la fuente común para agentes y colaboradores. `CLAUDE.md` solo lo referencia.
La consigna en [mission.md](mission.md) y el contrato del [README](README.md) definen la entrega;
la [bitácora](docs/progreso.md) registra el estado real, las decisiones y la evidencia.

## Comunicación y trabajo

- Respondé en el idioma del usuario; en español, usá voseo natural. Sé directo y breve:
  explicá primero en palabras simples y profundizá cuando haga falta o te lo pidan.
- Hacé como máximo una pregunta por vez y detenete hasta recibir respuesta. No presentes
  menús ni alternativas exhaustivas; explicá tradeoffs solo cuando exista una decisión real.
- Antes de confirmar una afirmación técnica, avisá que la vas a verificar y revisá código,
  documentación o evidencia. Reconocé errores propios y corregilos con pruebas.
- Entendé el problema y el contrato antes de programar. Preferí soluciones simples,
  interfaces pequeñas y cambios acotados; no agregues capas sin una responsabilidad real.
- Cargá la skill instalada que corresponda antes de aplicar un workflow y decí cuál usaste.
  Delegá cuando las instrucciones aplicables lo pidan; no impongas un workflow ajeno a la tarea.
- Nunca uses Gemini, directa o indirectamente, salvo pedido explícito en la sesión actual.
- Ante una pausa, detené el trabajo y no descargues ni ejecutes nada hasta que se retome.

## Ejecución, seguridad y Git

- No hagas builds después de cambios. Una autorización explícita puede habilitar un build
  concreto, pero no habilita los siguientes ni otros consumos de infraestructura.
- No lances GPU, entrenamiento, escrituras cloud ni descargas de pesos sin autorización
  para ese alcance. Antes de ejecutar, explicá recursos, timeout y posibles costos.
- No inventes credenciales ni permisos. No leas, imprimas ni pidas secretos por chat;
  usá variables de entorno o Secrets. Solo `.env.example` lleva placeholders versionables.
- No commitees ni hagas push sin pedido. Usá Conventional Commits con scope, sin atribuciones
  AI ni `Co-Authored-By`. Cada commit debe entregar un comportamiento con sus tests y docs.
- No incluyas `.atl/`, `.env` reales, cachés, pesos ni `corridas/` en commits. Preservá cambios
  ajenos y no hagas operaciones destructivas sin autorización específica.
- Para GitHub, usá primero el MCP; recurrí a `gh` si no está disponible o no soporta la operación.

## Contrato y arquitectura

- No modifiques `admision/`: es el kit de la cátedra y será reemplazado por su copia.
- `entrega.py` conecta las fábricas exigidas. Separá buscador, generación del modelo y
  configuración; mantené Modal fuera de la lógica del agente y las dependencias pesadas lazy.
- El agente operativo usa las cinco herramientas del kit, registra ocho campos y se evalúa
  con nueve controles. Respetá el máximo de 40 llamadas por caso y devolvé conversaciones completas.
- No llames servicios externos desde el agente operativo. W&B se usa para experimentos,
  no como dependencia de inferencia. No expongas respuestas esperadas al agente.
- Armá el buscador con la carpeta recibida y sus valores de `parametros.yaml`.
  No fijes umbrales variables en código, prompts o datos para memorizarlos.
- La referencia actual usa `AgenteJSON` y dos secciones completas por búsqueda.
  `sin_comunes` es un candidato explícito, no el default: no cambies experimentos por defecto
  sin registrar decisión y comparación. El límite de turnos del coordinador no es el del entorno.

## Datos, entrenamiento y reproducibilidad

- No uses práctica ni test para entrenamiento u optimización de prompt. El conjunto RAG
  de desarrollo sirve para experimentar, pero sus métricas no prueban generalización.
- Coordiná el contrato de conversaciones con Eze/Yas; su formato propuesto no equivale
  a un acuerdo del equipo. Validar estructura no garantiza decisiones correctas ni ausencia de contaminación.
- Fijá dependencias en `uv.lock`, revisiones inmutables de HF, configuración y semilla.
  Los requirements exportados salen del lock; no mantengas versiones contradictorias a mano.
- Cada corrida conserva snapshots, hashes, código y estado Git, configuración, resultados y
  errores. Un commit con `dirty=true` no reconstruye por sí solo los cambios locales.
- No sobrescribas corridas ni recortes contexto silenciosamente. Conservá las máscaras
  assistant-only y `labels=-100` para contexto y padding; no las reemplaces al formar lotes.
- Checkpoints reanudables incluyen optimizador, scheduler y RNG. Solo prometé recuperar
  el último checkpoint completo confirmado; nunca pasos posteriores o escrituras parciales.
- La entrega final debe funcionar en A100 de **40 GB**, no 80 GB. Las pruebas locales
  comprimidas son desarrollo y sus tiempos/resultados se reportan separados.
- No des por probado el ajuste de LoRA en memoria ni la compatibilidad CUDA por tests con dobles.
  El agente aún necesita soporte explícito para cargar adapters entrenados.

## Validación y documentación continua

- Probá por la interfaz pública y mantené tests junto al comportamiento. No hagas builds
  como validación; usá el entorno CPU y los comandos documentados para el alcance autorizado.
- Distinguí siempre: **implementado**, **probado con dobles**, **validado con dependencia real**
  y **validado en GPU**. Nunca presentes fixtures sintéticos como calidad del modelo.
- Reportá el comando, conjunto de datos, configuración y limitaciones de cada medición.
  Mirá ganancias y regresiones por caso, no solo el promedio. Registrá también los intentos fallidos.
- Después de cada unidad de trabajo, actualizá [docs/progreso.md](docs/progreso.md): estado,
  decisiones y motivos, evidencia, bloqueos y próximo paso. Enlazá la guía específica;
  evitá duplicar detalles que puedan quedar desactualizados. Actualizar docs no cierra issues automáticamente.
- Las ubicaciones y nombres de Secrets no son credenciales: `huggingface` es lectura del
  modelo base y `wandb-secret` contiene `WANDB_API_KEY`. El secreto de escritura HF aún
  no está habilitado; no asumas que el token de lectura permite publicar.

## Memoria persistente (Engram, cuando esté disponible)

- Consultá `mem_context` ante referencias a trabajo previo; si no alcanza, `mem_search`
  y `mem_get_observation` para recuperar contenido completo. Buscá también contexto relevante
  al iniciar trabajo que pueda haberse realizado antes.
- Guardá inmediatamente decisiones, convenciones, preferencias, configuraciones, fixes y
  descubrimientos importantes con `mem_save`: título breve, tipo, proyecto/scope y `topic_key`
  estable para el mismo tema. Contenido: **What**, **Why**, **Where** y **Learned** si aplica.
- Antes de cerrar una sesión o dar una tarea por terminada, guardá `mem_session_summary`
  con Goal, Instructions, Discoveries, Accomplished, Next Steps y Relevant Files.
- Tras compactación, persistí el resumen y recuperá contexto antes de continuar.
  Engram complementa la bitácora Markdown: no la reemplaza ni es requisito para leer el repo.
