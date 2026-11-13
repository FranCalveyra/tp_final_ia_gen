# Mail de entrega (borrador)

Formato de [mission.md, «Cómo se envía»](../mission.md#cómo-se-envía). Un solo mail por grupo, antes del
**viernes 20/11/2026**, con todos los integrantes en copia. Los `<…>` se completan al enviarlo.

```text
Para:     Paulo Veiga, Marco Sorondo
CC:       <mails de todos los integrantes>
Asunto:   [AI Gen] TP final — Agente de admisión de guardia — Grupo <número>

Integrantes:
- <Nombre y apellido> — <mail> — <usuario de GitHub>
- <Nombre y apellido> — <mail> — <usuario de GitHub>
- <Nombre y apellido> — <mail> — <usuario de GitHub>
- <Nombre y apellido> — <mail> — <usuario de GitHub>

Repositorio de GitHub:   https://github.com/FranCalveyra/tp_final_ia_gen
Commit entregado:        <hash completo del commit o tag de entrega en main>
Modelo en Hugging Face:  https://huggingface.co/FranCalveyra/tp-admision — versión del punto de control final: <revisión de 40 hex>, carpeta final
Proyecto de W&B:         <link del proyecto francalveyra-ia-gen/tp-admision>
Presentación en Loom:    <link>
```

## Antes de mandarlo

«La cátedra corre exactamente esa versión de `entrega.py` con ese punto de control. Todos los links tienen que
estar activos y accesibles al momento de la corrección.»

- [ ] **Revisión del adapter:** la misma que `ADAPTER_ENTREGA` en `admision_tp/configuracion.py` del commit
      entregado (hoy `770c9fac84d5718b0b2cf4c40c3451cad322ac9e`, la v39).
- [ ] **Commit entregado:** en `main`, con PR mergeados; conviene un tag (p. ej. `entrega-final`) para que no se mueva.
- [ ] **Acceso de la cátedra:** invitados al repo de GitHub, al repo HF privado y al proyecto W&B (#16). Probar
      que su `HF_TOKEN` baje el adapter: sin eso, `construir_agente()` falla con 401.
- [ ] **Link de W&B:** abrirlo sin sesión de un integrante para confirmar que la cátedra lo ve. El run del SFT
      registra su URL en `estado.json` (`wandb_url`).
- [ ] **README:** el «Cómo correr la entrega» apunta al mismo adapter y lote que el commit.
- [ ] **Pre-entrega (opcional):** la consigna la permite cuando esté todo; se puede seguir mejorando hasta la final.
