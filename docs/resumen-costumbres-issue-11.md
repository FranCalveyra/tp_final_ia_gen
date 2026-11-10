# Resumen de costumbres para compartir

Actualizado el 2026-11-10 (corrección de #10-#12). Detalle e IDs en [`costumbres.md`](../costumbres.md) y [`historial_costumbres.json`](../historial_costumbres.json).

## Costumbres confirmadas

1. **Avisar al voluntariado sólo a pacientes de 75 años o más sin acompañante.**
   Es una regla determinista, no una preferencia: `sí` si y sólo si edad ≥ 75 y no hay acompañante en la guardia. Los 17 operadores sin desvíos la cumplen en todas sus filas (185/185 `sí`, 141/141 `no` en mayores acompañados, 1.348/1.348 `no` en menores). El umbral y la condición salen de una búsqueda sobre las filas limpias, no están fijados en el código.

2. **Con obra social y prepaga, verificar sólo la prepaga.**
   Si la prepaga está activa se registra y notifica según las normas; si está inactiva se registra `sin_cobertura` y no se verifica la obra social (45/45 filas; una sola verificación en las 193 trazas con dos credenciales). La norma no fija cuál elegir: es costumbre, no regla escrita.

## Desvíos

Todo el `ruido` (160 filas) viene de `OP-04`, `OP-11` y `OP-17`, que se apartan de ambas costumbres en las dos direcciones. Es variación individual, no otra costumbre.

## Evidencia de las trazas

Las 2.000 trazas de `historial/trazas.jsonl` coinciden con las 2.000 filas del historial: IDs, operador y ficha final. Las normas originales siguen siendo la fuente para decidir qué es un error. Las 800 trazas de referencia de `entrenamiento/trazas.jsonl` sólo se usaron como verificación cruzada (74/74, 67/67, 659/659 para el voluntariado; 20/20 `sin_cobertura` con prepaga inactiva; 62/62 notificaciones con terminación reservada inactiva).

## Para el prompt y los datos

- Enseñar el aviso al voluntariado como la regla completa (edad y acompañante), aplicada siempre.
- Con dos credenciales, verificar sólo la prepaga y no recurrir a la obra social si está inactiva.
- NOR-COB-04: con terminación reservada, `sin_cobertura` y notificar a la entidad aunque la afiliación esté inactiva (salvo ART).
- No enseñar la conducta de `OP-04`, `OP-11` y `OP-17`: sus desvíos son `ruido` y quedan fuera del dataset.
