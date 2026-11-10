# Resumen de costumbres para compartir

## Costumbres confirmadas

1. **No avisar al voluntariado a pacientes menores de 75 años.**
   Se compararon situaciones equivalentes por `motivo × cobertura`, usando sólo filas sin violaciones normativas. En los ocho estratos comparables, los menores tienen `no` en 1.298 de 1.415 casos (91,7%); el grupo de 75 años o más es mixto: `sí` en 175 de 306 (57,2%). Los IDs completos de respaldo y de las excepciones están en `costumbres.md` y `historial_costumbres.json`.

2. **Priorizar la prepaga cuando se presentan dos credenciales.**
   En 171 filas limpias con dos credenciales, la credencial verificada corresponde a una prepaga en 162 casos (94,7%) y a una obra social en 9. La norma no fija un orden: es una preferencia observada, no una regla escrita.

## Patrón adicional investigado

En las filas limpias de pacientes menores de 75 años, `OP-04`, `OP-11` y `OP-17` concentran 117 avisos en 225 casos (52,0%). Los otros 17 operadores no tienen avisos en 1.190 casos. Es una señal reproducible y respaldada por IDs, pero no se confirma como costumbre hospitalaria: está concentrada en un subgrupo y no representa la práctica mayoritaria. Por eso no agrega una regla ni cambia automáticamente la clasificación.

## Evidencia de las trazas

Las 2.000 trazas de `historial/trazas.jsonl` coinciden con las 2.000 filas del historial: IDs, operador y ficha final. Se usaron para corroborar las decisiones y asociar el patrón adicional con operadores; las normas originales siguen siendo la fuente para decidir qué es un error.

## Para el prompt y los datos

- Enseñar la costumbre etaria como una preferencia contextual, nunca como norma universal.
- Cuando hay dos credenciales, preferir la prepaga sólo como tendencia observada; verificar siempre la credencial realmente utilizada.
- No enseñar la conducta de `OP-04`, `OP-11` y `OP-17` como regla del hospital: conservarla como variación histórica o ejemplo de ruido.
