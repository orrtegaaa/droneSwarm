# Versión 2 - Taxi-v3 con wrappers personalizados

En esta versión seguimos utilizando el entorno prediseñado `Taxi-v3` de Gymnasium.

No se ha creado todavía un entorno nuevo. En su lugar, se han añadido wrappers personalizados para modificar parte del comportamiento del entorno original.

## Cambios realizados

- **CustomRewardWrapper**  
  Modifica las recompensas del entorno:
  - penaliza más las acciones inválidas
  - premia entregas rápidas

- **CustomTimeLimitWrapper**  
  Limita el número máximo de pasos por episodio.

- **CustomObservationWrapper**  
  Cambia la observación para devolver:
  - el estado original
  - los pasos restantes normalizados

## Algoritmo usado

Se ha utilizado **Q-learning** para entrenar al agente.

## Estructura

- `main_v2.py`: ejecución completa de la versión 2
- `README_v2.md`: explicación de la versión
