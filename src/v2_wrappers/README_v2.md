# Versión 2 - Taxi-v3 con wrappers
En esta versión seguimos utilizando el entorno prediseñado `Taxi-v3` de Gymnasium.
La diferencia respecto a la versión 1 es que añadimos wrappers personalizados para modificar parte de su comportamiento sin crear todavía un entorno nuevo.

# Wrappers añadidos
- CustomRewardWrapper: modifica recompensas y penalizaciones.
- CustomTimeLimitWrapper: limita la duración máxima de cada episodio.
- CustomObservationWrapper: amplía la observación con información adicional.

# Objetivo
Analizar cómo afectan estos cambios al aprendizaje del agente manteniendo el mismo entorno base.
