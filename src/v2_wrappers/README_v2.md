# Versión 2 - Taxi-v3 con wrappers personalizados

En esta segunda versión del proyecto seguimos utilizando el entorno prediseñado Taxi-v3 de la librería Gymnasium.  

La idea principal de esta parte de la práctica no es crear todavía un entorno nuevo, sino trabajar sobre uno ya existente y modificar algunos aspectos de su funcionamiento mediante wrappers personalizados.

De esta forma, podemos comparar el comportamiento del entorno original con una versión adaptada, manteniendo la misma base pero introduciendo cambios en recompensas, observaciones y duración de los episodios.

---

# Objetivo de esta versión

El objetivo es analizar cómo afectan pequeñas modificaciones del entorno al aprendizaje del agente usando Reinforcement Learning.

Para ello, se mantiene el problema clásico de Taxi-v3, donde un taxi debe recoger a un pasajero y llevarlo a su destino, pero se añaden ciertas mejoras para hacerlo más interesante desde el punto de vista del entrenamiento.

---

# Wrappers implementados

## 1. CustomRewardWrapper

Este wrapper modifica el sistema de recompensas del entorno original.

Cambios realizados:

- Se penalizan más las acciones inválidas.
- Se añade una recompensa extra cuando la entrega se realiza en pocos pasos.

Con esto se busca que el agente aprenda a actuar de forma más eficiente y cometa menos errores.

---

## 2. CustomTimeLimitWrapper

Este wrapper limita el número máximo de pasos permitidos en cada episodio.

Si el agente supera ese límite sin completar la tarea, el episodio finaliza automáticamente.

Esto evita episodios demasiado largos y ayuda a que el entrenamiento sea más estable.

---

## 3. CustomObservationWrapper

Este wrapper modifica la observación que recibe el agente.

En lugar de devolver solo el estado original del entorno, ahora devuelve:

- El estado original.
- Los pasos restantes normalizados entre 0 y 1.

De esta forma, el agente dispone de algo más de información durante el aprendizaje.

---

# Algoritmo utilizado

Para entrenar al agente se ha utilizado el algoritmo Q-learning.

Se usa una Q-table donde se van almacenando los valores de cada estado y acción, actualizándose episodio tras episodio hasta mejorar la política aprendida.

También se aplica una estrategia epsilon-greedy, que combina:

- Exploración de acciones aleatorias al principio.
- Aprovechamiento de las mejores acciones aprendidas más adelante.

---

# Funcionamiento general del programa

El archivo principal ejecuta tres fases:

## Entrenamiento

El agente juega miles de episodios para aprender una política mejor.

## Evaluación

Se comprueba el rendimiento final midiendo:

- recompensa media,
- pasos medios,
- tasa de éxito.

## Demostración visual

Al final se muestra una simulación usando el render original de Taxi-v3 para observar el comportamiento aprendido.

---

# Estructura de archivos

- `main_v2.py` → código principal de la versión 2.
- `README_v2.md` → explicación de esta versión.

---

# Conclusión

Esta versión permite comprobar cómo pequeños cambios realizados mediante wrappers pueden influir en el aprendizaje del agente sin necesidad de crear todavía un entorno nuevo.

Además, sirve como paso intermedio entre la versión básica inicial y una futura versión personalizada más compleja.
