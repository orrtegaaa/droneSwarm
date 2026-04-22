## DroneSwarm v3_Custom

  Simulación básica de un dron de reparto en una cuadrícula con obstáculos

---

## Descripción

Este proyecto consiste en un programa en Python que simula el movimiento de un dron dentro de un mapa en forma de cuadrícula.

El dron parte desde una posición inicial y tiene que recorrer el entorno para completar varias entregas. En el mapa también hay obstáculos que representan edificios, por lo que no todos los caminos son válidos.

Al ejecutar el programa, primero se entrena el agente y después se muestra una ventana visual en la que puede verse cómo se mueve el dron por el mapa.

---

## Qué hace el programa

El programa incluye:

- un mapa de **8x8**
- una posición inicial para el dron
- **3 puntos de entrega**
- varios edificios bloqueando algunas casillas
- un sistema de entrenamiento
- una visualización final con **Pygame**

Durante la simulación, el dron se mueve por la cuadrícula hasta completar todas las entregas o llegar al límite de pasos.

---

## Qué se ve en pantalla

En la ventana gráfica aparecen:

- la cuadrícula del mapa
- los edificios como casillas bloqueadas
- los puntos de entrega marcados con letras
- el dron moviéndose por el entorno
- el número de pasos y la recompensa actual

Esto permite ver de forma clara el recorrido que realiza el dron.

---

## Archivo principal

- `drone_v3_custom.py` → contiene todo el programa:
  - definición del entorno
  - entrenamiento
  - visualización del dron

---

## Librerías usadas

- Python
- Gymnasium
- NumPy
- Pygame

---

## Cómo ejecutarlo

Instalar librerías:

```bash
pip install gymnasium numpy pygame

Ejecutar el programa:

python drone_v3_custom.py
