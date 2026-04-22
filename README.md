## DroneSwarm v3_Custom

Basic simulation of a delivery drone in a grid with obstacles

---

## Description

This project is a Python program that simulates the movement of a drone inside a grid map.

The drone starts from an initial position and has to move through the environment to complete several deliveries. There are also obstacles on the map that represent buildings, so not all paths are valid.

When the program runs, the agent is trained first, and then a visual window is shown where you can see how the drone moves on the map.

---

## What the program does

The program includes:

- an **8x8** map
- a starting position for the drone
- **3 delivery points**
- several buildings blocking some cells
- a training system
- a final visualization with **Pygame**

During the simulation, the drone moves through the grid until it completes all deliveries or reaches the step limit.

---

## What you see on the screen

In the visual window, you can see:

- the map grid
- the buildings as blocked cells
- the delivery points marked with letters
- the drone moving through the environment
- the number of steps and the current reward

This makes it easy to clearly see the path followed by the drone.

---

## Main file

- `drone_v3_custom.py` → contains the whole program:
  - environment definition
  - training
  - drone visualization

---

## Libraries used

- Python
- Gymnasium
- NumPy
- Pygame

---

## How to run it

Install the libraries:

```bash
pip install gymnasium numpy pygame

Run the program:

python drone_v3_custom.py
