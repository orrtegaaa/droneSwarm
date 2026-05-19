## DroneSwarm 3D

Simulation of two delivery drones managed by a central server agent in a 3D isometric grid

---

## Description

This project is a Python program that simulates two drones operating inside a 3D grid map. A central server agent learns a separate Q-learning policy for each drone and coordinates their movements at every step.

Each drone must pick up its assigned package and then fly to a helipad located on top of the tallest building. Buildings of different heights are distributed across the map, so drones must navigate around them in three dimensions.

When the program runs, the server agent is trained first using Q-learning. Once training is complete, a visual window opens where you can watch both drones move across the isometric map in real time.

---

## What the program does

The program includes:

- an **8x8x6** map (rows × columns × altitude)
- **2 drones** starting from adjacent positions at `(0,0)` and `(0,1)`
- a **server agent** at `(0,7)` that manages both drones independently
- **2 delivery packages** (P0 and P1) placed at ground level
- a **helipad** on the tallest building at `(4,4,6)` — the final goal for both drones
- buildings of different heights that drones must fly over or around
- a reward system with delivery bonuses, shaping, anti-loop penalties and collision detection
- a training system based on independent Q-tables per drone
- a final isometric 3D visualization with **Pygame**

During the simulation, drones move through the grid collecting packages and returning to the helipad, or the episode ends when the step limit is reached.

---

## What you see on the screen

In the visual window, you can see:

- the isometric 3D map with buildings of different heights
- the helipad marked with an **H** on the tallest building
- the two packages (P0 and P1) and their delivery status
- the server station in the top-right corner of the map
- both drones moving with trails showing their recent path
- altitude bars for each drone on the left side
- a right panel with episode number, reward, epsilon, Q-table sizes and drone status
- a live reward graph updated every episode

---

## Project structure

The project is split across three files following the same architecture as the previous version:

- `drone_env_3d.py` → environment definition, map configuration, all drawing functions and the visual loop
- `entrenamiento_3d.py` → server agent (`ServerAgent`) and training logic (`train`)
- `main_3d.py` → entry point, calls training and launches the visual demo

---

## Libraries used

- Python 3.10+
- Gymnasium
- NumPy
- Pygame

---

## How to run it

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main_3d.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Re-train the agent (same map) |
| `+` / `-` | Increase / decrease animation speed |
| `ESC` | Exit |
