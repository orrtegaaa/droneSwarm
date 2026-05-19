"""
Drone 3D RL - Version 3.0
==========================
2 drones gestionados por un agente servidor central.
Tablero fijo 8x8, edificios de distintas alturas, vista isometrica.

Arquitectura:
  - Agente Servidor: situado en (0,7), esquina derecha del mapa, no es un dron.
    Aprende la politica con Q-learning independiente para cada dron.
    Monitorea posiciones y emite acciones en cada step.
  - Dron 0 (azul)  : entrega paquete en punto P0
  - Dron 1 (naranja): entrega paquete en punto P1
  - Tras entregar, ambos drones vuelan a la H (helipad en el edificio mas alto)

Mapa fijo 8x8:
  - Tablero color verde-azulado oscuro (se aprecia bien)
  - Edificios de distintas alturas distribuidos por el mapa
  - Edificio central (fila 4, col 4) es el mas alto: tiene la H (helipad)
  - Paquetes P0 y P1 en posiciones fijas accesibles (suelo, alt 0)
  - Dron 0 sale desde (0,0) y Dron 1 desde (0,1); Servidor en (0,7)

Acciones por dron (6):
  0 norte  1 sur  2 oeste  3 este  4 subir  5 bajar
  (no hay accion esperar — deben moverse siempre)

Recompensas (por dron, gestionadas por el Servidor):
  +40   entrega en su paquete
  +15   llega al helipad tras entregar
  +2*d  shaping: bonus por acercarse al objetivo actual
  -1    coste por step
  -18   colision con edificio
  -10   colision entre drones
  -3    volar al nivel del suelo fuera de puntos especiales
  -3*n  anti-bucle (n = visitas extra a la misma celda)

Requisitos:
    pip install gymnasium numpy pygame

Controles:
    R      re-entrenar (mapa identico)
    +/-    velocidad animacion
    ESC    salir
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ─────────────────────────────────────────────

COLS = 8
ROWS = 8
MAX_ALT = 6
N_DRONES = 2
N_ACTIONS = 6  # norte, sur, oeste, este, subir, bajar

TRAIN_EPISODES = 25000
MAX_STEPS = 300
FPS_DEFAULT = 6

R_DELIVERY = +40
R_HELIPAD = +15
R_STEP = -1
R_BUILDING = -18
R_COLLISION = -10
R_FLOOR = -3
R_LOOP_BASE = -3
MAX_VISITS = 2
SHAPING_W = 2.0

ALPHA = 0.12
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.04
EPSILON_DECAY = 0.9998

# ─────────────────────────────────────────────
# MAPA FIJO 8x8
# BUILDING_H[fila][col] = altura edificio (0=suelo libre)
# El edificio mas alto esta en (4,4) con altura 6 -> tiene la H
# ─────────────────────────────────────────────
# col:  0  1  2  3  4  5  6  7
BUILDING_H = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # fila 0  <- inicio drones + servidor
    [0, 2, 0, 3, 0, 3, 2, 0],  # fila 1
    [0, 0, 0, 3, 0, 3, 0, 0],  # fila 2
    [0, 3, 0, 0, 0, 0, 3, 0],  # fila 3
    [0, 0, 0, 0, 6, 0, 0, 0],  # fila 4  <- helipad en (4,4) alt 6
    [0, 3, 0, 0, 0, 0, 3, 0],  # fila 5
    [0, 0, 0, 4, 0, 4, 0, 0],  # fila 6
    [0, 0, 0, 0, 0, 0, 0, 0],  # fila 7
]

SERVER_POS = (0, 7)  # servidor en esquina derecha del mapa (visible)
START_POSITIONS = [(0, 0, 0), (0, 1, 0)]  # posiciones iniciales drones
PACKAGE_POS = [(7, 1, 0), (7, 6, 0)]  # paquetes P0, P1 al nivel suelo
HELIPAD_POS = (4, 4, 6)  # encima del edificio mas alto
PACKAGE_NAMES = ['P0', 'P1']

# ─────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────

# Tablero: verde-azulado oscuro que se aprecia bien con los edificios grises
C_BG = (14, 20, 28)
C_FLOOR_TOP = (28, 55, 65)  # suelo verde-azul
C_FLOOR_LEFT = (18, 38, 46)
C_FLOOR_RIGHT = (12, 28, 35)

C_PANEL_BG = (10, 16, 24)
C_SEP = (35, 55, 70)
C_TEXT = (200, 215, 220)
C_TEXT_DIM = (80, 105, 115)
C_REWARD_POS = (80, 220, 140)
C_REWARD_NEG = (220, 80, 100)
C_GRID_EDGE = (40, 70, 80)

# Edificios: escala de grises azulados, mas claro = mas alto
BLDG_TOP = [(60, 65, 80), (72, 78, 95), (84, 90, 110), (96, 103, 122), (110, 115, 135), (125, 130, 150)]
BLDG_LEFT = [(38, 42, 55), (46, 52, 66), (55, 60, 78), (64, 70, 88), (74, 78, 97), (84, 90, 110)]
BLDG_RIGHT = [(28, 32, 45), (35, 40, 55), (42, 48, 65), (50, 56, 74), (58, 64, 85), (68, 74, 95)]

# Helipad (edificio mas alto, especial)
C_HELIPAD_TOP = (50, 180, 140)
C_HELIPAD_LEFT = (28, 110, 85)
C_HELIPAD_RIGHT = (20, 85, 65)
C_HELIPAD_H = (30, 220, 160)  # color letra H

# Paquetes
PKG_TOP = [(255, 200, 60), (255, 120, 80)]
PKG_LEFT = [(180, 135, 30), (180, 75, 45)]
PKG_RIGHT = [(140, 100, 20), (140, 55, 30)]
PKG_DONE = [(60, 70, 60), (60, 50, 50)]  # entregado (apagado)

# Servidor
C_SERVER = (180, 220, 255)
C_SERVER_D = (100, 150, 190)

# Drones
DRONE_COLORS = [(80, 170, 255), (255, 140, 60)]
DRONE_OUT = [(45, 110, 200), (200, 85, 25)]
TRAIL_COLORS = [(30, 80, 150), (150, 60, 20)]

# ─────────────────────────────────────────────
# PROYECCION ISOMETRICA
# ─────────────────────────────────────────────

TILE_W = 66
TILE_H = 33
TILE_Z = 24
ISO_ORIGIN_X = 370
ISO_ORIGIN_Y = 90


def iso(col, row, alt):
    x = ISO_ORIGIN_X + (col - row) * (TILE_W // 2)
    y = ISO_ORIGIN_Y + (col + row) * (TILE_H // 2) - alt * TILE_Z
    return int(x), int(y)


def dist3(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


# ─────────────────────────────────────────────
# ENTORNO GYMNASIUM
# ─────────────────────────────────────────────

class DroneEnv(gym.Env):
    """
    Entorno con 2 drones gestionados por el Servidor.
    step_all(acciones) -> rewards individuales, done.

    Estado por dron para Q-table del Servidor:
      (fila, col, alt, fase)
      fase: 0=buscando paquete, 1=yendo al helipad, 2=en helipad
    """

    _DELTA = {
        0: (-1, 0, 0), 1: (1, 0, 0),
        2: (0, -1, 0), 3: (0, 1, 0),
        4: (0, 0, 1), 5: (0, 0, -1),
    }

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=max(ROWS, COLS, MAX_ALT, 2),
                                            shape=(8,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = [list(p) for p in START_POSITIONS]
        self.delivered = [False] * N_DRONES
        self.at_helipad = [False] * N_DRONES
        self.steps = 0
        self.trails = [[tuple(p)] for p in START_POSITIONS]
        self.last_rewards = [0.0] * N_DRONES
        self.visit_counts = [defaultdict(int) for _ in range(N_DRONES)]
        for i, p in enumerate(START_POSITIONS):
            self.visit_counts[i][tuple(p)] = 1
        return self._obs(), {}

    def _blocked(self, r, c, alt):
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return True
        bh = BUILDING_H[r][c]
        if bh == 0:
            return False
        # puede volar encima del edificio (alt >= bh) o al suelo (alt==0) si no hay edificio
        return 1 <= alt < bh

    def _phase(self, i):
        if self.at_helipad[i]:   return 2
        if self.delivered[i]:    return 1
        return 0

    def state_key(self, i):
        r, c, a = self.positions[i]
        return (r, c, a, self._phase(i))

    def _obs(self):
        r0, c0, a0 = self.positions[0]
        r1, c1, a1 = self.positions[1]
        return np.array([r0, c0, a0, self._phase(0),
                         r1, c1, a1, self._phase(1)], dtype=np.int32)

    def step_all(self, actions):
        rewards = [0.0] * N_DRONES

        # drones en helipad esperan quietos sin coste
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                actions[i] = -1  # marcador: no mover

        # posiciones propuestas
        proposed = []
        for i in range(N_DRONES):
            if actions[i] == -1:
                proposed.append(list(self.positions[i]))
                continue
            dr, dc, da = self._DELTA[actions[i]]
            nr = max(0, min(ROWS - 1, self.positions[i][0] + dr))
            nc = max(0, min(COLS - 1, self.positions[i][1] + dc))
            na = max(0, min(MAX_ALT, self.positions[i][2] + da))
            proposed.append([nr, nc, na])

        # distancias previas al objetivo
        prev_dist = []
        for i in range(N_DRONES):
            goal = HELIPAD_POS if self.delivered[i] else PACKAGE_POS[i]
            prev_dist.append(dist3(self.positions[i], goal))

        # aplicar movimientos
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                continue

            rewards[i] += R_STEP
            nr, nc, na = proposed[i]

            if self._blocked(nr, nc, na):
                rewards[i] += R_BUILDING
            else:
                # colision entre drones
                other = 1 - i
                if proposed[other] == [nr, nc, na] and not self.at_helipad[other]:
                    rewards[i] += R_COLLISION
                else:
                    self.positions[i] = [nr, nc, na]
                    self.trails[i].append(tuple(self.positions[i]))
                    if len(self.trails[i]) > 35:
                        self.trails[i].pop(0)

                    # anti-bucle
                    cell = tuple(self.positions[i])
                    self.visit_counts[i][cell] += 1
                    extra = self.visit_counts[i][cell] - MAX_VISITS
                    if extra > 0:
                        rewards[i] += R_LOOP_BASE * extra

                    # penalizar suelo fuera de puntos especiales
                    if na == 0:
                        special = {tuple(START_POSITIONS[i]),
                                   tuple(PACKAGE_POS[i]),
                                   tuple(START_POSITIONS[1 - i])}
                        if tuple(self.positions[i]) not in special:
                            rewards[i] += R_FLOOR

        # reward shaping: acercamiento al objetivo
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                continue
            goal = HELIPAD_POS if self.delivered[i] else PACKAGE_POS[i]
            new_dist = dist3(self.positions[i], goal)
            rewards[i] += SHAPING_W * (prev_dist[i] - new_dist)

        # comprobar entregas y helipad
        for i in range(N_DRONES):
            pos_t = tuple(self.positions[i])
            if not self.delivered[i]:
                if pos_t == tuple(PACKAGE_POS[i]):
                    rewards[i] += R_DELIVERY
                    self.delivered[i] = True
                    # reset del contador de visitas al cambiar de fase (buscar paquete -> helipad)
                    self.visit_counts[i].clear()
            elif not self.at_helipad[i]:
                if pos_t == HELIPAD_POS:
                    rewards[i] += R_HELIPAD
                    self.at_helipad[i] = True

        self.last_rewards = rewards
        self.steps += 1
        terminated = all(self.at_helipad)
        truncated = self.steps >= MAX_STEPS
        return rewards, terminated, truncated


# ─────────────────────────────────────────────
# DIBUJO ISOMETRICO
# ─────────────────────────────────────────────

def draw_tile(surface, col, row, alt, ct, cl, cr, levels=1):
    """Dibuja un bloque isometrico solido de 'levels' pisos."""
    top = iso(col, row, alt + levels)
    right = iso(col + 1, row, alt + levels)
    bottom = iso(col + 1, row + 1, alt + levels)
    left = iso(col, row + 1, alt + levels)
    top_b = iso(col, row, alt)
    right_b = iso(col + 1, row, alt)
    bottom_b = iso(col + 1, row + 1, alt)
    left_b = iso(col, row + 1, alt)
    pygame.draw.polygon(surface, cl, [left, bottom, bottom_b, left_b])
    pygame.draw.polygon(surface, cr, [bottom, right, right_b, bottom_b])
    pygame.draw.polygon(surface, ct, [top, right, bottom, left])
    pygame.draw.polygon(surface, C_GRID_EDGE, [top, right, bottom, left], 1)


def _bcolors(h):
    idx = min(h - 1, len(BLDG_TOP) - 1)
    return BLDG_TOP[idx], BLDG_LEFT[idx], BLDG_RIGHT[idx]


def draw_H(surface, col, row, alt, font_h):
    """Dibuja la letra H centrada encima del helipad."""
    x, y = iso(col + 0.5, row + 0.5, alt + 1)
    # circulo de fondo
    pygame.draw.circle(surface, (20, 80, 60), (x, y), 14)
    pygame.draw.circle(surface, C_HELIPAD_H, (x, y), 14, 2)
    txt = font_h.render("H", True, C_HELIPAD_H)
    surface.blit(txt, txt.get_rect(center=(x, y)))


def draw_package(surface, idx, delivered, font_sm):
    """Dibuja un paquete en su posicion."""
    r, c, a = PACKAGE_POS[idx]
    x, y = iso(c + 0.5, r + 0.5, a + 1.0)
    if delivered:
        color = PKG_DONE[idx]
        pygame.draw.rect(surface, color, (x - 7, y - 7, 14, 14), border_radius=3)
        pygame.draw.rect(surface, (50, 60, 50), (x - 7, y - 7, 14, 14), 1, border_radius=3)
    else:
        ct, cl, cr = PKG_TOP[idx], PKG_LEFT[idx], PKG_RIGHT[idx]
        # caja del paquete como bloque pequeno
        pygame.draw.rect(surface, ct, (x - 8, y - 8, 16, 12), border_radius=2)
        pygame.draw.rect(surface, cl, (x - 8, y + 4, 8, 6), border_radius=1)
        pygame.draw.rect(surface, cr, (x, y + 4, 8, 6), border_radius=1)
        pygame.draw.rect(surface, (255, 255, 200), (x - 8, y - 8, 16, 12), 1, border_radius=2)
        lbl = font_sm.render(PACKAGE_NAMES[idx], True, (255, 255, 200))
        surface.blit(lbl, lbl.get_rect(center=(x, y - 2)))


def draw_server(surface, font_sm):
    """Dibuja el servidor en esquina derecha (0,7) — estacion de control."""
    r, c = SERVER_POS
    x, y = iso(c + 0.5, r + 0.5, 0.5)

    # base de la estacion: rectangulo con perspectiva
    base_pts = [
        (x - 14, y + 2), (x, y - 4),
        (x + 14, y + 2), (x, y + 8),
    ]
    pygame.draw.polygon(surface, C_SERVER_D, base_pts)
    pygame.draw.polygon(surface, C_SERVER, base_pts, 1)

    # cuerpo principal: caja
    pygame.draw.rect(surface, C_SERVER_D, (x - 10, y - 14, 20, 16), border_radius=3)
    pygame.draw.rect(surface, C_SERVER, (x - 10, y - 14, 20, 16), 1, border_radius=3)

    # pantalla/ventana en la caja
    pygame.draw.rect(surface, (20, 60, 90), (x - 7, y - 12, 14, 8), border_radius=2)
    pygame.draw.rect(surface, (80, 180, 220), (x - 7, y - 12, 14, 8), 1, border_radius=2)
    # lineas de "datos" en la pantalla
    for dy_off in [3, 6]:
        pygame.draw.line(surface, (60, 160, 200),
                         (x - 5, y - 12 + dy_off), (x + 5, y - 12 + dy_off), 1)

    # mástil
    pygame.draw.line(surface, C_SERVER, (x, y - 14), (x, y - 26), 2)

    # antena parabolica (arco)
    pygame.draw.arc(surface, C_SERVER,
                    (x - 10, y - 34, 20, 16), 0, 3.14, 2)
    pygame.draw.line(surface, C_SERVER, (x, y - 26), (x, y - 28), 2)

    # señal (circulos concentricos)
    for r_sig in [5, 9, 13]:
        pygame.draw.arc(surface, (*C_SERVER[:2], max(0, 120 - r_sig * 8)),
                        (x + 2 - r_sig, y - 32 - r_sig, r_sig * 2, r_sig * 2),
                        0.2, 1.4, 1)

    # etiqueta
    lbl = font_sm.render("CTRL", True, C_SERVER)
    surface.blit(lbl, lbl.get_rect(center=(x, y + 16)))


def draw_scene(surface, env, font_h, font_sm):
    """Renderiza el mapa completo en orden isometrico correcto."""
    order = sorted(
        [(r, c) for r in range(ROWS) for c in range(COLS)],
        key=lambda rc: rc[0] + rc[1]
    )

    for r, c in order:
        bh = BUILDING_H[r][c]
        is_helipad = (r == HELIPAD_POS[0] and c == HELIPAD_POS[1])

        if bh == 0:
            draw_tile(surface, c, r, 0, C_FLOOR_TOP, C_FLOOR_LEFT, C_FLOOR_RIGHT)
        elif is_helipad:
            # edificio mas alto con colores especiales
            draw_tile(surface, c, r, 0,
                      C_HELIPAD_TOP, C_HELIPAD_LEFT, C_HELIPAD_RIGHT, levels=bh)
        else:
            draw_tile(surface, c, r, 0, *_bcolors(bh), levels=bh)

    # H encima del helipad
    draw_H(surface, HELIPAD_POS[1], HELIPAD_POS[0], HELIPAD_POS[2], font_h)

    # paquetes
    for i in range(N_DRONES):
        draw_package(surface, i, env.delivered[i], font_sm)

    # servidor
    draw_server(surface, font_sm)

    # rastros
    for i in range(N_DRONES):
        trail = env.trails[i]
        base = TRAIL_COLORS[i]
        for t in range(1, len(trail)):
            r1, c1, a1 = trail[t - 1];
            r2, c2, a2 = trail[t]
            x1, y1 = iso(c1 + 0.5, r1 + 0.5, a1 + 0.6)
            x2, y2 = iso(c2 + 0.5, r2 + 0.5, a2 + 0.6)
            ratio = t / max(len(trail), 1)
            col_t = (min(255, base[0] + int(70 * ratio)),
                     min(255, base[1] + int(70 * ratio)),
                     min(255, base[2] + int(110 * ratio)))
            pygame.draw.line(surface, col_t, (x1, y1), (x2, y2), 2)

    # drones (orden profundidad)
    order_d = sorted(range(N_DRONES),
                     key=lambda i: env.positions[i][0] + env.positions[i][1])
    for i in order_d:
        _draw_drone(surface, env, i, font_sm)


def _draw_drone(surface, env, i, font_sm):
    r, c, a = env.positions[i]
    x, y = iso(c + 0.5, r + 0.5, a + 1.3)
    dc = DRONE_COLORS[i]
    do = DRONE_OUT[i]

    # anillo helipad
    if env.at_helipad[i]:
        pygame.draw.circle(surface, C_HELIPAD_H, (x, y), 22, 2)

    # sombra eliptica en el suelo
    bh = BUILDING_H[r][c]
    sx, sy = iso(c + 0.5, r + 0.5, bh + 0.05)
    pygame.draw.ellipse(surface, (6, 12, 18), (sx - 13, sy - 5, 26, 10))

    # ── FORMA DE DRON: cruz de brazos + motor en cada punta ──

    ARM = 13  # longitud del brazo desde el centro
    # brazos en perspectiva isometrica: diagonal NE/NO/SE/SO
    arm_dirs = [(-ARM, -ARM // 2), (ARM, -ARM // 2),
                (-ARM, ARM // 2), (ARM, ARM // 2)]
    rotor_r = 6  # radio del rotor

    # sombra de brazos
    for dx, dy in arm_dirs:
        pygame.draw.line(surface, (6, 12, 18), (x + 2, y + 2), (x + dx + 2, y + dy + 2), 4)

    # brazos
    for dx, dy in arm_dirs:
        pygame.draw.line(surface, do, (x, y), (x + dx, y + dy), 3)

    # rotores (disco + aro)
    for dx, dy in arm_dirs:
        rx, ry = x + dx, y + dy
        # disco interior
        pygame.draw.circle(surface, do, (rx, ry), rotor_r)
        pygame.draw.circle(surface, dc, (rx, ry), rotor_r - 2)
        # aro exterior girando
        pygame.draw.circle(surface, (200, 230, 255), (rx, ry), rotor_r, 1)

    # cuerpo central: hexagono aplanado
    body_pts = [
        (x - 8, y - 4), (x, y - 7), (x + 8, y - 4),
        (x + 8, y + 4), (x, y + 7), (x - 8, y + 4),
    ]
    # sombra cuerpo
    shadow_pts = [(px + 2, py + 2) for px, py in body_pts]
    pygame.draw.polygon(surface, (6, 12, 18), shadow_pts)
    # relleno
    pygame.draw.polygon(surface, dc, body_pts)
    # borde
    pygame.draw.polygon(surface, do, body_pts, 2)

    # camara/sensor inferior (punto brillante)
    pygame.draw.circle(surface, (220, 240, 255), (x, y + 2), 2)

    # numero del dron en el cuerpo
    font_lbl = pygame.font.SysFont("monospace", 9, bold=True)
    lbl = font_lbl.render(str(i), True, (255, 255, 255))
    surface.blit(lbl, lbl.get_rect(center=(x, y - 1)))

    # estado flotante encima
    font_tag = pygame.font.SysFont("monospace", 10)
    if env.at_helipad[i]:
        tag = font_tag.render("H ok", True, C_HELIPAD_H)
    elif env.delivered[i]:
        tag = font_tag.render("->H", True, (255, 220, 80))
    else:
        tag = font_tag.render(PACKAGE_NAMES[i], True, dc)
    surface.blit(tag, tag.get_rect(center=(x, y - 28)))


def draw_alt_bars(surface, env, font_sm, x0, y0):
    """Barras de altitud para los 2 drones."""
    spacing = 44
    bar_h = MAX_ALT * 20
    bar_w = 12
    for i in range(N_DRONES):
        bx = x0 + i * spacing
        cur = env.positions[i][2]
        for a in range(MAX_ALT + 1):
            ry = y0 + bar_h - a * 20
            color = DRONE_COLORS[i] if a == cur else (25, 40, 50)
            pygame.draw.rect(surface, color, (bx, ry - 8, bar_w, 16), border_radius=3)
            if a == cur:
                lv = font_sm.render(str(a), True, (255, 255, 255))
                surface.blit(lv, (bx + bar_w + 4, ry - 7))
        s = font_sm.render(f"D{i}", True, DRONE_COLORS[i])
        surface.blit(s, (bx, y0 + bar_h + 8))
    s = font_sm.render("ALT", True, C_TEXT_DIM)
    surface.blit(s, (x0, y0 - 18))


# ─────────────────────────────────────────────
# PANEL DERECHO
# ─────────────────────────────────────────────

def draw_panel(surface, server, hist, episode,
               total_reward, env, font_md, font_sm, win_w, win_h, fps):
    px = win_w - 225
    pw = 215
    pygame.draw.rect(surface, C_PANEL_BG, (px - 8, 0, pw + 16, win_h))
    pygame.draw.line(surface, C_SEP, (px - 8, 0), (px - 8, win_h), 1)

    def lbl(text, y, color=C_TEXT_DIM, f=font_sm):
        surface.blit(f.render(text, True, color), (px, y))

    def val(text, y, color=C_TEXT, f=font_md):
        surface.blit(f.render(text, True, color), (px, y))

    def sep(y):
        pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1)
        return y + 8

    y = 25
    lbl("DRONE 3D RL  v3", y, C_TEXT, font_md);
    y += 20
    lbl(f"Tablero {COLS}x{ROWS}x{MAX_ALT}  |  2 drones", y);
    y += 18
    lbl(f"Servidor en {SERVER_POS} (esq. derecha)  |  Helipad {HELIPAD_POS}", y);
    y += 20
    y = sep(y)

    lbl("EPISODIO", y);
    y += 13;
    val(str(episode), y);
    y += 21
    lbl("REWARD EP", y);
    y += 13
    rc = C_REWARD_POS if total_reward >= 0 else C_REWARD_NEG
    val(f"{total_reward:.0f}", y, rc);
    y += 21
    lbl("EPSILON", y);
    y += 13;
    val(f"{server.epsilon:.3f}", y);
    y += 21
    lbl("PASOS", y);
    y += 13;
    val(f"{env.steps}/{MAX_STEPS}", y);
    y += 21
    qsz = " / ".join(str(s) for s in server.q_sizes())
    lbl("Q-STATES", y);
    y += 13;
    val(qsz, y);
    y += 21
    y = sep(y)

    # estado drones
    lbl("ESTADO DRONES", y, C_TEXT);
    y += 16
    for i in range(N_DRONES):
        r, c, a = env.positions[i]
        rw = env.last_rewards[i]
        if env.at_helipad[i]:
            estado, ec = "helipad ok", C_HELIPAD_H
        elif env.delivered[i]:
            estado, ec = "-> helipad", (255, 220, 80)
        else:
            estado, ec = f"-> {PACKAGE_NAMES[i]}", DRONE_COLORS[i]
        rw_c = C_REWARD_POS if rw >= 0 else C_REWARD_NEG
        s = font_sm.render(f"D{i} ({r},{c},{a})  {estado}", True, ec)
        surface.blit(s, (px, y));
        y += 14
        s2 = font_sm.render(f"   rew step: {rw:.1f}", True, rw_c)
        surface.blit(s2, (px, y));
        y += 16

    # info servidor
    pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1);
    y += 8
    lbl("SERVIDOR", y, C_TEXT);
    y += 14
    s = font_sm.render(f"pos {SERVER_POS}  esq. derecha del mapa", True, C_SERVER)
    surface.blit(s, (px, y));
    y += 14
    s2 = font_sm.render(f"emite acciones para D0 y D1", True, C_TEXT_DIM)
    surface.blit(s2, (px, y));
    y += 16
    y = sep(y)

    # paquetes
    lbl("PAQUETES", y, C_TEXT);
    y += 15
    for i in range(N_DRONES):
        done = env.delivered[i]
        color = PKG_TOP[i] if not done else (60, 75, 65)
        mark = "OK entregado" if done else f"en {PACKAGE_POS[i]}"
        s = font_sm.render(f"{PACKAGE_NAMES[i]}: {mark}", True, color)
        surface.blit(s, (px, y));
        y += 14
    hp_n = sum(env.at_helipad)
    hp_c = C_HELIPAD_H if hp_n == N_DRONES else C_TEXT_DIM
    s = font_sm.render(f"Helipad: {hp_n}/{N_DRONES} drones", True, hp_c)
    surface.blit(s, (px, y));
    y += 14
    y = sep(y)

    # grafica reward
    if len(hist) > 5:
        lbl("REWARD (200 ep)", y);
        y += 13
        recent = hist[-200:]
        mn, mx = min(recent), max(recent)
        rng = max(mx - mn, 1)
        gh, gw = 50, pw - 14
        for t in range(1, len(recent)):
            x1 = px + int((t - 1) / len(recent) * gw)
            x2 = px + int(t / len(recent) * gw)
            y1 = y + gh - int((recent[t - 1] - mn) / rng * gh)
            y2 = y + gh - int((recent[t] - mn) / rng * gh)
            pygame.draw.line(surface, DRONE_COLORS[0], (x1, y1), (x2, y2), 1)
        pygame.draw.rect(surface, C_SEP, (px, y, gw, gh), 1)
        y += gh + 4
        lbl(f"media {np.mean(recent):.1f}  mejor {mx:.0f}", y);
        y += 15
    y = sep(y)

    # leyenda
    lbl("LEYENDA", y, C_TEXT);
    y += 14
    for i in range(N_DRONES):
        pygame.draw.circle(surface, DRONE_COLORS[i], (px + 6, y + 6), 6)
        lbl(f"   Dron {i} -> {PACKAGE_NAMES[i]}", y, DRONE_COLORS[i]);
        y += 14
    pygame.draw.rect(surface, C_SERVER_D, (px + 2, y + 2, 12, 12), border_radius=2)
    lbl(f"   Servidor (gestor)", y, C_SERVER);
    y += 14
    pygame.draw.circle(surface, C_HELIPAD_H, (px + 6, y + 6), 6)
    lbl(f"   Helipad H  {HELIPAD_POS}", y, C_HELIPAD_H);
    y += 14

    y = win_h - 58
    pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1);
    y += 8
    lbl(f"FPS: {fps}   +/- ajustar", y);
    y += 15
    lbl("R reentrenar  |  ESC salir", y)


# ─────────────────────────────────────────────
# BUCLE VISUAL PRINCIPAL
# ─────────────────────────────────────────────

def run_visual(server, hist, train_fn=None):
    """
    Bucle visual principal.
    train_fn: funcion de entrenamiento a llamar si el usuario pulsa R.
              Sigue el mismo patron que demo_visual(get_action_fn) de la v2.
    """
    WIN_W, WIN_H = 980, 730

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Drone 3D RL v3  |  Servidor + 2 drones  |  Helipad H")
    clock = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("monospace", 16, bold=True)
        font_md = pygame.font.SysFont("monospace", 13, bold=True)
        font_sm = pygame.font.SysFont("monospace", 11)
        font_h = pygame.font.SysFont("monospace", 14, bold=True)
    except Exception:
        font_lg = pygame.font.Font(None, 22)
        font_md = pygame.font.Font(None, 18)
        font_sm = pygame.font.Font(None, 15)
        font_h = pygame.font.Font(None, 20)

    env = DroneEnv()
    episode = 0
    fps = FPS_DEFAULT
    retrain = False

    while True:
        env.reset()
        sks = [env.state_key(i) for i in range(N_DRONES)]
        total_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit();
                        sys.exit()
                    if event.key == pygame.K_r:
                        retrain = True;
                        done = True
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        fps = min(30, fps + 1)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        fps = max(1, fps - 1)

            if retrain:
                break

            acts = [server.act(i, sks[i], greedy=True) for i in range(N_DRONES)]
            rwds, term, trunc = env.step_all(acts)
            sks = [env.state_key(i) for i in range(N_DRONES)]
            total_reward += sum(rwds)
            done = term or trunc

            screen.fill(C_BG)
            draw_scene(screen, env, font_h, font_sm)
            draw_alt_bars(screen, env, font_sm, 22, WIN_H - 210)

            title = font_lg.render(
                "Drone 3D RL v3  |  Servidor + Q indep.  |  shaping + anti-bucle",
                True, C_TEXT)
            screen.blit(title, (18, 12))

            draw_panel(screen, server, hist, episode,
                       total_reward, env, font_md, font_sm, WIN_W, WIN_H, fps)

            pygame.display.flip()
            clock.tick(fps)

        if retrain:
            retrain = False
            pygame.quit()
            print("\nRe-entrenando (mapa identico)...")
            server, hist = train_fn()
            run_visual(server, hist, train_fn)
            return

        episode += 1
        time.sleep(0.4)
