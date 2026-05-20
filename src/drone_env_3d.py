"""
Drone 3D RL - Version 3.0
==========================
2 drones managed by a central server agent.
Fixed 8x8 board, buildings with different heights, and isometric view.

Architecture:
  - Server Agent: located at (0,7), right corner of the map, it is not a drone.
    It learns the policy using independent Q-learning for each drone.
    It monitors positions and sends actions at each step.
  - Drone 0 (blue): delivers package P0
  - Drone 1 (orange): delivers package P1
  - After delivery, both drones fly to H, the helipad on the tallest building.

Fixed 8x8 map:
  - Dark green-blue board
  - Buildings with different heights
  - Central building at (4,4) is the tallest one and has the helipad
  - Packages P0 and P1 are placed in fixed accessible positions
  - Drone 0 starts at (0,0), Drone 1 starts at (0,1), and the server is at (0,7)

Actions per drone:
  0 north  1 south  2 west  3 east  4 up  5 down
  There is no wait action, so drones must always move.

Rewards per drone:
  +40   package delivered
  +15   reaches helipad after delivery
  +2*d  shaping bonus for moving closer to the current goal
  -1    step cost
  -18   collision with building
  -10   collision between drones
  -3    flying at ground level outside special points
  -3*n  anti-loop penalty, where n is extra visits to the same cell

Requirements:
    pip install gymnasium numpy pygame

Controls:
    R      retrain using the same map
    +/-    change animation speed
    ESC    exit
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
from collections import defaultdict


# GLOBAL CONFIGURATION
# Here we define the general size of the map, the number of drones,
# the actions, rewards and Q-learning parameters.

COLS = 8
ROWS = 8
MAX_ALT = 6
N_DRONES = 2

# Each drone has 6 possible actions:
# north, south, west, east, up and down.
N_ACTIONS = 6

# Training and visual simulation parameters.
TRAIN_EPISODES = 25000
MAX_STEPS = 300
FPS_DEFAULT = 6

# Reward values used in the environment.
# Positive rewards are used for completing goals.
# Negative rewards are used for bad movements or inefficient behaviour.
R_DELIVERY = +40
R_HELIPAD = +15
R_STEP = -1
R_BUILDING = -18
R_COLLISION = -10
R_FLOOR = -3
R_LOOP_BASE = -3
MAX_VISITS = 2
SHAPING_W = 2.0

# Q-learning parameters.
ALPHA = 0.12
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.04
EPSILON_DECAY = 0.9998


# FIXED 8x8 MAP
# BUILDING_H[row][col] = building height.
# 0 means free floor.
# The highest building is at (4,4), with height 6, and has the helipad.

# col:  0  1  2  3  4  5  6  7
BUILDING_H = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # row 0  <- drones and server start here
    [0, 2, 0, 3, 0, 3, 2, 0],  # row 1
    [0, 0, 0, 3, 0, 3, 0, 0],  # row 2
    [0, 3, 0, 0, 0, 0, 3, 0],  # row 3
    [0, 0, 0, 0, 6, 0, 0, 0],  # row 4  <- helipad at (4,4), altitude 6
    [0, 3, 0, 0, 0, 0, 3, 0],  # row 5
    [0, 0, 0, 4, 0, 4, 0, 0],  # row 6
    [0, 0, 0, 0, 0, 0, 0, 0],  # row 7
]

# Main fixed positions in the map.
SERVER_POS = (0, 7)                         # server in the right corner of the map
START_POSITIONS = [(0, 0, 0), (0, 1, 0)]    # initial drone positions
PACKAGE_POS = [(7, 1, 0), (7, 6, 0)]        # package positions at ground level
HELIPAD_POS = (4, 4, 6)                     # helipad on top of the highest building
PACKAGE_NAMES = ['P0', 'P1']


# COLOR 
# These colors are used only for the pygame visual part.
# They do not affect the learning process.

# General board colors.
C_BG = (14, 20, 28)
C_FLOOR_TOP = (28, 55, 65)
C_FLOOR_LEFT = (18, 38, 46)
C_FLOOR_RIGHT = (12, 28, 35)

# Panel and text colors.
C_PANEL_BG = (10, 16, 24)
C_SEP = (35, 55, 70)
C_TEXT = (200, 215, 220)
C_TEXT_DIM = (80, 105, 115)
C_REWARD_POS = (80, 220, 140)
C_REWARD_NEG = (220, 80, 100)
C_GRID_EDGE = (40, 70, 80)

# Building colors.
# Lighter colors are used for taller buildings.
BLDG_TOP = [(60, 65, 80), (72, 78, 95), (84, 90, 110), (96, 103, 122), (110, 115, 135), (125, 130, 150)]
BLDG_LEFT = [(38, 42, 55), (46, 52, 66), (55, 60, 78), (64, 70, 88), (74, 78, 97), (84, 90, 110)]
BLDG_RIGHT = [(28, 32, 45), (35, 40, 55), (42, 48, 65), (50, 56, 74), (58, 64, 85), (68, 74, 95)]

# Special helipad colors.
C_HELIPAD_TOP = (50, 180, 140)
C_HELIPAD_LEFT = (28, 110, 85)
C_HELIPAD_RIGHT = (20, 85, 65)
C_HELIPAD_H = (30, 220, 160)

# Package colors.
PKG_TOP = [(255, 200, 60), (255, 120, 80)]
PKG_LEFT = [(180, 135, 30), (180, 75, 45)]
PKG_RIGHT = [(140, 100, 20), (140, 55, 30)]
PKG_DONE = [(60, 70, 60), (60, 50, 50)]

# Server colors.
C_SERVER = (180, 220, 255)
C_SERVER_D = (100, 150, 190)

# Drone colors.
DRONE_COLORS = [(80, 170, 255), (255, 140, 60)]
DRONE_OUT = [(45, 110, 200), (200, 85, 25)]
TRAIL_COLORS = [(30, 80, 150), (150, 60, 20)]


# ISOMETRIC PROJECTION
# These values are used to convert map coordinates into screen coordinates.

TILE_W = 66
TILE_H = 33
TILE_Z = 24
ISO_ORIGIN_X = 370
ISO_ORIGIN_Y = 90


def iso(col, row, alt):
    """
    Converts a 3D grid position into a 2D isometric screen position.
    """
    x = ISO_ORIGIN_X + (col - row) * (TILE_W // 2)
    y = ISO_ORIGIN_Y + (col + row) * (TILE_H // 2) - alt * TILE_Z
    return int(x), int(y)


def dist3(a, b):
    """
    Calculates Manhattan distance in 3D.
    It is used to know if a drone is getting closer to its objective.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


# GYMNASIUM ENVIRONMENT
# This is the custom environment created by us.
# It manages the map, drones, collisions, rewards and final state.

class DroneEnv(gym.Env):
    """
    Environment with 2 drones managed by the server.

    step_all(actions) returns:
      - individual rewards
      - terminated
      - truncated

    State used for each drone in the Q-table:
      (row, column, altitude, phase)

    Phases:
      0 = searching package
      1 = going to helipad
      2 = already in helipad
    """

    # Movement for each action.
    _DELTA = {
        0: (-1, 0, 0), 1: (1, 0, 0),
        2: (0, -1, 0), 3: (0, 1, 0),
        4: (0, 0, 1), 5: (0, 0, -1),
    }

    def __init__(self):
        super().__init__()

        # The action space is discrete because each drone chooses one action
        # from a fixed list.
        self.action_space = spaces.Discrete(N_ACTIONS)

        # The observation contains the position and phase of both drones.
        self.observation_space = spaces.Box(low=0, high=max(ROWS, COLS, MAX_ALT, 2),
                                            shape=(8,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.
        """
        super().reset(seed=seed)

        # Reset positions and status of both drones.
        self.positions = [list(p) for p in START_POSITIONS]
        self.delivered = [False] * N_DRONES
        self.at_helipad = [False] * N_DRONES

        # Reset step counter.
        self.steps = 0

        # Trails are used only to draw the path in the visual simulation.
        self.trails = [[tuple(p)] for p in START_POSITIONS]

        # Last rewards are shown in the visual panel.
        self.last_rewards = [0.0] * N_DRONES

        # Visit counts are used to penalize loops.
        self.visit_counts = [defaultdict(int) for _ in range(N_DRONES)]
        for i, p in enumerate(START_POSITIONS):
            self.visit_counts[i][tuple(p)] = 1

        return self._obs(), {}

    def _blocked(self, r, c, alt):
        """
        Checks if a drone cannot be in a specific position.
        A position is blocked if it is outside the map or inside a building.
        """
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return True

        bh = BUILDING_H[r][c]

        if bh == 0:
            return False

        # If there is a building, the drone can only pass if it flies above it.
        return 1 <= alt < bh

    def _phase(self, i):
        """
        Returns the current phase of drone i.
        """
        if self.at_helipad[i]:
            return 2
        if self.delivered[i]:
            return 1
        return 0

    def state_key(self, i):
        """
        Returns the state key used in the Q-table for drone i.
        """
        r, c, a = self.positions[i]
        return (r, c, a, self._phase(i))

    def _obs(self):
        """
        Builds the full observation of the environment.
        It contains the information of both drones.
        """
        r0, c0, a0 = self.positions[0]
        r1, c1, a1 = self.positions[1]
        return np.array([r0, c0, a0, self._phase(0),
                         r1, c1, a1, self._phase(1)], dtype=np.int32)

    def step_all(self, actions):
        """
        Executes one action for each drone in the same step.
        """
        rewards = [0.0] * N_DRONES

        # If a drone is already at the helipad, it does not move anymore.
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                actions[i] = -1

        # First we calculate the proposed positions.
        # We do this before applying movements to detect collisions.
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

        # We save the previous distance to the current goal.
        # Later we compare it with the new distance for reward shaping.
        prev_dist = []
        for i in range(N_DRONES):
            goal = HELIPAD_POS if self.delivered[i] else PACKAGE_POS[i]
            prev_dist.append(dist3(self.positions[i], goal))

        # Now we apply the movements and give rewards or penalties.
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                continue

            # Small cost for each step.
            rewards[i] += R_STEP
            nr, nc, na = proposed[i]

            # Penalty if the drone hits a building.
            if self._blocked(nr, nc, na):
                rewards[i] += R_BUILDING
            else:
                other = 1 - i

                # Penalty if both drones want to go to the same position.
                if proposed[other] == [nr, nc, na] and not self.at_helipad[other]:
                    rewards[i] += R_COLLISION
                else:
                    # If there is no collision, the movement is accepted.
                    self.positions[i] = [nr, nc, na]

                    # Save the position for drawing the trail.
                    self.trails[i].append(tuple(self.positions[i]))
                    if len(self.trails[i]) > 35:
                        self.trails[i].pop(0)

                    # Anti-loop penalty.
                    # If a drone visits the same cell too many times,
                    # it receives an extra penalty.
                    cell = tuple(self.positions[i])
                    self.visit_counts[i][cell] += 1
                    extra = self.visit_counts[i][cell] - MAX_VISITS
                    if extra > 0:
                        rewards[i] += R_LOOP_BASE * extra

                    # Penalty for staying at ground level outside special points.
                    if na == 0:
                        special = {tuple(START_POSITIONS[i]),
                                   tuple(PACKAGE_POS[i]),
                                   tuple(START_POSITIONS[1 - i])}
                        if tuple(self.positions[i]) not in special:
                            rewards[i] += R_FLOOR

        # Reward shaping.
        # If the drone gets closer to its current goal, it receives a bonus.
        # If it moves away, the value becomes negative.
        for i in range(N_DRONES):
            if self.at_helipad[i]:
                continue

            goal = HELIPAD_POS if self.delivered[i] else PACKAGE_POS[i]
            new_dist = dist3(self.positions[i], goal)

            rewards[i] += SHAPING_W * (prev_dist[i] - new_dist)

        # Check if packages have been delivered or if drones reached the helipad.
        for i in range(N_DRONES):
            pos_t = tuple(self.positions[i])

            if not self.delivered[i]:
                if pos_t == tuple(PACKAGE_POS[i]):
                    rewards[i] += R_DELIVERY
                    self.delivered[i] = True

                    # Reset visits because the drone changes its objective.
                    self.visit_counts[i].clear()

            elif not self.at_helipad[i]:
                if pos_t == HELIPAD_POS:
                    rewards[i] += R_HELIPAD
                    self.at_helipad[i] = True

        # Save rewards for the visual panel.
        self.last_rewards = rewards

        # Increase episode step counter.
        self.steps += 1

        # The episode finishes when all drones are at the helipad.
        terminated = all(self.at_helipad)

        # If the maximum number of steps is reached, the episode is truncated.
        truncated = self.steps >= MAX_STEPS

        return rewards, terminated, truncated

# ISOMETRIC DRAWING
# This part contains the functions used to draw the map,
# buildings, packages, server and drones with pygame.

def draw_tile(surface, col, row, alt, ct, cl, cr, levels=1):
    """
    Draws a solid isometric block.

    surface: pygame screen
    col, row, alt: position of the block
    ct, cl, cr: top, left and right colors
    levels: height of the block
    """

    # Top points of the block
    top = iso(col, row, alt + levels)
    right = iso(col + 1, row, alt + levels)
    bottom = iso(col + 1, row + 1, alt + levels)
    left = iso(col, row + 1, alt + levels)

    # Bottom points of the block
    top_b = iso(col, row, alt)
    right_b = iso(col + 1, row, alt)
    bottom_b = iso(col + 1, row + 1, alt)
    left_b = iso(col, row + 1, alt)

    # Draw left side
    pygame.draw.polygon(surface, cl, [left, bottom, bottom_b, left_b])

    # Draw right side
    pygame.draw.polygon(surface, cr, [bottom, right, right_b, bottom_b])

    # Draw top side
    pygame.draw.polygon(surface, ct, [top, right, bottom, left])

    # Draw edge of the top face
    pygame.draw.polygon(surface, C_GRID_EDGE, [top, right, bottom, left], 1)


def _bcolors(h):
    """
    Returns the building colors depending on its height.
    Taller buildings use lighter colors.
    """
    idx = min(h - 1, len(BLDG_TOP) - 1)

    return BLDG_TOP[idx], BLDG_LEFT[idx], BLDG_RIGHT[idx]


def draw_H(surface, col, row, alt, font_h):
    """
    Draws the H letter on top of the helipad.
    """

    x, y = iso(col + 0.5, row + 0.5, alt + 1)

    # Background circle
    pygame.draw.circle(surface, (20, 80, 60), (x, y), 14)

    # Circle border
    pygame.draw.circle(surface, C_HELIPAD_H, (x, y), 14, 2)

    # H text
    txt = font_h.render("H", True, C_HELIPAD_H)
    surface.blit(txt, txt.get_rect(center=(x, y)))


def draw_package(surface, idx, delivered, font_sm):
    """
    Draws one package in its position.
    If it was already delivered, it appears darker.
    """

    r, c, a = PACKAGE_POS[idx]

    # Convert package position to screen position
    x, y = iso(c + 0.5, r + 0.5, a + 1.0)

    if delivered:
        # Delivered package is shown with a darker color
        color = PKG_DONE[idx]
        pygame.draw.rect(surface, color, (x - 7, y - 7, 14, 14), border_radius=3)
        pygame.draw.rect(surface, (50, 60, 50), (x - 7, y - 7, 14, 14), 1, border_radius=3)

    else:
        # Normal package colors
        ct, cl, cr = PKG_TOP[idx], PKG_LEFT[idx], PKG_RIGHT[idx]

        # Package box
        pygame.draw.rect(surface, ct, (x - 8, y - 8, 16, 12), border_radius=2)
        pygame.draw.rect(surface, cl, (x - 8, y + 4, 8, 6), border_radius=1)
        pygame.draw.rect(surface, cr, (x, y + 4, 8, 6), border_radius=1)
        pygame.draw.rect(surface, (255, 255, 200), (x - 8, y - 8, 16, 12), 1, border_radius=2)

        # Package label
        lbl = font_sm.render(PACKAGE_NAMES[idx], True, (255, 255, 200))
        surface.blit(lbl, lbl.get_rect(center=(x, y - 2)))


def draw_server(surface, font_sm):
    """
    Draws the server in the right corner of the map.
    It represents the control station.
    """

    r, c = SERVER_POS
    x, y = iso(c + 0.5, r + 0.5, 0.5)

    # Base of the station
    base_pts = [
        (x - 14, y + 2), (x, y - 4),
        (x + 14, y + 2), (x, y + 8),
    ]
    pygame.draw.polygon(surface, C_SERVER_D, base_pts)
    pygame.draw.polygon(surface, C_SERVER, base_pts, 1)

    # Main body of the station
    pygame.draw.rect(surface, C_SERVER_D, (x - 10, y - 14, 20, 16), border_radius=3)
    pygame.draw.rect(surface, C_SERVER, (x - 10, y - 14, 20, 16), 1, border_radius=3)

    # Screen of the station
    pygame.draw.rect(surface, (20, 60, 90), (x - 7, y - 12, 14, 8), border_radius=2)
    pygame.draw.rect(surface, (80, 180, 220), (x - 7, y - 12, 14, 8), 1, border_radius=2)

    # Small data lines on the screen
    for dy_off in [3, 6]:
        pygame.draw.line(surface, (60, 160, 200),
                         (x - 5, y - 12 + dy_off), (x + 5, y - 12 + dy_off), 1)

    # Mast
    pygame.draw.line(surface, C_SERVER, (x, y - 14), (x, y - 26), 2)

    # Antenna
    pygame.draw.arc(surface, C_SERVER,
                    (x - 10, y - 34, 20, 16), 0, 3.14, 2)
    pygame.draw.line(surface, C_SERVER, (x, y - 26), (x, y - 28), 2)

    # Signal waves
    for r_sig in [5, 9, 13]:
        pygame.draw.arc(surface, (*C_SERVER[:2], max(0, 120 - r_sig * 8)),
                        (x + 2 - r_sig, y - 32 - r_sig, r_sig * 2, r_sig * 2),
                        0.2, 1.4, 1)

    # Server label
    lbl = font_sm.render("CTRL", True, C_SERVER)
    surface.blit(lbl, lbl.get_rect(center=(x, y + 16)))


def draw_scene(surface, env, font_h, font_sm):
    """
    Draws the complete map in the correct isometric order.
    """

    # We sort the cells so the map is drawn from back to front.
    order = sorted(
        [(r, c) for r in range(ROWS) for c in range(COLS)],
        key=lambda rc: rc[0] + rc[1]
    )

    # Draw floor and buildings
    for r, c in order:
        bh = BUILDING_H[r][c]
        is_helipad = (r == HELIPAD_POS[0] and c == HELIPAD_POS[1])

        if bh == 0:
            # Free floor cell
            draw_tile(surface, c, r, 0, C_FLOOR_TOP, C_FLOOR_LEFT, C_FLOOR_RIGHT)

        elif is_helipad:
            # Highest building with special helipad colors
            draw_tile(surface, c, r, 0,
                      C_HELIPAD_TOP, C_HELIPAD_LEFT, C_HELIPAD_RIGHT, levels=bh)

        else:
            # Normal building
            draw_tile(surface, c, r, 0, *_bcolors(bh), levels=bh)

    # Draw H on the helipad
    draw_H(surface, HELIPAD_POS[1], HELIPAD_POS[0], HELIPAD_POS[2], font_h)

    # Draw packages
    for i in range(N_DRONES):
        draw_package(surface, i, env.delivered[i], font_sm)

    # Draw server
    draw_server(surface, font_sm)

    # Draw drone trails
    for i in range(N_DRONES):
        trail = env.trails[i]
        base = TRAIL_COLORS[i]

        for t in range(1, len(trail)):
            r1, c1, a1 = trail[t - 1]
            r2, c2, a2 = trail[t]

            x1, y1 = iso(c1 + 0.5, r1 + 0.5, a1 + 0.6)
            x2, y2 = iso(c2 + 0.5, r2 + 0.5, a2 + 0.6)

            ratio = t / max(len(trail), 1)

            # The trail becomes a little brighter near the current position.
            col_t = (min(255, base[0] + int(70 * ratio)),
                     min(255, base[1] + int(70 * ratio)),
                     min(255, base[2] + int(110 * ratio)))

            pygame.draw.line(surface, col_t, (x1, y1), (x2, y2), 2)

    # Draw drones with depth order
    order_d = sorted(range(N_DRONES),
                     key=lambda i: env.positions[i][0] + env.positions[i][1])

    for i in order_d:
        _draw_drone(surface, env, i, font_sm)


def _draw_drone(surface, env, i, font_sm):
    """
    Draws one drone with its body, arms, rotors, shadow and label.
    """

    r, c, a = env.positions[i]
    x, y = iso(c + 0.5, r + 0.5, a + 1.3)

    dc = DRONE_COLORS[i]
    do = DRONE_OUT[i]

    # If the drone is already at the helipad, draw a green circle around it.
    if env.at_helipad[i]:
        pygame.draw.circle(surface, C_HELIPAD_H, (x, y), 22, 2)

    # Ground shadow
    bh = BUILDING_H[r][c]
    sx, sy = iso(c + 0.5, r + 0.5, bh + 0.05)
    pygame.draw.ellipse(surface, (6, 12, 18), (sx - 13, sy - 5, 26, 10))

    # Drone shape: arms and rotors
    ARM = 13

    # Arms in isometric perspective
    arm_dirs = [(-ARM, -ARM // 2), (ARM, -ARM // 2),
                (-ARM, ARM // 2), (ARM, ARM // 2)]

    rotor_r = 6

    # Arm shadows
    for dx, dy in arm_dirs:
        pygame.draw.line(surface, (6, 12, 18), (x + 2, y + 2), (x + dx + 2, y + dy + 2), 4)

    # Arms
    for dx, dy in arm_dirs:
        pygame.draw.line(surface, do, (x, y), (x + dx, y + dy), 3)

    # Rotors
    for dx, dy in arm_dirs:
        rx, ry = x + dx, y + dy

        # Rotor inside
        pygame.draw.circle(surface, do, (rx, ry), rotor_r)
        pygame.draw.circle(surface, dc, (rx, ry), rotor_r - 2)

        # Outer ring
        pygame.draw.circle(surface, (200, 230, 255), (rx, ry), rotor_r, 1)

    # Central body of the drone
    body_pts = [
        (x - 8, y - 4), (x, y - 7), (x + 8, y - 4),
        (x + 8, y + 4), (x, y + 7), (x - 8, y + 4),
    ]

    # Body shadow
    shadow_pts = [(px + 2, py + 2) for px, py in body_pts]
    pygame.draw.polygon(surface, (6, 12, 18), shadow_pts)

    # Body fill
    pygame.draw.polygon(surface, dc, body_pts)

    # Body border
    pygame.draw.polygon(surface, do, body_pts, 2)

    # Small camera or sensor
    pygame.draw.circle(surface, (220, 240, 255), (x, y + 2), 2)

    # Drone number
    font_lbl = pygame.font.SysFont("monospace", 9, bold=True)
    lbl = font_lbl.render(str(i), True, (255, 255, 255))
    surface.blit(lbl, lbl.get_rect(center=(x, y - 1)))

    # Floating status text above the drone
    font_tag = pygame.font.SysFont("monospace", 10)

    if env.at_helipad[i]:
        tag = font_tag.render("H ok", True, C_HELIPAD_H)

    elif env.delivered[i]:
        tag = font_tag.render("->H", True, (255, 220, 80))

    else:
        tag = font_tag.render(PACKAGE_NAMES[i], True, dc)

    surface.blit(tag, tag.get_rect(center=(x, y - 28)))


def draw_alt_bars(surface, env, font_sm, x0, y0):
    """
    Draws altitude bars for the two drones.
    """

    spacing = 44
    bar_h = MAX_ALT * 20
    bar_w = 12

    for i in range(N_DRONES):
        bx = x0 + i * spacing
        cur = env.positions[i][2]

        # Draw altitude levels
        for a in range(MAX_ALT + 1):
            ry = y0 + bar_h - a * 20
            color = DRONE_COLORS[i] if a == cur else (25, 40, 50)

            pygame.draw.rect(surface, color, (bx, ry - 8, bar_w, 16), border_radius=3)

            # Show the current altitude number
            if a == cur:
                lv = font_sm.render(str(a), True, (255, 255, 255))
                surface.blit(lv, (bx + bar_w + 4, ry - 7))

        # Drone label under the bar
        s = font_sm.render(f"D{i}", True, DRONE_COLORS[i])
        surface.blit(s, (bx, y0 + bar_h + 8))

    # Altitude label
    s = font_sm.render("ALT", True, C_TEXT_DIM)
    surface.blit(s, (x0, y0 - 18))

# RIGHT PANEL
# This function draws the information panel on the right side
# of the pygame window.

def draw_panel(surface, server, hist, episode,
               total_reward, env, font_md, font_sm, win_w, win_h, fps):
    # Panel position and width
    px = win_w - 225
    pw = 215

    # Panel background
    pygame.draw.rect(surface, C_PANEL_BG, (px - 8, 0, pw + 16, win_h))

    # Vertical separator between the map and the panel
    pygame.draw.line(surface, C_SEP, (px - 8, 0), (px - 8, win_h), 1)

    def lbl(text, y, color=C_TEXT_DIM, f=font_sm):
        # Small helper to draw labels
        surface.blit(f.render(text, True, color), (px, y))

    def val(text, y, color=C_TEXT, f=font_md):
        # Small helper to draw values
        surface.blit(f.render(text, True, color), (px, y))

    def sep(y):
        # Draws a horizontal line to separate sections
        pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1)
        return y + 8

    # Initial y position in the panel
    y = 25

    # Title and general information
    lbl("DRONE 3D RL  v3", y, C_TEXT, font_md);
    y += 20
    lbl(f"Tablero {COLS}x{ROWS}x{MAX_ALT}  |  2 drones", y);
    y += 18
    lbl(f"Servidor en {SERVER_POS} (esq. derecha)  |  Helipad {HELIPAD_POS}", y);
    y += 20
    y = sep(y)

    # Episode number
    lbl("EPISODIO", y);
    y += 13;
    val(str(episode), y);
    y += 21

    # Total reward of the current episode
    lbl("REWARD EP", y);
    y += 13
    rc = C_REWARD_POS if total_reward >= 0 else C_REWARD_NEG
    val(f"{total_reward:.0f}", y, rc);
    y += 21

    # Current epsilon value
    lbl("EPSILON", y);
    y += 13;
    val(f"{server.epsilon:.3f}", y);
    y += 21

    # Current steps of the episode
    lbl("PASOS", y);
    y += 13;
    val(f"{env.steps}/{MAX_STEPS}", y);
    y += 21

    # Number of Q-states learned by each drone
    qsz = " / ".join(str(s) for s in server.q_sizes())
    lbl("Q-STATES", y);
    y += 13;
    val(qsz, y);
    y += 21
    y = sep(y)

    # Drone status section
    lbl("ESTADO DRONES", y, C_TEXT);
    y += 16

    for i in range(N_DRONES):
        # Current position and last reward of this drone
        r, c, a = env.positions[i]
        rw = env.last_rewards[i]

        # Text depends on the current phase of the drone
        if env.at_helipad[i]:
            estado, ec = "helipad ok", C_HELIPAD_H
        elif env.delivered[i]:
            estado, ec = "-> helipad", (255, 220, 80)
        else:
            estado, ec = f"-> {PACKAGE_NAMES[i]}", DRONE_COLORS[i]

        # Reward color depends on whether it is positive or negative
        rw_c = C_REWARD_POS if rw >= 0 else C_REWARD_NEG

        # Drone position and state
        s = font_sm.render(f"D{i} ({r},{c},{a})  {estado}", True, ec)
        surface.blit(s, (px, y));
        y += 14

        # Last reward of this drone
        s2 = font_sm.render(f"   rew step: {rw:.1f}", True, rw_c)
        surface.blit(s2, (px, y));
        y += 16

    # Server information section
    pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1);
    y += 8
    lbl("SERVIDOR", y, C_TEXT);
    y += 14

    # Server position
    s = font_sm.render(f"pos {SERVER_POS}  esq. derecha del mapa", True, C_SERVER)
    surface.blit(s, (px, y));
    y += 14

    # Server role
    s2 = font_sm.render(f"emite acciones para D0 y D1", True, C_TEXT_DIM)
    surface.blit(s2, (px, y));
    y += 16
    y = sep(y)

    # Packages section
    lbl("PAQUETES", y, C_TEXT);
    y += 15

    for i in range(N_DRONES):
        done = env.delivered[i]
        color = PKG_TOP[i] if not done else (60, 75, 65)
        mark = "OK entregado" if done else f"en {PACKAGE_POS[i]}"

        # Package status
        s = font_sm.render(f"{PACKAGE_NAMES[i]}: {mark}", True, color)
        surface.blit(s, (px, y));
        y += 14

    # Number of drones that reached the helipad
    hp_n = sum(env.at_helipad)
    hp_c = C_HELIPAD_H if hp_n == N_DRONES else C_TEXT_DIM

    s = font_sm.render(f"Helipad: {hp_n}/{N_DRONES} drones", True, hp_c)
    surface.blit(s, (px, y));
    y += 14
    y = sep(y)

    # Reward graph section
    if len(hist) > 5:
        lbl("REWARD (200 ep)", y);
        y += 13

        # We only show the last 200 episodes to keep the graph small
        recent = hist[-200:]

        mn, mx = min(recent), max(recent)

        # Avoid division by zero if all rewards are equal
        rng = max(mx - mn, 1)

        gh, gw = 50, pw - 14

        # Draw the reward curve
        for t in range(1, len(recent)):
            x1 = px + int((t - 1) / len(recent) * gw)
            x2 = px + int(t / len(recent) * gw)

            y1 = y + gh - int((recent[t - 1] - mn) / rng * gh)
            y2 = y + gh - int((recent[t] - mn) / rng * gh)

            pygame.draw.line(surface, DRONE_COLORS[0], (x1, y1), (x2, y2), 1)

        # Graph border
        pygame.draw.rect(surface, C_SEP, (px, y, gw, gh), 1)

        y += gh + 4

        # Average and best reward of the recent episodes
        lbl(f"media {np.mean(recent):.1f}  mejor {mx:.0f}", y);
        y += 15

    y = sep(y)

    # Legend section
    lbl("LEYENDA", y, C_TEXT);
    y += 14

    for i in range(N_DRONES):
        # Drone color and target package
        pygame.draw.circle(surface, DRONE_COLORS[i], (px + 6, y + 6), 6)
        lbl(f"   Dron {i} -> {PACKAGE_NAMES[i]}", y, DRONE_COLORS[i]);
        y += 14

    # Server legend
    pygame.draw.rect(surface, C_SERVER_D, (px + 2, y + 2, 12, 12), border_radius=2)
    lbl(f"   Servidor (gestor)", y, C_SERVER);
    y += 14

    # Helipad legend
    pygame.draw.circle(surface, C_HELIPAD_H, (px + 6, y + 6), 6)
    lbl(f"   Helipad H  {HELIPAD_POS}", y, C_HELIPAD_H);
    y += 14

    # Bottom controls section
    y = win_h - 58

    pygame.draw.line(surface, C_SEP, (px, y), (px + pw - 10, y), 1);
    y += 8

    # FPS controls
    lbl(f"FPS: {fps}   +/- ajustar", y);
    y += 15

    # Keyboard controls
    lbl("R reentrenar  |  ESC salir", y)


# MAIN VISUAL LOOP
# This function opens the pygame window and shows the trained drones
# moving in the custom environment.

def run_visual(server, hist, train_fn=None):
    """
    Main visual loop.

    It uses the trained server to choose actions for the drones.
    If the user presses R, the model is trained again using train_fn.
    """

    # Window size
    WIN_W, WIN_H = 980, 730

    # Initialize pygame and create the window
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Drone 3D RL v3  |  Servidor + 2 drones  |  Helipad H")
    clock = pygame.time.Clock()

    # Fonts used in the interface.
    # If the system cannot load these fonts, default pygame fonts are used.
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

    # Create a new environment only for the visual simulation
    env = DroneEnv()

    # Visual episode counter
    episode = 0

    # Initial animation speed
    fps = FPS_DEFAULT

    # This variable becomes True when the user wants to retrain
    retrain = False

    # Infinite loop: the visual simulation continues until the user closes it
    while True:

        # Reset the environment for a new visual episode
        env.reset()

        # Initial state keys for both drones
        sks = [env.state_key(i) for i in range(N_DRONES)]

        # Total reward obtained in this visual episode
        total_reward = 0.0

        done = False

        # Loop of one episode
        while not done:

            # Check keyboard and window events
            for event in pygame.event.get():

                # Close window
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()

                # Keyboard controls
                if event.type == pygame.KEYDOWN:

                    # ESC closes the simulation
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit();
                        sys.exit()

                    # R retrains the model
                    if event.key == pygame.K_r:
                        retrain = True;
                        done = True

                    # Increase FPS
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        fps = min(30, fps + 1)

                    # Decrease FPS
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        fps = max(1, fps - 1)

            # If retrain was selected, we leave the episode loop
            if retrain:
                break

            # The trained server chooses the best action for each drone
            acts = [server.act(i, sks[i], greedy=True) for i in range(N_DRONES)]

            # Apply the actions in the environment
            rwds, term, trunc = env.step_all(acts)

            # Update state keys after the movement
            sks = [env.state_key(i) for i in range(N_DRONES)]

            # Add both drone rewards to the episode reward
            total_reward += sum(rwds)

            # The episode ends if it finishes correctly or reaches max steps
            done = term or trunc

            # Clear screen with background color
            screen.fill(C_BG)

            # Draw map, buildings, packages, server and drones
            draw_scene(screen, env, font_h, font_sm)

            # Draw altitude bars
            draw_alt_bars(screen, env, font_sm, 22, WIN_H - 210)

            # Draw title
            title = font_lg.render(
                "Drone 3D RL v3  |  Servidor + Q indep.  |  shaping + anti-bucle",
                True, C_TEXT)
            screen.blit(title, (18, 12))

            # Draw information panel on the right
            draw_panel(screen, server, hist, episode,
                       total_reward, env, font_md, font_sm, WIN_W, WIN_H, fps)

            # Update pygame window
            pygame.display.flip()

            # Control animation speed
            clock.tick(fps)

        # If the user pressed R, train again and restart the visual simulation
        if retrain:
            retrain = False
            pygame.quit()
            print("\nRe-entrenando (mapa identico)...")
            server, hist = train_fn()
            run_visual(server, hist, train_fn)
            return

        # Move to next visual episode
        episode += 1

        # Small pause between episodes
        time.sleep(0.4)
