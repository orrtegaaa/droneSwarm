import sys
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# CONFIGURACIÓN DEL ENTORNO
GRID_SIZE = 8
CELL_PX = 80
MARGIN = 40
FPS = 6
MAX_STEPS = 150

# RECOMPENSAS
R_DELIVERY = 20
R_STEP = -1
R_BUILDING = -15
R_REVISIT = -5

# COLORES
BG = (25, 25, 25)
FG = (220, 220, 220)
GRID = (80, 80, 80)
BLOCK = (120, 120, 120)
DRONE = (240, 240, 240)
PENDING = (180, 180, 180)
DONE = (100, 100, 100)


class DroneSwarmEnv(gym.Env):
    """
    Entorno personalizado de dron sobre una cuadrícula 8x8.

    El dron debe entregar en varios puntos evitando edificios.
    """

    DELIVERY_POINTS = {"A": (0, 7), "B": (7, 0), "C": (5, 6)}
    BUILDINGS = {(2, 4), (1, 3), (4, 2), (6, 2), (6, 5), (5, 5), (5, 1)}
    START_POS = (0, 0)

    # arriba, abajo, izquierda, derecha, quedarse quieto
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=GRID_SIZE,
            shape=(5,),
            dtype=np.int32
        )

        self.pos = list(self.START_POS)
        self.pending = [1, 1, 1]
        self.steps = 0
        self.last_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = list(self.START_POS)
        self.pending = [1, 1, 1]
        self.steps = 0
        self.last_reward = 0

        return self._obs(), {}

    def step(self, action):
        dr, dc = self.DELTAS[action]

        new_r = max(0, min(GRID_SIZE - 1, self.pos[0] + dr))
        new_c = max(0, min(GRID_SIZE - 1, self.pos[1] + dc))

        reward = R_STEP

        if (new_r, new_c) in self.BUILDINGS:
            reward += R_BUILDING
        else:
            self.pos = [new_r, new_c]

        for i, point in enumerate(self.DELIVERY_POINTS.values()):
            if tuple(self.pos) == point:
                if self.pending[i] == 1:
                    self.pending[i] = 0
                    reward += R_DELIVERY
                else:
                    reward += R_REVISIT

        self.steps += 1
        self.last_reward = reward

        terminated = not any(self.pending)
        truncated = self.steps >= MAX_STEPS

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return np.array([self.pos[0], self.pos[1], *self.pending], dtype=np.int32)

    def state_key(self):
        """
        Estado discreto para usarlo como clave en la tabla Q.
        """
        return (self.pos[0], self.pos[1], self.pending[0], self.pending[1], self.pending[2])


def make_env():
    return DroneSwarmEnv()


def cell_center(row, col):
    x = MARGIN + col * CELL_PX + CELL_PX // 2
    y = MARGIN + row * CELL_PX + CELL_PX // 2
    return x, y


def draw_env(screen, env, font):
    screen.fill(BG)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(
                MARGIN + c * CELL_PX,
                MARGIN + r * CELL_PX,
                CELL_PX,
                CELL_PX
            )

            if (r, c) in env.BUILDINGS:
                pygame.draw.rect(screen, BLOCK, rect)
            else:
                pygame.draw.rect(screen, GRID, rect, 1)

    for i, (name, pos) in enumerate(env.DELIVERY_POINTS.items()):
        x, y = cell_center(*pos)
        color = DONE if env.pending[i] == 0 else PENDING
        pygame.draw.circle(screen, color, (x, y), 20)

        label = font.render(name, True, BG)
        screen.blit(label, label.get_rect(center=(x, y)))

    x, y = cell_center(*env.pos)
    pygame.draw.circle(screen, DRONE, (x, y), 14)

    status = font.render(
        f"Pasos: {env.steps}   Reward: {env.last_reward}",
        True,
        FG
    )
    screen.blit(status, (MARGIN, 10))

    pygame.display.flip()


def demo_visual(get_action_fn):
    """
    Demo visual recibiendo una función que dado un estado devuelve una acción.
    """
    pygame.init()

    size = MARGIN * 2 + GRID_SIZE * CELL_PX
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Drone Swarm RL")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)

    env = make_env()

    while True:
        env.reset()
        state = env.state_key()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            action = get_action_fn(state)
            _, _, terminated, truncated, _ = env.step(action)

            state = env.state_key()
            done = terminated or truncated

            draw_env(screen, env, font)
            clock.tick(FPS)

        time.sleep(0.5)