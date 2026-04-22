import sys
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# ENVIROMENT CONFIGURATION
GRID_SIZE = 8
CELL_PX = 80
MARGIN = 40
FPS = 6
MAX_STEPS = 150

# REWARDS
R_DELIVERY = 20
R_STEP = -1
R_BUILDING = -15
R_REVISIT = -5

# COLORS FOR THE VISUAL PRESENTATION
BG = (25, 25, 25)
FG = (220, 220, 220)
GRID = (80, 80, 80)
BLOCK = (120, 120, 120)
DRONE = (240, 240, 240)
PENDING = (180, 180, 180)
DONE = (100, 100, 100)


class DroneSwarmEnv(gym.Env):
    """
    Custom drone delivery environment WITH AN 8X8 GRID.

    The drone starts from a fixed position and must complete
    deliveries at different points while avoiding buildings.
    """

    # DELIVERY LOCATIONS
    DELIVERY_POINTS = {"A": (0, 7), "B": (7, 0), "C": (5, 6)}

    # CELLS WITH BUILDINGS
    BUILDINGS = {(2, 4), (1, 3), (4, 2), (6, 2), (6, 5), (5, 5), (5, 1)}

    # INITIAL POSITION OF THE DRONE
    START_POS = (0, 0)

    # ACTIONS: UP, DOWN, RIGHT, LEFT
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self):
        super().__init__()

        # DICRETE ACTION SPACE WITH 5 POSIBLE ACTIONS
        self.action_space = spaces.Discrete(5)

        # OBSERVATION SPACE:
        # [ROW, COL, PENDING_A, PENDING_B, PENDING_C]
        self.observation_space = spaces.Box(
            low=0,
            high=GRID_SIZE,
            shape=(5,),
            dtype=np.int32
        )

        # INTERNAL STATE VARIABLES
        self.pos = list(self.START_POS)
        self.pending = [1, 1, 1]
        self.steps = 0
        self.last_reward = 0

    def reset(self, seed=None, options=None):
        '''
        RESETS ENVIROMENT TO INITIAL STATE
        '''

        super().reset(seed=seed)

        self.pos = list(self.START_POS)
        self.pending = [1, 1, 1]
        self.steps = 0
        self.last_reward = 0

        return self._obs(), {}

    def step(self, action):
        '''
        Apply one action to the environment.

        The drone moves according to the selected action, receives
        a reward and the episode may finish if all deliveries are
        completed or the step limit is reached.

        '''

        dr, dc = self.DELTAS[action]

        # COMPUTE THE CANDIDATE NEXT POSITION
        new_r = max(0, min(GRID_SIZE - 1, self.pos[0] + dr))
        new_c = max(0, min(GRID_SIZE - 1, self.pos[1] + dc))

        # STEP PENALTY, DEFAULT
        reward = R_STEP

        # IF THE DRONE HITS A BUILDING APPLIES A PENALTY AND KEEPS POSITION
        if (new_r, new_c) in self.BUILDINGS:
            reward += R_BUILDING
        else:
            self.pos = [new_r, new_c]

        # CHECK WETHER THE DRONE HAS REACHED A DELIVERY POINT
        for i, point in enumerate(self.DELIVERY_POINTS.values()):
            if tuple(self.pos) == point:
                if self.pending[i] == 1:
                    # FIRST VISIT SUCCESSFUL DELIVERY
                    self.pending[i] = 0
                    reward += R_DELIVERY
                else:
                    # REPEATED VISIT: UNNECESARY REVISIT PENALTY
                    reward += R_REVISIT

        self.steps += 1
        self.last_reward = reward

        # EPISODE FINISHES WHEN ALL DELIVERIES ARE DONE
        terminated = not any(self.pending)

        # EPISODE FINISHES IF MAX NUMBER OF STEPS ARE COMPLETED
        truncated = self.steps >= MAX_STEPS

        # RETURNS NEW OBSERVATION, REWARDS EARNED AND FLAGS TO KNOW IF THE EPISODE ENDED
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        # OBSERVATION ARRAY WITH: CURRENT ROW, CURRENT COLUMN AND PENDING DELIVERIES
        return np.array([self.pos[0], self.pos[1], *self.pending], dtype=np.int32)

    def state_key(self):
        """
        Return a discrete version of the current state.

        This tuple is used as the key in the Q-table.
        """

        return (self.pos[0], self.pos[1], self.pending[0], self.pending[1], self.pending[2])


def make_env():
    # NEW INSTANCE OF THE CUSTOM ENVIROMENT
    return DroneSwarmEnv()


def cell_center(row, col):
    # COMPUTE THE COORDINATES OF THE CENTER OF A GRID CELL
    x = MARGIN + col * CELL_PX + CELL_PX // 2
    y = MARGIN + row * CELL_PX + CELL_PX // 2
    return x, y


def draw_env(screen, env, font):
    # FILL BACKGROUND BEFORE NEW FRAME IS CREATED
    screen.fill(BG)

    # DRAW GRID CELLS AND BUILDINGS
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(
                MARGIN + c * CELL_PX,
                MARGIN + r * CELL_PX,
                CELL_PX,
                CELL_PX
            )

            # BUILDINGS AS FILLED BLOCKS
            if (r, c) in env.BUILDINGS:
                pygame.draw.rect(screen, BLOCK, rect)
            else:
                # DRAW NORMAL CELLS AS GRID OUTLINES
                pygame.draw.rect(screen, GRID, rect, 1)

    # DRAW DELIVERY POINTS AND INDICATE IF THEY ARE PENDING OR COMPLETED
    for i, (name, pos) in enumerate(env.DELIVERY_POINTS.items()):
        x, y = cell_center(*pos)
        color = DONE if env.pending[i] == 0 else PENDING
        pygame.draw.circle(screen, color, (x, y), 20)

        # LABEL OF EACH DELIVERY POINT
        label = font.render(name, True, BG)
        screen.blit(label, label.get_rect(center=(x, y)))

    # DRONE IN ITS CURRENT POSITION
    x, y = cell_center(*env.pos)
    pygame.draw.circle(screen, DRONE, (x, y), 14)

    # CURRENT STEP COUNT AND LAST REWARD ON SCREEN
    status = font.render(
        f"Pasos: {env.steps}   Reward: {env.last_reward}",
        True,
        FG
    )
    screen.blit(status, (MARGIN, 10))

    # UPDATE DISPLAY WITH A NEW FRAME
    pygame.display.flip()


def demo_visual(get_action_fn):
    """
    Run a visual demonstration of the environment.

    The function receives another function that takes a state
    and returns the action chosen by the agent.
    """
    # INITIALIZE PYGAME AND CREATE VISUAL WINDOW
    pygame.init()

    size = MARGIN * 2 + GRID_SIZE * CELL_PX
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Drone Swarm RL")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)

    # NEW ENVIROMENT FOR THE DEMO
    env = make_env()

    # CONTINUE SHOW EPISODES UNTIL THE USER CLOSES THE WINDOW
    while True:
        # RESETS ENVIROMENT BEFORE EACH EPISODE
        env.reset()
        state = env.state_key()
        done = False

        while not done:
            # ALLOW USER TO CLOSE OR EXIT WITH ESC
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            # CHOOSE ACTION WITH LEARNED POLICY
            action = get_action_fn(state)
            _, _, terminated, truncated, _ = env.step(action)

            # UPDATE STATE AND CHECK IF EPISODE HAS FINISHED
            state = env.state_key()
            done = terminated or truncated

            # DRAWS UPDATED ENVIROMENT
            draw_env(screen, env, font)
            clock.tick(FPS)

        # PAUSE BEFORE NEXT EPISODE
        time.sleep(0.5)