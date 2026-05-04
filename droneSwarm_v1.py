import sys
import time
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

GRID_SIZE = 7
CELL_PX = 80
MARGIN = 40
FPS = 6
TRAIN_EPISODES = 8000
MAX_STEPS = 150

R_DELIVERY = 20
R_STEP = -1
R_BUILDING = -15
R_REVISIT = -5

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995

BG = (25, 25, 25)
FG = (220, 220, 220)
GRID = (80, 80, 80)
BLOCK = (120, 120, 120)
DRONE = (240, 240, 240)
PENDING = (180, 180, 180)
DONE = (100, 100, 100)


class DroneSwarmEnv(gym.Env):
    DELIVERY_POINTS = {'A': (0, 6), 'B': (6, 0), 'C': (6, 6)}
    BUILDINGS = {(2, 2), (2, 4), (4, 2), (4, 4), (3, 3), (1, 5), (5, 1)}
    START_POS = (0, 0)
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=GRID_SIZE, shape=(5,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = list(self.START_POS)
        self.pending = [1, 1, 1]
        self.steps = 0
        self.last_reward = 0
        return self._obs(), {}

    def step(self, action):
        dr, dc = self.DELTAS[action]
        r = max(0, min(GRID_SIZE - 1, self.pos[0] + dr))
        c = max(0, min(GRID_SIZE - 1, self.pos[1] + dc))
        reward = R_STEP

        if (r, c) in self.BUILDINGS:
            reward += R_BUILDING
        else:
            self.pos = [r, c]

        for i, point in enumerate(self.DELIVERY_POINTS.values()):
            if tuple(self.pos) == point:
                if self.pending[i]:
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
        return np.array([*self.pos, *self.pending], dtype=np.int32)

    def state_key(self):
        return (*self.pos, *self.pending)


class QLearningAgent:
    def __init__(self, n_actions):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.epsilon = EPSILON_START
        self.n_actions = n_actions

    def choose_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, s, a, r, s2, done):
        target = r + (0 if done else GAMMA * np.max(self.q_table[s2]))
        self.q_table[s][a] += ALPHA * (target - self.q_table[s][a])

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def train(episodes=TRAIN_EPISODES):
    env = DroneSwarmEnv()
    agent = QLearningAgent(env.action_space.n)
    history = []

    for ep in range(episodes):
        env.reset()
        state = env.state_key()
        total = 0

        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_key()
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total += reward
            if terminated or truncated:
                break

        agent.decay_epsilon()
        history.append(total)

        if (ep + 1) % 500 == 0:
            print(f'Episodio {ep + 1}: media {np.mean(history[-500:]):.1f}')

    return agent, history


def cell_center(row, col):
    return MARGIN + col * CELL_PX + CELL_PX // 2, MARGIN + row * CELL_PX + CELL_PX // 2


def draw(screen, env, font):
    screen.fill(BG)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(MARGIN + c * CELL_PX, MARGIN + r * CELL_PX, CELL_PX, CELL_PX)
            pygame.draw.rect(screen, BLOCK if (r, c) in env.BUILDINGS else GRID, rect, 0 if (r, c) in env.BUILDINGS else 1)

    for i, (name, pos) in enumerate(env.DELIVERY_POINTS.items()):
        x, y = cell_center(*pos)
        pygame.draw.circle(screen, DONE if not env.pending[i] else PENDING, (x, y), 20)
        label = font.render(name, True, BG)
        screen.blit(label, label.get_rect(center=(x, y)))

    x, y = cell_center(*env.pos)
    pygame.draw.circle(screen, DRONE, (x, y), 14)

    status = font.render(f'Pasos: {env.steps}  Reward: {env.last_reward}', True, FG)
    screen.blit(status, (MARGIN, 10))
    pygame.display.flip()


def run_visual(agent):
    pygame.init()
    size = MARGIN * 2 + GRID_SIZE * CELL_PX
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption('Drone Swarm RL')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    env = DroneSwarmEnv()

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

            action = agent.choose_action(state, greedy=True)
            _, _, terminated, truncated, _ = env.step(action)
            state = env.state_key()
            done = terminated or truncated
            draw(screen, env, font)
            clock.tick(FPS)

        time.sleep(0.5)


if __name__ == '__main__':
    agent, _ = train()
    run_visual(agent)