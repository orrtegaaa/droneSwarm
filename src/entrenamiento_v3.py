import random
from collections import defaultdict

import numpy as np

from drone_env import make_env, demo_visual

# TRAINING AND LEARNING PARAMETERS
EPISODES = 8000 #NUMBER OF TRAINING EPISODES
EVAL_EPISODES = 100  #NUMBER OF

ALPHA = 0.1 #
GAMMA = 0.95 #

EPSILON = 1.0 #
EPSILON_MIN = 0.05 #
EPSILON_DECAY = 0.9995 #

# ENVIRONMENT ACTIONS
N_ACTIONS = 5

# Q-TABLE
Q = defaultdict(lambda: np.zeros(N_ACTIONS))


def choose_action(state, epsilon):
    """
    Política epsilon-greedy
    """
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(np.argmax(Q[state]))


def train():
    global EPSILON

    env = make_env()

    rewards_history = []
    steps_history = []

    print("====================================")
    print("Entrenando versión 3 personalizada...")
    print("====================================")

    for episode in range(1, EPISODES + 1):
        env.reset()
        state = env.state_key()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(state, EPSILON)

            _, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_key()

            Q[state][action] = Q[state][action] + ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            total_reward += reward
            steps += 1

            done = terminated or truncated

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        rewards_history.append(total_reward)
        steps_history.append(steps)

        if episode % 500 == 0:
            print(
                f"Episodio {episode:4d} | "
                f"epsilon={EPSILON:.3f} | "
                f"reward medio={np.mean(rewards_history[-500:]):.2f} | "
                f"pasos medios={np.mean(steps_history[-500:]):.2f}"
            )

    env.close()
    print("Entrenamiento finalizado.\n")


def evaluate():
    env = make_env()

    rewards = []
    steps_list = []
    successes = 0

    print("==================================")
    print("Evaluación final")
    print("==================================")

    for episode in range(EVAL_EPISODES):
        env.reset()
        state = env.state_key()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = int(np.argmax(Q[state]))

            _, reward, terminated, truncated, _ = env.step(action)
            state = env.state_key()

            total_reward += reward
            steps += 1

            done = terminated or truncated

            if terminated:
                successes += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    print(f"Recompensa media: {np.mean(rewards):.2f}")
    print(f"Pasos medios:     {np.mean(steps_list):.2f}")
    print(f"Tasa de éxito:    {(successes / EVAL_EPISODES) * 100:.2f}%")

    env.close()


def demo():
    print("\n====================================")
    print("Demostración política aprendida")
    print("====================================\n")

    def greedy_action(state):
        return int(np.argmax(Q[state]))

    demo_visual(greedy_action)