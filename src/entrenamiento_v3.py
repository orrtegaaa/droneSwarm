import random
from collections import defaultdict

import numpy as np

from drone_env import make_env, demo_visual

# TRAINING AND LEARNING PARAMETERS
EPISODES = 8000
EVAL_EPISODES = 100

#Q-LEARNING PARAMETERS
ALPHA = 0.1 #
GAMMA = 0.95 #

#EPSILON-GREEDY PARAMETERS
EPSILON = 1.0 #
EPSILON_MIN = 0.05 #
EPSILON_DECAY = 0.9995 #

# ENVIRONMENT ACTIONS
N_ACTIONS = 5

# Q-TABLE: EACH STATE STORES AN ARRAY WITH ONE Q-VALUE PER ACTION
Q = defaultdict(lambda: np.zeros(N_ACTIONS))


def choose_action(state, epsilon):
    """
    Select an action using an epsilon-greedy policy.

    With probability epsilon, the agent explores by choosing
    a random action. Otherwise, it exploits the best action
    currently stored in the Q-table.
    """
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(np.argmax(Q[state]))


def train():
    '''
    Train the agent in the custom environment.

    In each episode, the agent interacts with the environment,
    updates the Q-table and gradually reduces epsilon so that
    exploration decreases over time.
    '''

    global EPSILON

    env = make_env()

    rewards_history = []
    steps_history = []

    print("====================================")
    print("Entrenando versión 3 entorno personalizado...")
    print("====================================")

    for episode in range(1, EPISODES + 1):
        # RESET THE ENVIROMENT AT THE BEGINNING OF EACH EPISODE
        env.reset()
        state = env.state_key()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            # CHOOSE AN ACTION USING EPSILON-GREEDY STRATEGY
            action = choose_action(state, EPSILON)

            # EXECUTE THE ACTION IN THE ENVIROMENT
            _, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_key()

            # UPDATE THE Q-VALUE USING THE Q-LEARNING FOMRULA
            Q[state][action] = Q[state][action] + ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )

            # MOVE TO THE NEXT STATE
            state = next_state
            total_reward += reward
            steps += 1

            # EPISODE ENDS IF ITS TERMINATED OR TRUNCATED
            done = terminated or truncated

        # REDUCES EPSILON AFTER EACH EPISODE
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        rewards_history.append(total_reward)
        steps_history.append(steps)

        # PRINT PROGRESS EVERY 500 EPISODES
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
    '''
    Evaluate the learned policy without exploration.

    During evaluation, the agent always chooses the action
    with the highest Q-value for the current state.
    '''

    env = make_env()

    rewards = []
    steps_list = []
    successes = 0

    print("==================================")
    print("Evaluación final")
    print("==================================")

    for episode in range(EVAL_EPISODES):
        # RESETS ENVIROMENT FOR A NEW EVALUATION EPISODE
        env.reset()
        state = env.state_key()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            # CHOOSE THE BEST KNOWN ACTION
            action = int(np.argmax(Q[state]))

            _, reward, terminated, truncated, _ = env.step(action)
            state = env.state_key()

            total_reward += reward
            steps += 1

            done = terminated or truncated

            #COUNT SUCCESSFUL EPISODES
            if terminated:
                successes += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    # PRINT FINAL STATISTICS
    print(f"Recompensa media: {np.mean(rewards):.2f}")
    print(f"Pasos medios:     {np.mean(steps_list):.2f}")
    print(f"Tasa de éxito:    {(successes / EVAL_EPISODES) * 100:.2f}%")

    env.close()


def demo():
    '''
    Run a visual demonstration of the learned policy.

    The agent acts greedily, always selecting the best action
    according to the Q-table, and the environment is displayed
    visually using pygame.
    '''

    print("\n====================================")
    print("Demostración política aprendida")
    print("====================================\n")

    def greedy_action(state):
        # RETURN THE ACTION WITH THE HIGHEST Q-VALUE
        return int(np.argmax(Q[state]))

    demo_visual(greedy_action)