import gymnasium as gym
import numpy as np
import random
import time

# Fixed parameters
EPISODES = 5000
EVAL_EPISODES = 100
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# BASE ENVIRONMENT
# Here we create a function to initialize the environment, 
#   which will be used in both training and evaluation phases. 
def make_env():
    return gym.make("Taxi-v3")

# Q-LEARNING
# We initialize the Q-table, which is a 2D array where rows represent states and columns represent actions.
n_states = 500
n_actions = 6
Q = np.zeros((n_states, n_actions))

# Action selection function using epsilon-greedy strategy, which allows the agent to balance exploration during training.
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    return int(np.argmax(Q[state]))

# TRAINING
# The training function implements the Q-learning algorithm, 
#   where the agent interacts with the environment, 
#   updates the Q-table based on the rewards received, and decays the exploration rate (epsilon) over time.
def train():
    global EPSILON

    env = make_env()

    rewards_history = []
    steps_history = []

    print("====================================")
    print("     Starting training...")
    print("====================================")

    for episode in range(1, EPISODES + 1):

        obs, info = env.reset()
        state = obs

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(state, EPSILON)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs
            # Q-learning update rule
            # The Q-table is updated based on the reward received and the maximum future reward from the next state,
            Q[state, action] = Q[state, action] + ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state, action]
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
                f"Episode {episode:4d} | "
                f"epsilon={EPSILON:.3f} | "
                f"average reward={np.mean(rewards_history[-500:]):.2f} | "
                f"average steps={np.mean(steps_history[-500:]):.2f}"
            )

    env.close()
    print("Training completed.\n")


# EVALUATION 
# The evaluation function tests the trained agent over a set number of episodes, 
#   tracking the average reward, average steps taken, and success rate.
def evaluate():
    env = make_env()

    rewards = []
    steps_list = []
    successes = 0

    print("==================================")
    print("     Final evaluation")
    print("==================================")

    for episode in range(EVAL_EPISODES):

        obs, info = env.reset()
        state = obs

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = int(np.argmax(Q[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            state = next_obs

            total_reward += reward
            steps += 1

            done = terminated or truncated

            if terminated:
                successes += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average steps: {np.mean(steps_list):.2f}")
    print(f"Success rate: {(successes / EVAL_EPISODES) * 100:.2f}%")

    env.close()

#Visual demo 
# The demo function runs a single episode with the trained agent, 
#   rendering the environment to visually demonstrate the agent's performance.
def demo():
    env = gym.make("Taxi-v3", render_mode="human")

    print("\n====================================")
    print("     Trained agent demonstration")
    print("====================================\n")

    obs, info = env.reset()
    state = obs

    done = False
    total_reward = 0

    while not done:
        action = int(np.argmax(Q[state]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = obs
        done = terminated or truncated

        time.sleep(0.4)

    print("Recompensa final:", total_reward)

    env.close()