# CREAR ENTORNO V2
def make_env():
    env = gym.make("Taxi-v3")

    env = CustomRewardWrapper(env)
    env = CustomTimeLimitWrapper(env, MAX_STEPS)
    env = CustomObservationWrapper(env, MAX_STEPS)

    return env


# Q-LEARNING
n_states = 500
n_actions = 6

Q = np.zeros((n_states, n_actions))


def choose_action(state, epsilon):
    """
    Política epsilon-greedy
    """
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    return int(np.argmax(Q[state]))



# ENTRENAMIENTO

def train():
    global EPSILON

    env = make_env()

    rewards_history = []
    steps_history = []

    print("====================================")
    print("Entrenando versión 2 con wrappers...")
    print("====================================")

    for episode in range(1, EPISODES + 1):

        obs, info = env.reset()
        state = int(obs[0])

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = choose_action(state, EPSILON)

            next_obs, reward, terminated, truncated, info = env.step(action)

            next_state = int(next_obs[0])

            # Fórmula Q-learning
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
                f"Episodio {episode:4d} | "
                f"epsilon={EPSILON:.3f} | "
                f"reward medio={np.mean(rewards_history[-500:]):.2f} | "
                f"pasos medios={np.mean(steps_history[-500:]):.2f}"
            )

    env.close()

    print("Entrenamiento finalizado.\n")



# EVALUACIÓN

def evaluate():
    env = make_env()

    rewards = []
    steps_list = []
    successes = 0

    print("==================================")
    print("Evaluación final")
    print("==================================")

    for episode in range(EVAL_EPISODES):

        obs, info = env.reset()
        state = int(obs[0])

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = int(np.argmax(Q[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)

            state = int(next_obs[0])

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



# DEMO VISUAL

def demo():
    env = gym.make("Taxi-v3", render_mode="human")

    print("\n====================================")
    print("Demostración política aprendida")
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

