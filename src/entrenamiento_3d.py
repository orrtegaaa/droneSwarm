import numpy as np
from collections import defaultdict

from drone_env_3d import (
    DroneEnv,
    N_DRONES, N_ACTIONS, TRAIN_EPISODES, MAX_STEPS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    SERVER_POS, HELIPAD_POS, PACKAGE_NAMES,
    COLS, ROWS, MAX_ALT,
)


# SERVER AGENT
# The server is the central controller of the drones.
# It uses one independent Q-table for each drone.

class ServerAgent:
    """
    Central server placed at SERVER_POS.
    It controls the drones using independent Q-learning.

    Each drone has its own Q-table, so they do not share exactly
    the same learning information.
    """

    def __init__(self):
        # Position of the server in the environment
        self.pos = SERVER_POS

        # We create one Q-table for each drone.
        # defaultdict avoids errors when a new state appears for the first time.
        self.qtables = [defaultdict(lambda: np.zeros(N_ACTIONS))
                        for _ in range(N_DRONES)]

        # Epsilon controls the exploration level.
        # At the beginning the agent explores more.
        self.epsilon = EPSILON_START

    def act(self, drone_id, sk, greedy=False):
        # If greedy is False, the drone can explore with probability epsilon.
        # This means it may choose a random action.
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)

        # If it does not explore, it chooses the action with the highest Q-value.
        return int(np.argmax(self.qtables[drone_id][sk]))

    def update(self, drone_id, s, a, r, sn, done):
        # We select the Q-table of the drone that has acted.
        q = self.qtables[drone_id]

        # If the episode is finished, there is no future reward.
        # If not, we take the best Q-value of the next state.
        best = 0.0 if done else float(np.max(q[sn]))

        # Q-learning update formula.
        q[s][a] += ALPHA * (r + GAMMA * best - q[s][a])

    def decay(self):
        # Epsilon is reduced after each episode.
        # This makes the agent explore less over time.
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def q_sizes(self):
        # Returns how many states have been visited by each drone.
        return [len(qt) for qt in self.qtables]



# TRAINING
# This function trains the server agent in the drone environment.

def train(episodes=TRAIN_EPISODES):
    """
    Trains the ServerAgent using independent Q-learning for each drone.

    In every episode, the server chooses actions for both drones.
    Then the environment returns the rewards and the new states.
    With that information, each Q-table is updated.
    """

    # Create the custom drone environment
    env = DroneEnv()

    # Create the central server agent
    server = ServerAgent()

    # This list stores the total reward obtained in each episode
    hist = []

    # Best reward found during training
    best = -np.inf

    # Initial information shown before training starts
    print(f"\n{'=' * 60}")
    print(f"  Drone 3D RL v3  |  {COLS}x{ROWS}x{MAX_ALT}  |  2 drones")
    print(f"  Servidor en {SERVER_POS}  |  Helipad en {HELIPAD_POS}")
    print(f"  {episodes} episodios  |  Q independiente por dron")
    print(f"  Shaping + anti-bucle + sin accion esperar")
    print(f"{'=' * 60}")

    # Main training loop
    for ep in range(episodes):

        # Reset the environment at the start of each episode
        env.reset()

        # Get the initial state key for each drone
        sks = [env.state_key(i) for i in range(N_DRONES)]

        # Total reward of this episode
        total = 0.0

        # Each episode can last at most MAX_STEPS
        for _ in range(MAX_STEPS):

            # The server chooses one action for each drone
            acts = [server.act(i, sks[i]) for i in range(N_DRONES)]

            # The environment executes all drone actions at the same time
            rwds, term, trunc = env.step_all(acts)

            # Get the new state of each drone after moving
            new_sks = [env.state_key(i) for i in range(N_DRONES)]

            # The episode finishes if it terminates normally or is truncated
            done = term or trunc

            # Update the Q-table of each drone separately
            for i in range(N_DRONES):
                server.update(i, sks[i], acts[i], rwds[i], new_sks[i], done)

            # Update current states
            sks = new_sks

            # Add the rewards of both drones
            total += sum(rwds)

            # If the episode is finished, we stop this episode
            if done:
                break

        # Reduce epsilon after each episode
        server.decay()

        # Save the total reward of the episode
        hist.append(total)

        # Update the best result if this episode is better
        if total > best:
            best = total

        # Show progress every 1000 episodes
        if (ep + 1) % 1000 == 0:
            avg = np.mean(hist[-1000:])
            pct = 100 * (ep + 1) / episodes
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            qsz = " + ".join(str(s) for s in server.q_sizes())

            print(f"  [{bar}] {pct:5.1f}%  "
                  f"eps={server.epsilon:.3f}  "
                  f"avg={avg:7.1f}  "
                  f"best={best:.0f}  "
                  f"Q={qsz}")

    # Final training information
    print(f"\n  Completed. Q-states: {' + '.join(str(s) for s in server.q_sizes())}")
    print(f"{'=' * 60}\n")

    # Return the trained server and the reward history
    return server, hist
