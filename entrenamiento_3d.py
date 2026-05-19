import numpy as np
from collections import defaultdict

from drone_env_3d import (
    DroneEnv,
    N_DRONES, N_ACTIONS, TRAIN_EPISODES, MAX_STEPS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    SERVER_POS, HELIPAD_POS, PACKAGE_NAMES,
    COLS, ROWS, MAX_ALT,
)


# ─────────────────────────────────────────────
# AGENTE SERVIDOR (Q-learning independiente por dron)
# ─────────────────────────────────────────────

class ServerAgent:
    """
    Servidor central en (0,0) que gestiona 2 Q-learners independientes.
    Cada Q-table tiene estado (fila, col, alt, fase) — espacio pequeno.
    """

    def __init__(self):
        self.pos = SERVER_POS
        self.qtables = [defaultdict(lambda: np.zeros(N_ACTIONS))
                        for _ in range(N_DRONES)]
        self.epsilon = EPSILON_START

    def act(self, drone_id, sk, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.qtables[drone_id][sk]))

    def update(self, drone_id, s, a, r, sn, done):
        q = self.qtables[drone_id]
        best = 0.0 if done else float(np.max(q[sn]))
        q[s][a] += ALPHA * (r + GAMMA * best - q[s][a])

    def decay(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def q_sizes(self):
        return [len(qt) for qt in self.qtables]


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

def train(episodes=TRAIN_EPISODES):
    """
    Entrena el ServerAgent usando Q-learning independiente por dron.

    En cada episodio, el servidor selecciona acciones para ambos drones,
    actualiza sus Q-tables y reduce epsilon gradualmente para que la
    exploracion disminuya con el tiempo.
    """
    env = DroneEnv()
    server = ServerAgent()
    hist = []
    best = -np.inf

    print(f"\n{'=' * 60}")
    print(f"  Drone 3D RL v3  |  {COLS}x{ROWS}x{MAX_ALT}  |  2 drones")
    print(f"  Servidor en {SERVER_POS}  |  Helipad en {HELIPAD_POS}")
    print(f"  {episodes} episodios  |  Q independiente por dron")
    print(f"  Shaping + anti-bucle + sin accion esperar")
    print(f"{'=' * 60}")

    for ep in range(episodes):
        env.reset()
        sks = [env.state_key(i) for i in range(N_DRONES)]
        total = 0.0

        for _ in range(MAX_STEPS):
            acts = [server.act(i, sks[i]) for i in range(N_DRONES)]
            rwds, term, trunc = env.step_all(acts)
            new_sks = [env.state_key(i) for i in range(N_DRONES)]
            done = term or trunc

            for i in range(N_DRONES):
                server.update(i, sks[i], acts[i], rwds[i], new_sks[i], done)

            sks = new_sks
            total += sum(rwds)
            if done:
                break

        server.decay()
        hist.append(total)
        if total > best:
            best = total

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

    print(f"\n  Completado. Q-states: {' + '.join(str(s) for s in server.q_sizes())}")
    print(f"{'=' * 60}\n")
    return server, hist
