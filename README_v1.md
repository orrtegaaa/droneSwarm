# Reinforcement Learning Practice — Version 1

## Version objective

This first version implements a basic reinforcement learning agent that learns to solve the **Taxi-v3** environment from Gymnasium. The goal of the environment is for a taxi to pick up a passenger from a random position on a 5×5 grid and drop them off at their destination in as few steps as possible, maximising cumulative reward.

---

## Algorithm: Q-Learning

The agent learns through **Q-Learning**, an *off-policy* tabular reinforcement learning algorithm. The core idea is to maintain a Q-table — a lookup table that stores the expected value of taking each possible action from each possible state.

### Update rule

At every time step, the Q-table is updated using the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Symbol | Meaning |
|--------|---------|
| `α` (ALPHA) | Learning rate — controls how much weight is given to new information |
| `γ` (GAMMA) | Discount factor — weighs the importance of future rewards |
| `r` | Immediate reward received after taking action `a` in state `s` |
| `s'` | Next state reached after taking the action |

### Exploration: ε-greedy strategy

To prevent the agent from always exploiting what it already knows, an **ε-greedy** policy is used:

- With probability `ε` → a **random action** is chosen (exploration).
- With probability `1 - ε` → the **best known action** according to the Q-table is chosen (exploitation).

`ε` starts at `1.0` (full exploration) and is multiplied by a decay factor at the end of each episode until it reaches a minimum of `0.01`, progressively shifting the agent's behaviour towards exploitation of acquired knowledge.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPISODES` | 5,000 | Training episodes |
| `EVAL_EPISODES` | 100 | Evaluation episodes |
| `ALPHA` | 0.1 | Learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `EPSILON` | 1.0 | Initial exploration rate |
| `EPSILON_MIN` | 0.01 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.995 | Per-episode epsilon decay factor |

---

## Program structure

The project is split into two files:

### `main_v1.py`

Entry point of the program. It imports the three main functions from `entrenamiento_v1.py` and calls them in order:

```
main()
train()     → Agent training
evaluate()  → Evaluation of the trained agent
demo()      → Visual demonstration
```

### `entrenamiento_v1.py`

Contains all the agent logic, organised into five blocks:

#### 1. `make_env()`
Initialises and returns the `Taxi-v3` Gymnasium environment. It is used in both the training and evaluation phases to ensure both work on the same environment configuration.

#### 2. Global Q-table
A zero-initialised matrix of shape `(500, 6)` is defined — 500 possible environment states × 6 available actions (north, south, east, west, pick up, drop off). This table is shared across all three phases of the program.

#### 3. `choose_action(state, epsilon)`
Implements the ε-greedy policy. Receives the current state and the current epsilon value, and returns the action to execute.

#### 4. `train()`
Main training loop over `EPISODES` episodes. At each step:
1. An action is selected via `choose_action`.
2. The action is executed in the environment.
3. The Q-table is updated using the Bellman equation.
4. Epsilon is decayed at the end of each episode.

Every 500 episodes, a summary is printed showing the current epsilon, average reward, and average steps over the last block.

#### 5. `evaluate()`
Runs `EVAL_EPISODES` episodes using only the greedy policy (no exploration — `argmax` over the Q-table). Computes and prints:
- Average reward
- Average number of steps
- Success rate (percentage of episodes where the taxi completed the task)

#### 6. `demo()`
Runs a single episode with `render_mode="human"` to visually display the trained agent's behaviour, with a 0.4-second pause between steps for easier observation.

---

## General execution flow

```
Start -> Training (5,000 episodes) - ε decays from 1.0 → 0.01 - Q-table updated episode by episode -> Evaluation (100 episodes, greedy policy) - Reward, steps, and success rate measured -> Visual demonstration (1 episode, render_mode="human") - 0.4 s pause between steps -> End
```

---

## Dependencies

- [`gymnasium`](https://gymnasium.farama.org/) — simulation environment
- `numpy` — matrix operations on the Q-table
- `random` — random action selection during exploration
- `time` — step pacing control in the visual demonstration
