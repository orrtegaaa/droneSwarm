# Version 2 - Taxi-v3 with Custom Wrappers

In this second version of the project, we continue using the predefined **Taxi-v3** environment from the **Gymnasium** library.  

The main idea of this part of the assignment is not to create a new environment yet, but to work on an existing one and modify some aspects of its behaviour by using **custom wrappers**.

In this way, we can compare the original environment with an adapted version, keeping the same base while introducing changes in rewards, observations and episode duration.

---

# Objective of this Version

The objective is to analyse how small modifications in the environment can affect the agent’s learning process using **Reinforcement Learning**.

For this reason, the classic Taxi-v3 problem is maintained, where a taxi must pick up a passenger and take them to their destination, but some improvements are added to make the training process more interesting.

---

# Implemented Wrappers

## 1. CustomRewardWrapper

This wrapper modifies the reward system of the original environment.

Changes made:

- Invalid actions receive a higher penalty.
- An extra reward is given when the delivery is completed in a low number of steps.

The purpose is to encourage the agent to act more efficiently and make fewer mistakes.

---

## 2. CustomTimeLimitWrapper

This wrapper limits the maximum number of steps allowed in each episode.

If the agent exceeds that limit without completing the task, the episode ends automatically.

This prevents episodes from becoming too long and helps the training process remain more stable.

---

## 3. CustomObservationWrapper

This wrapper modifies the observation received by the agent.

Instead of returning only the original state of the environment, it now returns:

- The original state.
- The remaining steps normalised between 0 and 1.

In this way, the agent receives additional useful information during learning.

---

# Algorithm Used

The agent was trained using the **Q-learning** algorithm.

A **Q-table** is used to store the value of each state-action pair, updating it episode after episode in order to improve the learned policy.

An **epsilon-greedy** strategy is also applied, combining:

- Random exploration at the beginning.
- Exploitation of the best learned actions later.

---

# General Program Structure

The main file executes three stages:

## Training

The agent plays thousands of episodes in order to learn a better policy.

## Evaluation

The final performance is measured using:

- average reward,
- average number of steps,
- success rate.

## Visual Demonstration

At the end, a simulation is shown using the original Taxi-v3 render mode in order to observe the learned behaviour.

---

# File Structure

- `main_v2.py` → main code for version 2.
- `README_v2.md` → explanation of this version.

---

# Conclusion

This version shows how small changes made through wrappers can influence the agent’s learning process without creating a completely new environment.

In addition, it works as an intermediate step between the initial basic version and a future more advanced custom version.
