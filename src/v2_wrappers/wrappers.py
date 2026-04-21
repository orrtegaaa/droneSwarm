import gymnasium as gym
from gymnasium import RewardWrapper, ObservationWrapper, Wrapper
from gymnasium import spaces
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if reward == -10:
            reward -= 5

        if reward == 20 and self.steps < 20:
            reward += 10

        return obs, reward, terminated, truncated, info


class CustomObservationWrapper(ObservationWrapper):
    def __init__(self, env, max_steps=80):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([499.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        self.steps = 0
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        remaining = max(0.0, 1.0 - self.steps / self.max_steps)
        return np.array([float(obs), remaining], dtype=np.float32)


class CustomTimeLimitWrapper(Wrapper):
    def __init__(self, env, max_steps=80):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            info["time_limit_reached"] = True
        else:
            info["time_limit_reached"] = False

        return obs, reward, terminated, truncated, info


def make_env_v2():
    env = gym.make("Taxi-v3")
    env = CustomRewardWrapper(env)
    env = CustomTimeLimitWrapper(env, max_steps=80)
    env = CustomObservationWrapper(env, max_steps=80)
    return env
