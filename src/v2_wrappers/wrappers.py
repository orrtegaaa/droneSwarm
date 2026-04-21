import gymnasium as gym
from gymnasium import RewardWrapper, ObservationWrapper, Wrapper, spaces
import numpy as np
import random
import time



# WRAPPER 1 - RECOMPENSAS PERSONALIZADAS

class CustomRewardWrapper(RewardWrapper):
    """
    Modifica las recompensas del entorno original.

    Cambios:
    - Penaliza más acciones incorrectas
    - Premio extra si entrega rápido
    """

    def __init__(self, env):
        super().__init__(env)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1

        # Acción inválida (Taxi-v3 da -10)
        if reward == -10:
            reward -= 5

        # Si entrega rápido
        if reward == 20 and self.steps <= 20:
            reward += 10

        return obs, reward, terminated, truncated, info



# WRAPPER 2 - LÍMITE DE PASOS

class CustomTimeLimitWrapper(Wrapper):
    """
    Finaliza el episodio si supera MAX_STEPS.
    """

    def __init__(self, env, max_steps):
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
            info["time_limit"] = True
        else:
            info["time_limit"] = False

        return obs, reward, terminated, truncated, info




# WRAPPER 3 - OBSERVACIÓN MODIFICADA

class CustomObservationWrapper(ObservationWrapper):
    """
    Devuelve:
    [estado_original, pasos_restantes_normalizados]
    """

    def __init__(self, env, max_steps):
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

        return np.array(
            [float(obs), remaining],
            dtype=np.float32
        )
