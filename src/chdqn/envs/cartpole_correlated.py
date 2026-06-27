from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class CorrelatedNoiseConfig:
    rho: float = 0.9
    sigma: float = 0.02
    reward_scale: float = 1.0
    seed: int = 0


class CartPoleCorrelatedNoise:
    """CartPole-v1 with AR(1) correlated observation noise on the two exposed dimensions.

    Partial observability: only position (obs[0]) and pole angle (obs[2]) are exposed.

    Noise model:
        ε_t = ρ · ε_{t-1} + η_t,   η_t ~ N(0, σ²)

    This is the environment Ch-DQN is designed for: temporally correlated noise that
    IID-replay DQN cannot exploit, but temporal smoothing can.
    """

    def __init__(self, config: CorrelatedNoiseConfig | None = None) -> None:
        self.config = config or CorrelatedNoiseConfig()
        self.env = gym.make("CartPole-v1")
        self.rng = np.random.default_rng(self.config.seed)
        self._noise = np.zeros(2, dtype=np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        return np.array([obs[0], obs[2]], dtype=np.float32)

    def _advance_noise(self) -> np.ndarray:
        eta = self.rng.normal(0.0, self.config.sigma, size=2).astype(np.float32)
        self._noise = self.config.rho * self._noise + eta
        return self._noise.copy()

    def reset(self) -> np.ndarray:
        gym_seed = int(self.rng.integers(0, 2**31))
        obs, _ = self.env.reset(seed=gym_seed)
        self._noise = np.zeros(2, dtype=np.float32)
        return self._mask(obs) + self._advance_noise()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return self._mask(obs) + self._advance_noise(), float(reward) * self.config.reward_scale, done, info

    def close(self) -> None:
        self.env.close()
