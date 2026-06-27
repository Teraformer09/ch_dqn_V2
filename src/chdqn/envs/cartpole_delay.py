from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class DelayConfig:
    delay: int = 3
    reward_scale: float = 1.0
    seed: int = 0


class CartPoleDelay:
    """CartPole-v1 with delayed observations: the agent sees o_t = s_{t-d}.

    Partial observability: only position (obs[0]) and pole angle (obs[2]) are exposed.

    Tests whether temporal memory models (DRQN, Ch-DQN) outperform memoryless DQN
    by recovering information about the current state from a stale history.
    DRQN is expected to lead here; Ch-DQN's smoother provides moderate benefit.
    """

    def __init__(self, config: DelayConfig | None = None) -> None:
        self.config = config or DelayConfig()
        self.env = gym.make("CartPole-v1")
        self.rng = np.random.default_rng(self.config.seed)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.config.delay + 1)

    @property
    def action_space(self):
        return self.env.action_space

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        return np.array([obs[0], obs[2]], dtype=np.float32)

    def reset(self) -> np.ndarray:
        gym_seed = int(self.rng.integers(0, 2**31))
        obs, _ = self.env.reset(seed=gym_seed)
        masked = self._mask(obs)
        self._buffer.clear()
        for _ in range(self.config.delay + 1):
            self._buffer.append(masked.copy())
        return self._buffer[0].copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self._buffer.append(self._mask(obs))
        return self._buffer[0].copy(), float(reward) * self.config.reward_scale, done, info

    def close(self) -> None:
        self.env.close()
