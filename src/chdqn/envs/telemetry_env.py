from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TelemetryConfig:
    noise_type: str = "mixed"
    process_noise_std: float = 0.02
    sensor_noise_std: float = 0.05
    ar_coeff: float = 0.8
    action_scale: float = 0.1
    reward_scale: float = 1.0
    max_steps: int = 100
    termination_bound: float = 5.0
    seed: int = 0


class TelemetryEnv:
    """1-D AR(1) telemetry control environment — no gym dependency.

    Dynamics:   x_{t+1} = α·x_t + u_t + w_t
    Observation: o_t = x_t + v_t
    Reward:      r_t = −|x_t|   (goal: keep state near zero)
    Actions:     0 → push left (−δ),  1 → push right (+δ)

    Noise modes
    -----------
    gaussian  : w_t ~ N(0, σ²)
    correlated: w_t = 0.8·w_{t-1} + N(0, σ²)   (AR drift)
    mixed     : Gaussian base + 20%-probability exponential spike + AR drift

    The mixed mode is the primary use case: structured sensor noise with occasional
    outliers and slow drift — the regime where temporal smoothing wins over IID methods.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._x: float = 0.0
        self._step: int = 0
        self._drift: float = 0.0

    @property
    def action_space(self):
        class _Discrete:
            n = 2
        return _Discrete()

    def reset(self) -> np.ndarray:
        self._x = float(self.rng.uniform(-0.5, 0.5))
        self._step = 0
        self._drift = 0.0
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        u = self.config.action_scale * (1.0 if action == 1 else -1.0)
        self._x = self.config.ar_coeff * self._x + u + self._process_noise()
        self._step += 1
        obs = self._observe()
        reward = -abs(self._x) * self.config.reward_scale
        done = abs(self._x) > self.config.termination_bound or self._step >= self.config.max_steps
        return obs, reward, done, {}

    def close(self) -> None:
        pass

    def _process_noise(self) -> float:
        cfg = self.config
        if cfg.noise_type == "gaussian":
            return float(self.rng.normal(0.0, cfg.process_noise_std))
        if cfg.noise_type == "correlated":
            eta = float(self.rng.normal(0.0, cfg.process_noise_std))
            self._drift = 0.8 * self._drift + eta
            return self._drift
        if cfg.noise_type == "mixed":
            base = float(self.rng.normal(0.0, cfg.process_noise_std))
            spike = float(self.rng.exponential(cfg.process_noise_std)) if self.rng.random() < 0.2 else 0.0
            eta = float(self.rng.normal(0.0, cfg.process_noise_std))
            self._drift = 0.8 * self._drift + eta
            return base + spike + self._drift
        return 0.0

    def _observe(self) -> np.ndarray:
        v = float(self.rng.normal(0.0, self.config.sensor_noise_std))
        return np.array([self._x + v], dtype=np.float32)
