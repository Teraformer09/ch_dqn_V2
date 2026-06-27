from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(slots=True)
class BenchmarkMetrics:
    mean_reward: float
    reward_std: float
    mean_td: float
    td_variance: float
    mean_loss: float
    convergence_episode: int
    stability_std_last_50: float


def mean_reward(rewards: Sequence[float]) -> float:
    if not rewards:
        return 0.0
    return float(torch.tensor(list(rewards), dtype=torch.float32).mean())


def reward_std(rewards: Sequence[float]) -> float:
    if not rewards:
        return 0.0
    return float(torch.tensor(list(rewards), dtype=torch.float32).std(unbiased=False))


def td_mean(td_errors: Sequence[float]) -> float:
    if not td_errors:
        return 0.0
    return float(torch.tensor(list(td_errors), dtype=torch.float32).mean())


def td_variance(td_errors: Sequence[float]) -> float:
    if not td_errors:
        return 0.0
    return float(torch.tensor(list(td_errors), dtype=torch.float32).var(unbiased=False))


def convergence_episode(rewards: Sequence[float], threshold: float = 150.0) -> int:
    for idx, reward in enumerate(rewards, start=1):
        if reward >= threshold:
            return idx
    return -1


def stability_std(rewards: Sequence[float], window: int = 50) -> float:
    if not rewards:
        return 0.0
    values = rewards[-window:]
    return float(torch.tensor(list(values), dtype=torch.float32).std(unbiased=False))


def build_metrics(rewards: Sequence[float], td_errors: Sequence[float], losses: Sequence[float]) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        mean_reward=mean_reward(rewards),
        reward_std=reward_std(rewards),
        mean_td=td_mean(td_errors),
        td_variance=td_variance(td_errors),
        mean_loss=float(torch.tensor(list(losses), dtype=torch.float32).mean()) if losses else 0.0,
        convergence_episode=convergence_episode(rewards),
        stability_std_last_50=stability_std(rewards, window=50),
    )
