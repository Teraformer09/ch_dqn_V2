from __future__ import annotations

import random
from typing import Iterable

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(values: Iterable[float] | torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype)
    return torch.tensor(list(values), dtype=dtype)


def huber(error: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_error = error.abs()
    quadratic = torch.minimum(abs_error, torch.tensor(delta, dtype=error.dtype, device=error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def pairwise_variance(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim < 2:
        raise ValueError("Expected sequence with time dimension.")
    return sequence.var(dim=0, unbiased=False).mean()


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, p=2)


def finite_difference_lipschitz(fn, x: torch.Tensor, eps: float = 1e-3) -> float:
    perturb = torch.full_like(x, eps)
    y1 = fn(x)
    y2 = fn(x + perturb)
    denominator = torch.norm(perturb, p=2).item()
    if denominator == 0:
        return 0.0
    return torch.norm(y2 - y1, p=2).item() / denominator


def entropy(probs: torch.Tensor) -> torch.Tensor:
    safe = probs.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=-1)


def clamp_observation(obs: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    if torch.isnan(obs).any() or torch.isinf(obs).any():
        raise ValueError("Observation contains NaN or Inf.")
    return obs.clamp(-limit, limit)


def lambda_schedule(step: int) -> float:
    if step < 50:
        return 0.01
    if step < 150:
        return 0.05
    return 0.1


def build_smoothing_input(h_seq: torch.Tensor, t: int, k: int) -> torch.Tensor:
    """Build a windowed h-only context vector for the temporal smoother.

    Uses only the belief state history h (not z) to force information asymmetry:
    the smoother sees temporal structure while the forward model sees the noisy
    present observation — preventing trivial identity collapse.
    """
    if h_seq.ndim != 2:
        raise ValueError("Expected [T, H] tensor.")
    pieces = []
    for offset in range(-k, k + 1):
        index = min(max(t + offset, 0), h_seq.shape[0] - 1)
        pieces.append(h_seq[index])
    return torch.cat(pieces, dim=-1)
