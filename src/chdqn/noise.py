from __future__ import annotations

import torch


def gaussian_noise(shape: torch.Size, std: float = 0.01) -> torch.Tensor:
    return torch.randn(shape) * std


def uniform_noise(shape: torch.Size, bound: float = 0.02) -> torch.Tensor:
    return (torch.rand(shape) * 2.0 - 1.0) * bound


def exponential_noise(shape: torch.Size, scale: float = 0.05) -> torch.Tensor:
    rate = 1.0 / scale
    return torch.distributions.Exponential(rate).sample(shape) - (1.0 / rate)


def correlated_noise(shape: torch.Size, rho: float = 0.8, std: float = 0.01) -> torch.Tensor:
    if len(shape) != 2:
        raise ValueError("correlated_noise expects [time, dim] shape")
    noise = torch.zeros(shape)
    for t in range(1, shape[0]):
        noise[t] = rho * noise[t - 1] + torch.randn(shape[1]) * std
    return noise


def mixed_noise(shape: torch.Size, std: float = 0.01, spike_prob: float = 0.2, spike_scale: float = 0.03) -> torch.Tensor:
    base = gaussian_noise(shape, std)
    spikes = (torch.rand(shape) < spike_prob).float() * spike_scale
    drift = torch.zeros(shape)
    for t in range(1, shape[0]):
        drift[t] = 0.8 * drift[t - 1] + torch.randn(shape[1]) * std
    return base + spikes + drift
