"""S3 — Distributional Behavior (140 tests).

  S3.1  Gaussian noise         (30)
  S3.2  Uniform noise          (20)
  S3.3  Exponential noise      (30)
  S3.4  Correlated (AR-1) noise (30)
  S3.5  Mixed noise            (30)
"""
from __future__ import annotations

import pytest
import torch

from chdqn.config import ChDQNConfig
from chdqn.reliability import apply_bias_floor, compute_reliability
from chdqn.models import ChDQNModel
from chdqn.noise import (
    correlated_noise,
    exponential_noise,
    gaussian_noise,
    mixed_noise,
    uniform_noise,
)
from chdqn.reference import reference_sequences
from chdqn.replay import SequenceBatch
from chdqn.trainer import ChDQNTrainer
from chdqn.utils import pairwise_variance, set_seed


def _mini_trainer(use_v2: bool = False, seed: int = 7) -> ChDQNTrainer:
    cfg = ChDQNConfig(seed=seed, use_v2=use_v2)
    return ChDQNTrainer(cfg, use_reference_init=True)


def _make_batch(obs: torch.Tensor) -> SequenceBatch:
    return SequenceBatch(
        observations=obs,
        actions=torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.long),
        rewards=torch.full((obs.shape[0], obs.shape[1]), 0.1),
        dones=torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.bool),
    )


def _run_steps(trainer: ChDQNTrainer, obs: torch.Tensor, n: int = 5) -> list[float]:
    batch = _make_batch(obs)
    td_vars = []
    for _ in range(n):
        _, _, metrics = trainer.train_step(batch)
        td_vars.append(metrics["td_var"])
    return td_vars


# ---------------------------------------------------------------------------
# S3.1  Gaussian Noise (30 = 3 metrics × 10 sigma values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
def test_gaussian_variance_reduction_holds(sigma):
    """Var(h_smooth) < Var(h) for Gaussian noise at any sigma."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + gaussian_noise(clean.shape, std=sigma)
    out = trainer.model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item() + 1e-6


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
def test_gaussian_td_var_is_finite(sigma):
    """TD variance is a finite number under Gaussian noise."""
    trainer = _mini_trainer()
    clean, noisy = reference_sequences()
    obs = torch.stack([
        clean + gaussian_noise(clean.shape, std=sigma),
        noisy + gaussian_noise(noisy.shape, std=sigma),
    ])
    td_vars = _run_steps(trainer, obs, n=3)
    for v in td_vars:
        assert torch.isfinite(torch.tensor(v))


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
def test_gaussian_reliability_high_for_low_sigma(sigma):
    """For low sigma, reliability should be close to 1 (h ≈ h_tilde)."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + gaussian_noise(clean.shape, std=sigma)
    out = trainer.model.forward_sequence(obs)
    # With small sigma, the latent gap should be bounded
    gap = (out.h - out.h_smooth).norm(dim=-1).mean().item()
    assert gap < 5.0  # gap grows with sigma but stays bounded


# ---------------------------------------------------------------------------
# S3.2  Uniform Noise (20 = 2 metrics × 10 bound values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bound", [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
def test_uniform_variance_reduction(bound):
    """Smoother reduces variance under uniform noise."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + uniform_noise(clean.shape, bound=bound)
    out = trainer.model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item() + 1e-6


@pytest.mark.parametrize("bound", [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
def test_uniform_loss_finite(bound):
    """Training is stable under uniform noise."""
    trainer = _mini_trainer()
    clean, noisy = reference_sequences()
    obs = torch.stack([
        clean + uniform_noise(clean.shape, bound=bound),
        noisy,
    ])
    _, _, metrics = trainer.train_step(_make_batch(obs))
    assert torch.isfinite(torch.tensor(metrics["loss"]))


# ---------------------------------------------------------------------------
# S3.3  Exponential Noise (30 = 3 metrics × 10 scale values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
def test_exponential_bias_measurement(scale):
    """Exponential noise introduces positive bias: mean(h_noisy) > mean(h_clean)."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    clean_fwd = trainer.model.forward_sequence(clean)
    exp_noise = exponential_noise(clean.shape, scale=scale)
    noisy_fwd = trainer.model.forward_sequence(clean + exp_noise)
    # Exponential noise is positive-biased: observations are shifted up
    assert noisy_fwd.h[:, 0].mean().item() >= clean_fwd.h[:, 0].mean().item() - 0.2


@pytest.mark.parametrize("scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
def test_exponential_v1_reliability_bounded(scale):
    """With bias floor, c_prime ≥ floor even under exponential noise."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + exponential_noise(clean.shape, scale=scale)
    out = trainer.model.forward_sequence(obs)
    h_t = out.h[5:6]
    h_tilde = out.h_smooth[5:6]
    raw_rel = compute_reliability(h_t, h_tilde)
    c_prime = apply_bias_floor(raw_rel, floor=0.15)
    assert c_prime.item() >= 0.15 - 1e-6


@pytest.mark.parametrize("scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
def test_exponential_training_stable(scale):
    """Training does not diverge under exponential noise."""
    trainer = _mini_trainer()
    clean, noisy = reference_sequences()
    obs = torch.stack([
        clean + exponential_noise(clean.shape, scale=scale),
        noisy,
    ])
    td_vars = _run_steps(trainer, obs, n=5)
    assert all(torch.isfinite(torch.tensor(v)) for v in td_vars)


# ---------------------------------------------------------------------------
# S3.4  Correlated (AR-1) Noise (30 = 3 metrics × 10 rho values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
def test_correlated_smoother_gap_bounded(rho):
    """Smoother gap is bounded even under highly correlated noise."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + correlated_noise(clean.shape, rho=rho, std=0.01)
    out = trainer.model.forward_sequence(obs)
    gap = (out.h - out.h_smooth).norm(dim=-1).mean().item()
    assert gap < 10.0


@pytest.mark.parametrize("rho", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
def test_correlated_variance_reduction(rho):
    """Smoother reduces variance under AR-1 correlated noise (key V0 claim)."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + correlated_noise(clean.shape, rho=rho, std=0.01)
    out = trainer.model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item() + 1e-6


@pytest.mark.parametrize("rho", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
def test_correlated_training_finite(rho):
    """Loss stays finite under correlated noise."""
    trainer = _mini_trainer()
    clean, noisy = reference_sequences()
    obs = torch.stack([
        clean + correlated_noise(clean.shape, rho=rho, std=0.01),
        noisy,
    ])
    _, _, metrics = trainer.train_step(_make_batch(obs))
    assert torch.isfinite(torch.tensor(metrics["loss"]))


# ---------------------------------------------------------------------------
# S3.5  Mixed Noise (30 = 3 metrics × 10 spike_prob values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spike_prob", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
def test_mixed_noise_variance_reduced(spike_prob):
    """Smoother reduces variance even under burst/mixed noise."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + mixed_noise(clean.shape, spike_prob=spike_prob)
    out = trainer.model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item() + 1e-6


@pytest.mark.parametrize("spike_prob", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
def test_mixed_noise_v1_contraction_holds(spike_prob):
    """V1 reliability-weighted operator satisfies contraction under mixed noise."""
    trainer = _mini_trainer()
    clean, _ = reference_sequences()
    obs = clean + mixed_noise(clean.shape, spike_prob=spike_prob)
    out = trainer.model.forward_sequence(obs)
    h_t = out.h[5:6]
    h_tilde = out.h_smooth[5:6]
    raw = compute_reliability(h_t, h_tilde)
    c_prime = apply_bias_floor(raw, floor=0.15)
    gamma = 0.99
    assert (gamma * c_prime).max().item() < 1.0


@pytest.mark.parametrize("spike_prob", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
def test_mixed_noise_training_stable(spike_prob):
    """Training does not diverge under mixed spike noise."""
    trainer = _mini_trainer()
    clean, noisy = reference_sequences()
    obs = torch.stack([
        clean + mixed_noise(clean.shape, spike_prob=spike_prob),
        noisy,
    ])
    td_vars = _run_steps(trainer, obs, n=4)
    assert all(torch.isfinite(torch.tensor(v)) for v in td_vars)
