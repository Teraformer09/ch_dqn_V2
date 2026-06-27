"""S1 — Mathematical Consistency (80 tests).

Maps 1-to-1 to theoretical claims in the Ch-DQN paper:
  S1.1  Latent consistency  (20)
  S1.2  Smoother correctness (20)
  S1.3  Variance reduction   (20)
  S1.4  Bellman operator     (20)
"""
from __future__ import annotations

import pytest
import torch

from chdqn.config import ChDQNConfig
from chdqn.reliability import apply_bias_floor, compute_reliability, double_dqn_target
from chdqn.models import ChDQNModel
from chdqn.noise import correlated_noise, exponential_noise, gaussian_noise, uniform_noise
from chdqn.reference import reference_sequences
from chdqn.utils import pairwise_variance, set_seed

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 7) -> ChDQNModel:
    cfg = ChDQNConfig(seed=seed)
    set_seed(seed)
    return ChDQNModel(cfg, use_reference_init=True)


def _forward_noisy(model: ChDQNModel, sigma: float = 0.01):
    clean, _ = reference_sequences()
    noise = gaussian_noise(clean.shape, std=sigma)
    return model.forward_sequence(clean + noise)


# ---------------------------------------------------------------------------
# S1.1  Latent Consistency (20 tests = 4 base × 5 seeds)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 3, 7, 13, 42])
def test_filter_deterministic_same_input(seed):
    """h_t = f(h_{t-1}, z_t) is a pure function — same input always yields same output."""
    model = _make_model(seed)
    prev_h = torch.tensor([[0.1, -0.1]])
    z = torch.tensor([[0.08, 0.05]])
    h1 = model.filter(prev_h, z)
    h2 = model.filter(prev_h, z)
    assert torch.allclose(h1, h2, atol=1e-7)


@pytest.mark.parametrize("scale", [0.05, 0.1, 0.5, 1.0, 2.0])
def test_filter_contractive_under_perturbation(scale):
    """Local contractivity: ||f(h1) - f(h2)|| < ||h1 - h2||."""
    model = _make_model()
    h1 = torch.tensor([[0.1, -0.1]]) * scale
    h2 = torch.tensor([[0.11, -0.11]]) * scale
    z = torch.tensor([[0.085, 0.055]])
    out1 = model.filter(h1, z)
    out2 = model.filter(h2, z)
    assert torch.norm(out1 - out2).item() < torch.norm(h1 - h2).item()


@pytest.mark.parametrize("seed", [1, 3, 7, 13, 42])
def test_hidden_state_finite_after_sequence(seed):
    """No NaN/Inf in hidden states over a full sequence."""
    model = _make_model(seed)
    out = _forward_noisy(model)
    assert torch.isfinite(out.h).all()
    assert torch.isfinite(out.q).all()


@pytest.mark.parametrize("noise_std", [0.001, 0.01, 0.05, 0.1, 0.3])
def test_hidden_norm_bounded(noise_std):
    """tanh activation keeps ||h_t||_inf <= 1."""
    model = _make_model()
    clean, _ = reference_sequences()
    obs = clean + gaussian_noise(clean.shape, std=noise_std)
    out = model.forward_sequence(obs)
    assert out.h.abs().max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# S1.2  Smoother Correctness (20 tests = 4 base × 5 param settings)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 3, 7, 13, 42])
def test_smoother_nontrivial(seed):
    """h_tilde ≠ h_t — smoother is not the identity."""
    model = _make_model(seed)
    out = _forward_noisy(model, sigma=0.05)
    # h_smooth is the smoother applied to every timestep
    assert not torch.allclose(out.h, out.h_smooth, atol=1e-4)


@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1, 0.2, 0.4])
def test_smoother_output_finite(sigma):
    """Smoother never produces NaN/Inf."""
    model = _make_model()
    out = _forward_noisy(model, sigma=sigma)
    assert torch.isfinite(out.h_smooth).all()


@pytest.mark.parametrize("seed", [1, 3, 7, 13, 42])
def test_smoother_bounded(seed):
    """Smoother output magnitude is reasonable (not exploding)."""
    model = _make_model(seed)
    out = _forward_noisy(model, sigma=0.1)
    assert out.h_smooth.abs().max().item() < 10.0


@pytest.mark.parametrize("noise_std", [0.001, 0.01, 0.05, 0.1, 0.2])
def test_smoother_gap_positive(noise_std):
    """With nonzero noise, ||h - h_smooth|| > 0 (smoother does something)."""
    model = _make_model()
    out = _forward_noisy(model, sigma=noise_std)
    gap = (out.h - out.h_smooth).norm(dim=-1).mean().item()
    assert gap > 0.0


# ---------------------------------------------------------------------------
# S1.3  Variance Reduction (20 tests = 4 conditions × 5 sigma values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.01, 0.03, 0.05, 0.1, 0.15])
def test_variance_reduction_gaussian(sigma):
    """Var(h_smooth) < Var(h) under Gaussian noise."""
    model = _make_model()
    clean, _ = reference_sequences()
    obs = clean + gaussian_noise(clean.shape, std=sigma)
    out = model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item()


@pytest.mark.parametrize("bound", [0.01, 0.03, 0.05, 0.1, 0.15])
def test_variance_reduction_uniform(bound):
    """Var(h_smooth) < Var(h) under uniform noise."""
    model = _make_model()
    clean, _ = reference_sequences()
    obs = clean + uniform_noise(clean.shape, bound=bound)
    out = model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item()


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.85, 0.95])
def test_variance_reduction_correlated(rho):
    """Smoother reduces variance under AR(1) correlated noise."""
    model = _make_model()
    clean, _ = reference_sequences()
    obs = clean + correlated_noise(clean.shape, rho=rho, std=0.01)
    out = model.forward_sequence(obs)
    assert pairwise_variance(out.h_smooth).item() < pairwise_variance(out.h).item()


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.04, 0.08])
def test_smoother_reduces_variance_monotone_with_noise(sigma):
    """Variance gap (h_var - h_smooth_var) is positive for all tested sigma."""
    model = _make_model()
    clean, _ = reference_sequences()
    obs = clean + gaussian_noise(clean.shape, std=sigma)
    out = model.forward_sequence(obs)
    gap = pairwise_variance(out.h).item() - pairwise_variance(out.h_smooth).item()
    assert gap >= 0.0


# ---------------------------------------------------------------------------
# S1.4  Bellman Operator (20 tests = 4 checks × 5 conditions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("floor", [0.05, 0.10, 0.15, 0.20, 0.25])
def test_bias_floor_bounds_c_prime(floor):
    """c_prime = ε + (1-ε)*c_t ∈ [ε, 1.0] for any c_t ∈ [0, 1]."""
    for c_val in [0.0, 0.3, 0.7, 0.95, 1.0]:
        c_t = torch.tensor([c_val])
        c_prime = apply_bias_floor(c_t, floor)
        assert c_prime.item() >= floor - 1e-6
        assert c_prime.item() <= 1.0 + 1e-6


@pytest.mark.parametrize("floor", [0.05, 0.10, 0.15, 0.20, 0.25])
def test_bellman_contraction_holds(floor):
    """γ * max(c_prime) < 1 for γ=0.99 and any floor ≥ 0.05."""
    gamma = 0.99
    for c_val in [0.0, 0.5, 0.9, 1.0]:
        c_t = torch.tensor([c_val])
        c_prime = apply_bias_floor(c_t, floor)
        assert (gamma * c_prime).max().item() < 1.0


@pytest.mark.parametrize("gap", [0.0, 0.1, 0.3, 0.6, 1.0])
def test_reliability_in_unit_interval(gap):
    """c_t = exp(-||h - h_tilde||^2) ∈ (0, 1]."""
    h = torch.zeros(1, 2)
    h_tilde = torch.full((1, 2), gap / 2.0)
    c_t = compute_reliability(h, h_tilde)
    assert c_t.item() > 0.0
    assert c_t.item() <= 1.0 + 1e-6


@pytest.mark.parametrize("gamma", [0.90, 0.95, 0.99, 0.995, 0.999])
def test_double_dqn_target_with_c_prime_is_finite(gamma):
    """Target is finite when c_prime is applied to the Bellman backup."""
    rewards = torch.tensor([0.1])
    dones = torch.tensor([False])
    online_q = torch.tensor([[0.2, 0.1]])
    target_q = torch.tensor([[0.2, 0.1]])
    c_prime = torch.tensor([0.88])
    tgt = double_dqn_target(rewards, dones, online_q, target_q, gamma=gamma, clip_value=1.0, c_prime=c_prime)
    assert torch.isfinite(tgt).all()
    assert tgt.abs().max().item() <= 1.0 + 1e-5
