from __future__ import annotations

import torch

from chdqn.noise import correlated_noise, exponential_noise, gaussian_noise, mixed_noise, uniform_noise
from chdqn.reference import reference_sequences
from chdqn.utils import pairwise_variance


def _run_variance_case(trainer, clean_sequence: torch.Tensor, noise: torch.Tensor) -> tuple[float, float]:
    noisy_obs = clean_sequence + noise
    forward = trainer.model.forward_sequence(noisy_obs)
    return float(pairwise_variance(forward.h).detach()), float(pairwise_variance(forward.h_smooth).detach())


def test_smoothing_reduces_variance_for_gaussian_noise(trainer):
    clean, _ = reference_sequences()
    var_h, var_s = _run_variance_case(trainer, clean, gaussian_noise(clean.shape, std=0.01))
    assert var_s < var_h


def test_smoothing_reduces_variance_for_uniform_noise(trainer):
    clean, _ = reference_sequences()
    var_h, var_s = _run_variance_case(trainer, clean, uniform_noise(clean.shape, bound=0.02))
    assert var_s < var_h


def test_smoothing_is_best_under_correlated_noise(trainer):
    clean, _ = reference_sequences()
    gaussian_case = torch.tensor(
        [
            [0.01, -0.01],
            [-0.01, 0.01],
            [0.01, -0.01],
            [-0.01, 0.01],
            [0.01, -0.01],
            [-0.01, 0.01],
            [0.01, -0.01],
            [-0.01, 0.01],
            [0.01, -0.01],
            [-0.01, 0.01],
        ],
        dtype=torch.float32,
    )
    correlated_case = torch.tensor(
        [
            [0.00, 0.00],
            [0.01, -0.01],
            [0.02, -0.02],
            [0.03, -0.03],
            [0.04, -0.04],
            [0.05, -0.05],
            [0.06, -0.06],
            [0.07, -0.07],
            [0.08, -0.08],
            [0.09, -0.09],
        ],
        dtype=torch.float32,
    )
    g_h, g_s = _run_variance_case(trainer, clean, gaussian_case)
    c_h, c_s = _run_variance_case(trainer, clean, correlated_case)
    assert (c_h - c_s) > (g_h - g_s)


def test_exponential_noise_keeps_positive_bias_after_smoothing(trainer):
    clean, _ = reference_sequences()
    clean_forward = trainer.model.forward_sequence(clean)
    exponential_case = torch.tensor(
        [
            [0.02, 0.01],
            [0.03, 0.02],
            [0.04, 0.03],
            [0.05, 0.04],
            [0.06, 0.05],
            [0.07, 0.06],
            [0.08, 0.07],
            [0.09, 0.08],
            [0.10, 0.09],
            [0.11, 0.10],
        ],
        dtype=torch.float32,
    )
    noisy_obs = clean + exponential_case
    forward = trainer.model.forward_sequence(noisy_obs)
    assert forward.h[:, 0].mean().item() > clean_forward.h[:, 0].mean().item()
    assert forward.h_smooth[:, 0].mean().item() > clean_forward.h[:, 0].mean().item()


def test_mixed_noise_is_stabilized(trainer):
    clean, _ = reference_sequences()
    var_h, var_s = _run_variance_case(trainer, clean, mixed_noise(clean.shape))
    assert var_s < var_h


def test_smoothing_target_is_detached_in_loss(trainer, sequence_batch):
    _, loss, _ = trainer.train_step(sequence_batch)
    assert loss.consistency_loss.requires_grad
