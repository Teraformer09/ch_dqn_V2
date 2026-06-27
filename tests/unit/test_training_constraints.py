from __future__ import annotations

import torch

from chdqn.reference import reference_sequences


def test_smoother_is_training_time_component_only(trainer):
    clean, _ = reference_sequences()
    forward = trainer.model.forward_sequence(clean)
    step = trainer.model.forward_step(clean[0].unsqueeze(0), trainer.model.init_hidden())
    assert forward.h_smooth is not None
    assert step.h_smooth is None


def test_consistency_target_is_detached(trainer, sequence_batch):
    trainer.sgd.zero_grad(set_to_none=True)
    _, loss, _ = trainer.train_step(sequence_batch)
    assert loss.total_loss.requires_grad
    assert loss.latent_gap.item() >= 0.0


def test_train_step_keeps_gradients_finite(trainer, sequence_batch):
    trainer.train_step(sequence_batch)
    grads = []
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            grads.append(parameter.grad.detach())
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_evaluate_sequence_variance_returns_expected_keys(trainer):
    clean, _ = reference_sequences()
    metrics = trainer.evaluate_sequence_variance(clean)
    assert set(metrics) == {"latent_forward_var", "latent_smooth_var", "q_var"}


def test_train_step_returns_metric_dict(trainer, sequence_batch):
    _, _, metrics = trainer.train_step(sequence_batch)
    assert {"td_mean", "td_var", "loss", "latent_gap", "reliability"}.issubset(set(metrics))


def test_reference_td_errors_are_monotone_in_magnitude():
    expected = torch.tensor([-0.10, -0.082, -0.061, -0.044, -0.031, -0.022, -0.014, -0.008])
    assert torch.all(expected.abs()[1:] <= expected.abs()[:-1])
