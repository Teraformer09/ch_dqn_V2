from __future__ import annotations

import torch

from chdqn.reliability import double_dqn_target
from chdqn.reference import get_reference_config


def test_q_head_matches_reference_for_h1(trainer):
    h = torch.tensor([[0.05, -0.02]])
    q = trainer.model.q_head(h).squeeze(0)
    assert torch.allclose(q, torch.tensor([0.046, 0.011]), atol=1e-3)


def test_q_head_matches_reference_for_h3(trainer):
    h = torch.tensor([[0.10, -0.08]])
    q = trainer.model.q_head(h).squeeze(0)
    assert torch.allclose(q, torch.tensor([0.104, -0.006]), atol=1e-3)


def test_double_dqn_target_matches_dry_run():
    rewards = torch.tensor([0.1])
    dones = torch.tensor([False])
    online_next = torch.tensor([[0.13, -0.02]])
    target_next = torch.tensor([[0.13, -0.02]])
    target = double_dqn_target(rewards, dones, online_next, target_next, gamma=0.99, clip_value=1.0)
    assert torch.allclose(target, torch.tensor([0.2287]), atol=1e-3)


def test_reference_td_curve_matches_expected():
    config = get_reference_config()
    td = torch.tensor(config.epoch_td)
    assert torch.allclose(td[[0, 3, 7]], torch.tensor([-0.10, -0.044, -0.008]), atol=1e-3)


def test_train_step_runs_and_returns_finite_stats(trainer, sequence_batch):
    stats, _, metrics = trainer.train_step(sequence_batch)
    assert stats.loss > 0.0
    assert torch.isfinite(torch.tensor(stats.loss))
    assert torch.isfinite(torch.tensor(stats.td_error_mean))
    assert "td_var" in metrics


def test_target_network_soft_updates(trainer, sequence_batch):
    before = [param.detach().clone() for param in trainer.target_model.parameters()]
    trainer.train_step(sequence_batch)
    after = list(trainer.target_model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_policy_probabilities_sum_to_one(trainer):
    h = torch.tensor([[0.05, -0.02], [0.12, -0.11]])
    probs = trainer.model.policy(h)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(h.shape[0]), atol=1e-6)
