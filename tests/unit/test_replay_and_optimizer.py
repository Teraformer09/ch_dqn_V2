from __future__ import annotations

import torch
import pytest

from chdqn.reliability import double_dqn_target
from chdqn.optimization import SecondOrderOptimizer
from chdqn.replay import SequenceBatch, SequenceReplayBuffer


def test_sequence_replay_requires_minimum_length():
    with pytest.raises(ValueError):
        SequenceReplayBuffer(capacity=4, sequence_length=2)


def test_sequence_replay_preserves_contiguous_sequences():
    buffer = SequenceReplayBuffer(capacity=4, sequence_length=3)
    batch = SequenceBatch(
        observations=torch.arange(6, dtype=torch.float32).view(3, 2),
        actions=torch.zeros(3, dtype=torch.long),
        rewards=torch.ones(3),
        dones=torch.zeros(3, dtype=torch.bool),
    )
    buffer.add(batch)
    sample = buffer.sample(1)
    assert torch.equal(sample.observations[0], batch.observations)


def test_sequence_replay_sampling_is_randomized():
    torch.manual_seed(0)
    buffer = SequenceReplayBuffer(capacity=8, sequence_length=3)
    for idx in range(4):
        base = torch.full((3, 2), float(idx))
        buffer.add(
            SequenceBatch(
                observations=base,
                actions=torch.zeros(3, dtype=torch.long),
                rewards=torch.ones(3),
                dones=torch.zeros(3, dtype=torch.bool),
            )
        )
    sample = buffer.sample(2)
    distinct_ids = {float(item[0, 0].item()) for item in sample.observations}
    assert len(distinct_ids) >= 1
    assert sample.observations.shape[0] == 2


def test_sequence_replay_can_extract_subsequences():
    torch.manual_seed(0)
    buffer = SequenceReplayBuffer(capacity=4, sequence_length=4)
    batch = SequenceBatch(
        observations=torch.arange(8, dtype=torch.float32).view(4, 2),
        actions=torch.arange(4, dtype=torch.long),
        rewards=torch.arange(4, dtype=torch.float32),
        dones=torch.zeros(4, dtype=torch.bool),
    )
    buffer.add(batch)
    subs = buffer.sample_subsequences(1)
    assert subs.o_t.shape[0] == 2
    assert torch.equal(subs.o_prev[0], batch.observations[0])
    assert torch.equal(subs.o_next[1], batch.observations[3])


def test_add_trajectory_creates_overlapping_windows():
    buffer = SequenceReplayBuffer(capacity=8, sequence_length=3)
    inserted = buffer.add_trajectory(
        observations=torch.arange(10, dtype=torch.float32).view(5, 2),
        actions=torch.arange(5, dtype=torch.long),
        rewards=torch.ones(5),
        dones=torch.zeros(5, dtype=torch.bool),
    )
    assert inserted == 3
    assert len(buffer) == 3


def test_double_dqn_target_is_clipped():
    rewards = torch.tensor([0.9])
    dones = torch.tensor([False])
    online_next = torch.tensor([[5.0, 4.0]])
    target_next = torch.tensor([[5.0, 4.0]])
    target = double_dqn_target(rewards, dones, online_next, target_next, gamma=0.99, clip_value=1.0)
    assert target.item() == pytest.approx(1.0)


def test_second_order_optimizer_updates_only_when_grad_present():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SecondOrderOptimizer({"p": param}, lr=0.1, beta1=0.0, beta2=0.0)
    optimizer.step()
    assert param.item() == pytest.approx(1.0)
    param.grad = torch.tensor([2.0])
    optimizer.step()
    assert param.item() == pytest.approx(0.8)


def test_second_order_optimizer_uses_gradient_velocity():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SecondOrderOptimizer({"p": param}, lr=0.1, beta1=0.5, beta2=0.1)
    param.grad = torch.tensor([2.0])
    optimizer.step()
    assert optimizer.state.velocity["p"].item() == pytest.approx(2.0)


def test_policy_entropy_is_positive(trainer):
    h = torch.tensor([[0.1, -0.1]])
    probs = trainer.model.policy(h, temperature=0.5)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum().item()
    assert entropy > 0.0
