from __future__ import annotations

import pytest
import torch


def pytest_configure(config):
    for mark in ("S6", "BEHAVIOR", "V1", "V2", "CRITICAL"):
        config.addinivalue_line("markers", f"{mark}: custom test group")

from chdqn.config import ChDQNConfig
from chdqn.reference import get_reference_dry_run, reference_sequences
from chdqn.replay import SequenceBatch
from chdqn.trainer import ChDQNTrainer


@pytest.fixture()
def config() -> ChDQNConfig:
    return ChDQNConfig()


@pytest.fixture()
def trainer(config: ChDQNConfig) -> ChDQNTrainer:
    return ChDQNTrainer(config, use_reference_init=True)


@pytest.fixture()
def reference_run():
    return get_reference_dry_run()


@pytest.fixture()
def sequence_batch() -> SequenceBatch:
    clean, noisy = reference_sequences()
    observations = torch.stack([clean, noisy], dim=0)
    actions = torch.zeros((2, clean.shape[0]), dtype=torch.long)
    rewards = torch.full((2, clean.shape[0]), 0.1, dtype=torch.float32)
    dones = torch.zeros((2, clean.shape[0]), dtype=torch.bool)
    return SequenceBatch(observations=observations, actions=actions, rewards=rewards, dones=dones)
