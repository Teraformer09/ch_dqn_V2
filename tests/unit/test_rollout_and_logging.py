from __future__ import annotations

from pathlib import Path

import torch

from chdqn.logger import CSVMetricLogger, LogRecord
from chdqn.reference import reference_sequences
from chdqn.replay import SequenceBatch, SequenceReplayBuffer
from chdqn.rollout import RolloutCollector


class DummyEnv:
    def __init__(self) -> None:
        self._state = 0

    def reset(self):
        self._state = 0
        return [0.1, -0.1]

    def step(self, action: int):
        self._state += 1
        done = self._state >= 12
        reward = 1.0 if action in (0, 1) else 0.0
        return [0.1 + 0.01 * self._state, -0.1], reward, done


def test_rollout_collector_runs_without_smoother(trainer):
    replay = SequenceReplayBuffer(capacity=16, sequence_length=trainer.config.sequence_length)
    collector = RolloutCollector(trainer.config)
    env = DummyEnv()
    stats = collector.collect_episode(env, trainer.model, replay, max_steps=20)
    assert stats.episode_length >= trainer.config.sequence_length
    assert len(replay) > 0


def test_rollout_collector_epsilon_bounds(trainer):
    collector = RolloutCollector(trainer.config)
    assert trainer.config.epsilon_end <= collector.current_epsilon() <= trainer.config.epsilon_start


def test_csv_logger_writes_records(tmp_path: Path):
    log_path = tmp_path / "metrics.csv"
    with CSVMetricLogger(log_path) as logger:
        logger.log(LogRecord(episode=1, step=1, reward=1.0, td_mean=0.1, td_var=0.01, loss=0.2))
    text = log_path.read_text(encoding="utf-8")
    assert "reward" in text
    assert "td_var" in text


def test_train_on_replay_uses_subsequence_pipeline(trainer):
    clean, noisy = reference_sequences()
    replay = SequenceReplayBuffer(capacity=4, sequence_length=trainer.config.sequence_length)
    for obs in (clean, noisy):
        replay.add(
            SequenceBatch(
                observations=obs,
                actions=torch.zeros(obs.shape[0], dtype=torch.long),
                rewards=torch.full((obs.shape[0],), 0.1, dtype=torch.float32),
                dones=torch.zeros(obs.shape[0], dtype=torch.bool),
            )
        )
    stats, _, metrics = trainer.train_on_replay(replay, batch_size=2)
    assert stats.td_error_var >= 0.0
    assert metrics["td_var"] >= 0.0
