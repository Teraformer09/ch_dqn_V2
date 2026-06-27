"""S6 — Real Learning Verification (4 tests).

Verifies that the Ch-DQN system learns in practice, not just that components
compute correct values.  These tests are intentionally slow (they run full RL
training loops) and are gated by the ``S6`` pytest mark so CI can skip them
with ``pytest -m 'not S6'``.

  S6.1  Learning curve improves over 120 episodes (V2, gaussian noise)
  S6.2  V2 final reward >= V0 final reward (100 episodes, gaussian noise)
  S6.3  Meta-controller alpha_t adapts (max - min > 0.01) in V2 training
  S6.4  No NaN values appear in any logged signal during V2 training
"""
from __future__ import annotations

import csv
import dataclasses
import math
import tempfile
from pathlib import Path

import pytest

from chdqn.config import ChDQNConfig
from chdqn.trainer import CartPoleRLRunner

pytestmark = pytest.mark.S6


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _base_config(**overrides) -> ChDQNConfig:
    cfg = ChDQNConfig(
        train_episodes=100,
        max_steps_per_episode=200,
        batch_size=32,
        min_replay_sequences=16,
        seed=42,
        epsilon_decay=2000,
        learning_rate=0.005,
        gamma=0.99,
    )
    return dataclasses.replace(cfg, **overrides)


def _train_with_log(model_type: str, episodes: int, *, noise_type: str = "gaussian") -> tuple[list[float], dict[str, list]]:
    """Run training, return (rewards_per_episode, signal_columns_dict)."""
    cfg = _base_config(train_episodes=episodes)
    runner = CartPoleRLRunner(cfg, model_type=model_type, noise_type=noise_type)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        log_path = fh.name

    summary = runner.train(log_path=log_path, seed=42)

    # Read the CSV log
    signals: dict[str, list] = {}
    try:
        with open(log_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key, val in row.items():
                    signals.setdefault(key, []).append(val)
    finally:
        Path(log_path).unlink(missing_ok=True)

    rewards_logged = [float(v) for v in signals.get("reward", [])]
    return rewards_logged, signals


# ── S6.1 — Learning curve improvement ─────────────────────────────────────────

class TestS61LearningCurve:
    def test_v2_reward_improves_over_120_episodes(self):
        """Mean reward in last 30 episodes > mean reward in first 30 episodes."""
        rewards, _ = _train_with_log("V2", episodes=120)
        assert len(rewards) >= 60, f"Expected >=60 episodes logged, got {len(rewards)}"
        first_30 = sum(rewards[:30]) / 30
        last_30  = sum(rewards[-30:]) / 30
        assert last_30 > first_30, (
            f"Expected learning: last_30={last_30:.3f} should exceed first_30={first_30:.3f}"
        )

    def test_dqn_reward_improves_over_120_episodes(self):
        """DQN baseline should also learn on CartPole."""
        rewards, _ = _train_with_log("DQN", episodes=120)
        assert len(rewards) >= 60
        first_30 = sum(rewards[:30]) / 30
        last_30  = sum(rewards[-30:]) / 30
        assert last_30 > first_30, (
            f"DQN not learning: last_30={last_30:.3f} vs first_30={first_30:.3f}"
        )


# ── S6.2 — V2 reward >= V0 ─────────────────────────────────────────────────────

class TestS62RelativePerformance:
    def test_v2_not_worse_than_v0(self):
        """V2 final mean reward should be >= V0 final mean reward."""
        rewards_v0, _ = _train_with_log("V0", episodes=100)
        rewards_v2, _ = _train_with_log("V2", episodes=100)

        assert len(rewards_v0) >= 20 and len(rewards_v2) >= 20, "Insufficient episodes"

        mean_v0 = sum(rewards_v0[-20:]) / 20
        mean_v2 = sum(rewards_v2[-20:]) / 20

        # V2 should match or beat V0 on average (allow 20% tolerance)
        tolerance = 0.20 * abs(mean_v0) if abs(mean_v0) > 1e-6 else 0.5
        assert mean_v2 >= mean_v0 - tolerance, (
            f"V2 ({mean_v2:.3f}) significantly underperforms V0 ({mean_v0:.3f})"
        )


# ── S6.3 — Alpha adaptation ────────────────────────────────────────────────────

class TestS63AlphaAdaptation:
    def test_v2_alpha_is_not_constant(self):
        """Meta-controller alpha_t must vary (max - min > 0.01) during V2 training."""
        _, signals = _train_with_log("V2", episodes=100)

        alpha_vals = [float(v) for v in signals.get("alpha_t", []) if v.strip() != ""]
        if not alpha_vals:
            pytest.skip("No alpha_t values logged — check CSVMetricLogger fields")

        alpha_range = max(alpha_vals) - min(alpha_vals)
        assert alpha_range > 0.01, (
            f"Alpha_t appears constant: max={max(alpha_vals):.4f}, "
            f"min={min(alpha_vals):.4f}, range={alpha_range:.4f}"
        )

    def test_v2_gamma_film_is_not_constant(self):
        """FiLM gamma_t should vary as latent gap changes."""
        _, signals = _train_with_log("V2", episodes=100)

        gamma_vals = [float(v) for v in signals.get("gamma_t", []) if v.strip() != ""]
        if not gamma_vals:
            pytest.skip("No gamma_t values logged")

        gamma_range = max(gamma_vals) - min(gamma_vals)
        assert gamma_range > 1e-4, (
            f"gamma_t appears constant: range={gamma_range:.6f}"
        )


# ── S6.4 — No NaN in training ─────────────────────────────────────────────────

class TestS64NoNaN:
    _NUMERIC_COLS = ("reward", "td_mean", "td_var", "loss", "latent_gap",
                     "alpha_t", "gamma_t", "c_t")

    def test_v2_no_nan_in_any_signal(self):
        """No NaN should appear in any numeric logged column during V2 training."""
        _, signals = _train_with_log("V2", episodes=50)

        for col in self._NUMERIC_COLS:
            if col not in signals:
                continue
            vals = signals[col]
            nan_count = sum(
                1 for v in vals
                if v.strip() and math.isnan(float(v))
            )
            assert nan_count == 0, (
                f"Column '{col}' contains {nan_count} NaN value(s)"
            )

    def test_v0_no_nan_in_reward_and_loss(self):
        """V0 reward and loss must be finite throughout training."""
        _, signals = _train_with_log("V0", episodes=50)

        for col in ("reward", "loss", "td_mean", "td_var"):
            vals = signals.get(col, [])
            for i, v in enumerate(vals):
                if v.strip():
                    fv = float(v)
                    assert not math.isnan(fv), f"NaN in col={col} at row {i}"
                    assert not math.isinf(fv), f"Inf in col={col} at row {i}"
