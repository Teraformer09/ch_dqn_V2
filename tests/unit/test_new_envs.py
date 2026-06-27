from __future__ import annotations

import numpy as np
import pytest
import torch

from chdqn.config import ChDQNConfig
from chdqn.envs.cartpole_correlated import CartPoleCorrelatedNoise, CorrelatedNoiseConfig
from chdqn.envs.cartpole_delay import CartPoleDelay, DelayConfig
from chdqn.envs.telemetry_env import TelemetryConfig, TelemetryEnv
from chdqn.replay import SequenceReplayBuffer
from chdqn.rollout import RolloutCollector
from chdqn.trainer import ChDQNTrainer


# ---------------------------------------------------------------------------
# CartPoleCorrelatedNoise
# ---------------------------------------------------------------------------


def test_correlated_noise_obs_shape():
    env = CartPoleCorrelatedNoise(CorrelatedNoiseConfig(seed=0))
    obs = env.reset()
    assert obs.shape == (2,), f"Expected (2,), got {obs.shape}"
    next_obs, reward, done, _ = env.step(0)
    assert next_obs.shape == (2,)
    assert isinstance(done, bool)
    env.close()


def test_correlated_noise_has_temporal_autocorrelation():
    """Consecutive observations should be correlated (rho=0.9 means strong lag-1 autocorr)."""
    env = CartPoleCorrelatedNoise(CorrelatedNoiseConfig(rho=0.9, sigma=0.02, seed=42))
    env.reset()
    obs_sequence = []
    for _ in range(200):
        obs, _, done, _ = env.step(0)
        obs_sequence.append(float(obs[0]))
        if done:
            env.reset()
    env.close()

    arr = np.array(obs_sequence)
    lag1_corr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    assert lag1_corr > 0.5, f"Expected lag-1 correlation > 0.5 for rho=0.9, got {lag1_corr:.3f}"


def test_correlated_noise_reset_produces_varied_states():
    env = CartPoleCorrelatedNoise(CorrelatedNoiseConfig(seed=7))
    obs1 = env.reset()
    obs2 = env.reset()
    assert not np.allclose(obs1, obs2), "Consecutive resets returned identical observations"
    env.close()


# ---------------------------------------------------------------------------
# CartPoleDelay
# ---------------------------------------------------------------------------


def test_delay_obs_shape():
    env = CartPoleDelay(DelayConfig(delay=3, seed=0))
    obs = env.reset()
    assert obs.shape == (2,)
    next_obs, _, done, _ = env.step(1)
    assert next_obs.shape == (2,)
    assert isinstance(done, bool)
    env.close()


def test_delay_env_returns_stale_observations():
    """With delay=3, the agent at step t should see the observation from reset (step 0)
    for the first 3 steps because the buffer is pre-filled with the reset observation."""
    env = CartPoleDelay(DelayConfig(delay=3, seed=0))
    reset_obs = env.reset()
    for _ in range(3):
        obs, _, done, _ = env.step(0)
        assert np.allclose(obs, reset_obs, atol=1e-6), \
            f"Expected stale (reset) obs for first delay={3} steps, got different obs"
        if done:
            break
    env.close()


def test_delay_reset_produces_varied_states():
    env = CartPoleDelay(DelayConfig(seed=99))
    obs1 = env.reset()
    obs2 = env.reset()
    assert not np.allclose(obs1, obs2)
    env.close()


# ---------------------------------------------------------------------------
# TelemetryEnv
# ---------------------------------------------------------------------------


def test_telemetry_env_reset_and_step():
    env = TelemetryEnv(TelemetryConfig(seed=0))
    obs = env.reset()
    assert obs.shape == (1,), f"Expected (1,), got {obs.shape}"
    next_obs, reward, done, info = env.step(1)
    assert next_obs.shape == (1,)
    assert reward <= 0.0, "Reward should be non-positive (= −|x|)"
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()


def test_telemetry_action_affects_state():
    """Action 0 (push left) and action 1 (push right) must drive states apart."""
    env_left = TelemetryEnv(TelemetryConfig(seed=5))
    env_right = TelemetryEnv(TelemetryConfig(seed=5))
    # same initial state
    env_left.reset()
    env_right.reset()
    env_left._x = 0.0
    env_right._x = 0.0
    env_left._drift = 0.0
    env_right._drift = 0.0

    for _ in range(10):
        env_left.step(0)   # push left
        env_right.step(1)  # push right

    assert env_left._x < env_right._x, \
        f"Left push should decrease state relative to right push: {env_left._x:.4f} vs {env_right._x:.4f}"


def test_telemetry_terminates_on_boundary():
    # _x = 2.0 → x_new = 0.8*2.0 + u + noise ≈ 1.6, comfortably above bound=0.5
    env = TelemetryEnv(TelemetryConfig(termination_bound=0.5, seed=0))
    env.reset()
    env._x = 2.0
    _, _, done, _ = env.step(1)  # push right: keeps x large
    assert done, "Expected termination when post-step |x| > termination_bound"
    env.close()


def test_telemetry_terminates_on_max_steps():
    env = TelemetryEnv(TelemetryConfig(max_steps=5, termination_bound=100.0, seed=0))
    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(1)
    assert done, "Expected termination after max_steps"
    env.close()


# ---------------------------------------------------------------------------
# Integration: CHDQN training on new envs
# ---------------------------------------------------------------------------


def test_chdqn_trains_on_correlated_env():
    config = ChDQNConfig(
        train_episodes=2,
        eval_episodes=1,
        max_steps_per_episode=20,
        min_replay_sequences=1,
        batch_size=1,
        sequence_length=5,
        seed=0,
    )
    trainer = ChDQNTrainer(config)
    collector = RolloutCollector(config)
    replay = SequenceReplayBuffer(capacity=32, sequence_length=config.sequence_length)
    env = CartPoleCorrelatedNoise(CorrelatedNoiseConfig(seed=0))

    for _ in range(2):
        stats = collector.collect_episode(env, trainer.model, replay, max_steps=20)
        assert stats.episode_length > 0
        if len(replay) >= 1:
            step_stats, _, metrics = trainer.train_on_replay(replay, batch_size=1)
            assert torch.isfinite(torch.tensor(step_stats.loss))
            assert metrics["td_var"] >= 0.0

    env.close()


def test_chdqn_trains_on_telemetry_env():
    config = ChDQNConfig(
        obs_dim=1,
        latent_dim=2,
        num_actions=2,
        train_episodes=2,
        eval_episodes=1,
        max_steps_per_episode=20,
        min_replay_sequences=1,
        batch_size=1,
        sequence_length=5,
        seed=0,
    )
    trainer = ChDQNTrainer(config)
    collector = RolloutCollector(config)
    replay = SequenceReplayBuffer(capacity=32, sequence_length=config.sequence_length)
    env = TelemetryEnv(TelemetryConfig(seed=0))

    for _ in range(2):
        stats = collector.collect_episode(env, trainer.model, replay, max_steps=20)
        assert stats.episode_length > 0
        if len(replay) >= 1:
            step_stats, _, metrics = trainer.train_on_replay(replay, batch_size=1)
            assert torch.isfinite(torch.tensor(step_stats.loss))
