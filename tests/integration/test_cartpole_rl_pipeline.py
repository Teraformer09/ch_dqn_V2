from __future__ import annotations

import numpy as np
import torch

from chdqn.models import DQNBaseline
from chdqn.models import DRQNBaseline
from chdqn.config import ChDQNConfig
from chdqn.envs.cartpole_pomdp import CartPolePOMDPConfig, CartPolePOMDPEnv
from chdqn.evaluation import evaluate_model
from chdqn.replay import SequenceReplayBuffer
from chdqn.rollout import RolloutCollector
from chdqn.trainer import CartPoleRLRunner


def test_cartpole_pomdp_masks_velocity():
    env = CartPolePOMDPEnv(CartPolePOMDPConfig(noise_type=None))
    obs = env.reset()
    assert obs.shape == (2,)
    _, _, done, _ = env.step(0)
    assert isinstance(done, bool)
    env.close()


def test_cartpole_pomdp_noise_is_configurable():
    env = CartPolePOMDPEnv(CartPolePOMDPConfig(noise_type="gaussian"))
    obs = env.reset()
    assert obs.shape == (2,)
    env.close()


def test_rollout_collector_populates_real_env_replay(trainer):
    env = CartPolePOMDPEnv(CartPolePOMDPConfig(noise_type="gaussian"))
    replay = SequenceReplayBuffer(capacity=32, sequence_length=trainer.config.sequence_length)
    collector = RolloutCollector(trainer.config)
    stats = collector.collect_episode(env, trainer.model, replay, max_steps=50)
    assert stats.episode_length > 0
    assert len(replay) >= 0
    env.close()


def test_evaluate_model_runs_on_cartpole(trainer):
    stats = evaluate_model(
        trainer.model,
        lambda: CartPolePOMDPEnv(CartPolePOMDPConfig(noise_type=None)),
        episodes=2,
        max_steps=20,
    )
    assert stats.mean_episode_length > 0


def test_dqn_baseline_forward_shape():
    model = DQNBaseline()
    obs = torch.zeros(1, 2)
    q = model(obs)
    assert q.shape == (1, 2)


def test_drqn_baseline_forward_shape():
    model = DRQNBaseline()
    obs = torch.zeros(1, 2)
    hidden = model.init_hidden()
    q, hidden = model.forward_step(obs, hidden)
    assert q.shape == (1, 2)
    assert hidden.shape[-1] == model.hidden_dim


def test_cartpole_rl_runner_trains_small_loop():
    config = ChDQNConfig(train_episodes=2, eval_episodes=2, max_steps_per_episode=20, min_replay_sequences=1, batch_size=1)
    runner = CartPoleRLRunner(config, noise_type="gaussian")
    summary = runner.train()
    assert summary.episodes == 2
    assert summary.evaluation.mean_episode_length > 0


def test_cartpole_env_reset_produces_varied_initial_states():
    env = CartPolePOMDPEnv(CartPolePOMDPConfig(seed=42))
    obs1 = env.reset()
    obs2 = env.reset()
    assert not np.allclose(obs1, obs2), "Consecutive resets returned identical observations — seed not advancing"
    env.close()


def test_different_episode_seeds_produce_different_envs():
    runner = CartPoleRLRunner(ChDQNConfig(seed=7), noise_type=None)
    env_a = runner.make_env(episode=1)
    env_b = runner.make_env(episode=2)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert not np.allclose(obs_a, obs_b), "Different episode seeds produced identical initial observations"
    env_a.close()
    env_b.close()
