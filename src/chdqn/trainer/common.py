from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from chdqn.evaluation import evaluate_run
from chdqn.utils import BenchmarkLogger
from chdqn.utils import load_yaml_config, resolve_results_dir, set_global_seed

from chdqn.models import DQNBaseline
from chdqn.models import DRQNBaseline
from chdqn.models import R2D2Baseline
from chdqn.config import ChDQNConfig
from chdqn.envs import CartPoleCorrelatedNoise, CorrelatedNoiseConfig
from chdqn.envs import CartPoleDelay, DelayConfig
from chdqn.envs import CartPolePOMDPConfig, CartPolePOMDPEnv
from chdqn.envs import TelemetryConfig, TelemetryEnv
from chdqn.models import ChDQNModel
from chdqn.replay import SequenceReplayBuffer
from chdqn.rollout import RolloutCollector
from chdqn.trainer import ChDQNTrainer


# ── Minimum replay fill before training starts ───────────────────────────────
_MIN_REPLAY = 100


def load_runner_config(path: str | Path, *, seed: int) -> dict:
    config = load_yaml_config(path)
    config["seed"] = seed
    return config


def make_env(config: dict, seed: int):
    env_type = config["env"]
    if env_type == "cartpole_pomdp":
        return CartPolePOMDPEnv(
            CartPolePOMDPConfig(
                noise_type=config["noise"],
                gaussian_std=float(config.get("noise_std", 0.02)),
                uniform_bound=float(config.get("uniform_bound", 0.05)),
                exponential_scale=float(config.get("exponential_scale", 0.1)),
                correlated_rho=float(config.get("correlated_rho", 0.9)),
                spike_prob=float(config.get("spike_prob", 0.2)),
                reward_scale=float(config.get("reward_scale", 0.1)),
                seed=seed,
                is_non_stationary=bool(config.get("is_non_stationary", False)),
            )
        )
    if env_type == "cartpole_correlated":
        return CartPoleCorrelatedNoise(
            CorrelatedNoiseConfig(
                rho=float(config.get("rho", 0.9)),
                sigma=float(config.get("sigma", 0.02)),
                reward_scale=float(config.get("reward_scale", 1.0)),
                seed=seed,
            )
        )
    if env_type == "cartpole_delay":
        return CartPoleDelay(
            DelayConfig(
                delay=int(config.get("delay", 3)),
                reward_scale=float(config.get("reward_scale", 1.0)),
                seed=seed,
            )
        )
    if env_type == "telemetry":
        return TelemetryEnv(
            TelemetryConfig(
                noise_type=config["noise"],
                reward_scale=float(config.get("reward_scale", 1.0)),
                max_steps=int(config.get("max_steps", 100)),
                seed=seed,
            )
        )
    raise ValueError(f"Unknown env type: {env_type!r}")


def epsilon_for_episode(config: dict, episode: int) -> float:
    start = float(config["epsilon_start"])
    end = float(config["epsilon_end"])
    decay = float(config.get("epsilon_decay", 0.995))
    if decay < 1.0:
        return max(end, start * (decay ** max(0, episode - 1)))
    ratio = min(1.0, max(0, episode - 1) / max(1.0, decay))
    return start + ratio * (end - start)


@dataclass(slots=True)
class RunnerOutput:
    csv_path: Path
    rewards: list[float]
    td_means: list[float]
    td_vars: list[float]
    losses: list[float]
    summary: dict[str, float]


def _finalize_output(
    csv_path: Path,
    rewards: list[float],
    td_means: list[float],
    td_vars: list[float],
    losses: list[float],
) -> RunnerOutput:
    bundle = evaluate_run(rewards, td_means, losses)
    summary = {
        "mean_reward": bundle.metrics.mean_reward,
        "reward_std": bundle.metrics.reward_std,
        "mean_td": bundle.metrics.mean_td,
        "td_variance": bundle.metrics.td_variance,
        "mean_loss": bundle.metrics.mean_loss,
        "convergence_episode": float(bundle.metrics.convergence_episode),
        "stability_std_last_50": bundle.metrics.stability_std_last_50,
    }
    return RunnerOutput(
        csv_path=csv_path,
        rewards=rewards,
        td_means=td_means,
        td_vars=td_vars,
        losses=losses,
        summary=summary,
    )


# ── Shared ChDQN config factory ───────────────────────────────────────────────

def _make_chdqn_config(config: dict, seed: int, *, variant: str = "V2") -> ChDQNConfig:
    """Build a ChDQNConfig from YAML dict, with variant-specific overrides."""
    obs_dim = int(config.get("obs_dim", 2))
    cfg = ChDQNConfig(
        obs_dim=obs_dim,
        latent_dim=max(4, obs_dim * 2),
        num_actions=int(config.get("num_actions", 2)),
        gamma=float(config["gamma"]),
        learning_rate=float(config["learning_rate"]),
        sequence_length=int(config["seq_len"]),
        batch_size=int(config["batch_size"]),
        epsilon_start=float(config["epsilon_start"]),
        epsilon_end=float(config["epsilon_end"]),
        epsilon_decay=max(1, int(config.get("epsilon_decay", 10_000))),
        lambda_cons=float(config.get("lambda1", 0.01)),
        lambda_pred=float(config.get("lambda2", 0.05)),
        reward_scale=float(config.get("reward_scale", 0.1)),
        seed=seed,
        train_episodes=int(config["episodes"]),
        eval_episodes=int(config.get("eval_episodes", 20)),
        max_steps_per_episode=int(config["max_steps"]),
        min_replay_sequences=max(_MIN_REPLAY, int(config["batch_size"])),
        train_updates_per_episode=int(config.get("train_updates_per_episode", 4)),
        is_non_stationary=bool(config.get("is_non_stationary", False)),
    )
    # Variant-specific overrides
    if variant == "V0":
        cfg.reliability_floor = 1.0  # c_prime = 1 → disables weighting
        cfg.use_v2 = False
    elif variant == "V1":
        cfg.use_v2 = False
    elif variant == "V2":
        cfg.use_v2 = True
    return cfg


# ── Unified ChDQN variant runner ─────────────────────────────────────────────

def _run_chdqn_variant(
    config_path: str | Path,
    *,
    seed: int = 0,
    variant: str = "V2",
    episodes_override: int | None = None,
) -> RunnerOutput:
    config = load_runner_config(config_path, seed=seed)
    if episodes_override:
        config["episodes"] = episodes_override
    set_global_seed(seed)

    model_config = _make_chdqn_config(config, seed, variant=variant)
    trainer = ChDQNTrainer(model_config)
    collector = RolloutCollector(model_config)
    replay = SequenceReplayBuffer(capacity=50_000, sequence_length=model_config.sequence_length)

    label = variant.lower()
    out_dir = resolve_results_dir(Path(config_path).stem)
    csv_path = out_dir / f"{label}_seed{seed}.csv"

    rewards: list[float] = []
    td_means: list[float] = []
    td_vars: list[float] = []
    losses: list[float] = []

    with BenchmarkLogger(csv_path) as logger:
        for episode in range(1, model_config.train_episodes + 1):
            env = make_env(config, seed + episode)
            rollout = collector.collect_episode(
                env, trainer.model, replay,
                max_steps=model_config.max_steps_per_episode,
            )
            env.close()
            reward = rollout.episode_reward
            td_mean = td_var = loss = 0.0
            latent_gap = reliability = alpha = gamma_film = 0.0

            if len(replay) >= model_config.min_replay_sequences:
                for _ in range(model_config.train_updates_per_episode):
                    stats, _, metrics = trainer.train_on_replay(
                        replay, batch_size=min(model_config.batch_size, len(replay))
                    )
                    td_mean = metrics.get("td_mean", 0.0)
                    td_var = metrics.get("td_var", 0.0)
                    loss = metrics.get("loss", 0.0)
                    latent_gap = metrics.get("latent_gap", 0.0)
                    reliability = metrics.get("reliability", 1.0)
                    alpha = metrics.get("alpha", 0.0)
                    gamma_film = metrics.get("gamma_film", 1.0)

            rewards.append(reward)
            td_means.append(td_mean)
            td_vars.append(td_var)
            losses.append(loss)
            logger.log(
                episode=episode,
                reward=reward,
                td_mean=td_mean,
                td_var=td_var,
                loss=loss,
                latent_gap=latent_gap,
                reliability=reliability,
                alpha=alpha,
                gamma_film=gamma_film,
                model=variant,
                noise=config["noise"],
                seed=seed,
            )

    return _finalize_output(csv_path, rewards, td_means, td_vars, losses)


def run_v0(config_path, *, seed=0, episodes_override=None) -> RunnerOutput:
    return _run_chdqn_variant(config_path, seed=seed, variant="V0", episodes_override=episodes_override)


def run_v1(config_path, *, seed=0, episodes_override=None) -> RunnerOutput:
    return _run_chdqn_variant(config_path, seed=seed, variant="V1", episodes_override=episodes_override)


def run_v2(config_path, *, seed=0, episodes_override=None) -> RunnerOutput:
    return _run_chdqn_variant(config_path, seed=seed, variant="V2", episodes_override=episodes_override)


# keep original name for backward compat
def run_chdqn(config_path, *, seed=0, episodes_override=None) -> RunnerOutput:
    return _run_chdqn_variant(config_path, seed=seed, variant="V2", episodes_override=episodes_override)


# ── DQN ───────────────────────────────────────────────────────────────────────

def run_dqn(config_path: str | Path, *, seed: int = 0, episodes_override: int | None = None) -> RunnerOutput:
    config = load_runner_config(config_path, seed=seed)
    if episodes_override:
        config["episodes"] = episodes_override
    set_global_seed(seed)

    obs_dim = int(config.get("obs_dim", 2))
    num_actions = int(config.get("num_actions", 2))
    episodes = int(config["episodes"])
    gamma = float(config["gamma"])
    lr = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    max_steps = int(config["max_steps"])
    epsilon = float(config["epsilon_start"])

    model = DQNBaseline(obs_dim=obs_dim, num_actions=num_actions)
    target = DQNBaseline(obs_dim=obs_dim, num_actions=num_actions)
    target.load_state_dict(model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    replay: deque = deque(maxlen=10_000)

    out_dir = resolve_results_dir(Path(config_path).stem)
    csv_path = out_dir / f"dqn_seed{seed}.csv"
    rewards: list[float] = []
    td_means: list[float] = []
    td_vars: list[float] = []
    losses: list[float] = []

    with BenchmarkLogger(csv_path) as logger:
        for episode in range(1, episodes + 1):
            env = make_env(config, seed + episode)
            obs = torch.tensor(env.reset(), dtype=torch.float32)
            total_reward = 0.0
            episode_td: list[float] = []
            episode_loss: list[float] = []
            for _ in range(max_steps):
                if random.random() < epsilon:
                    action = random.randrange(num_actions)
                else:
                    action = int(model(obs.unsqueeze(0)).argmax(dim=-1).item())
                next_obs_raw, reward, done, _ = env.step(action)
                next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)
                replay.append((obs.clone(), action, reward, next_obs.clone(), done))
                total_reward += reward
                obs = next_obs
                if len(replay) >= max(_MIN_REPLAY, batch_size):
                    sample = random.sample(list(replay), batch_size)
                    b_obs = torch.stack([s[0] for s in sample])
                    b_action = torch.tensor([s[1] for s in sample], dtype=torch.long)
                    b_reward = torch.tensor([s[2] for s in sample], dtype=torch.float32)
                    b_next = torch.stack([s[3] for s in sample])
                    b_done = torch.tensor([s[4] for s in sample], dtype=torch.float32)
                    q = model(b_obs).gather(1, b_action.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        y = b_reward + gamma * (1.0 - b_done) * target(b_next).max(dim=1).values
                    td = q - y
                    loss = F.smooth_l1_loss(q, y)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    for tp, sp in zip(target.parameters(), model.parameters()):
                        tp.data.mul_(0.95).add_(sp.data, alpha=0.05)
                    episode_td.extend(td.detach().tolist())
                    episode_loss.append(float(loss.detach()))
                if done:
                    break
            env.close()
            epsilon = epsilon_for_episode(config, episode)
            td_mean = float(torch.tensor(episode_td).mean()) if episode_td else 0.0
            td_var = float(torch.tensor(episode_td).var(unbiased=False)) if episode_td else 0.0
            loss_val = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
            rewards.append(total_reward)
            td_means.append(td_mean)
            td_vars.append(td_var)
            losses.append(loss_val)
            logger.log(episode=episode, reward=total_reward, td_mean=td_mean, td_var=td_var,
                       loss=loss_val, model="DQN", noise=config["noise"], seed=seed)
    return _finalize_output(csv_path, rewards, td_means, td_vars, losses)


# ── DRQN ──────────────────────────────────────────────────────────────────────

def run_drqn(config_path: str | Path, *, seed: int = 0, episodes_override: int | None = None) -> RunnerOutput:
    config = load_runner_config(config_path, seed=seed)
    if episodes_override:
        config["episodes"] = episodes_override
    set_global_seed(seed)

    obs_dim = int(config.get("obs_dim", 2))
    num_actions = int(config.get("num_actions", 2))
    episodes = int(config["episodes"])
    gamma = float(config["gamma"])
    lr = float(config["learning_rate"])
    seq_len = int(config["seq_len"])
    max_steps = int(config["max_steps"])
    epsilon = float(config["epsilon_start"])

    model = DRQNBaseline(obs_dim=obs_dim, num_actions=num_actions)
    target = DRQNBaseline(obs_dim=obs_dim, num_actions=num_actions)
    target.load_state_dict(model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    replay: deque = deque(maxlen=1000)

    out_dir = resolve_results_dir(Path(config_path).stem)
    csv_path = out_dir / f"drqn_seed{seed}.csv"
    rewards: list[float] = []
    td_means: list[float] = []
    td_vars: list[float] = []
    losses: list[float] = []

    with BenchmarkLogger(csv_path) as logger:
        for episode in range(1, episodes + 1):
            env = make_env(config, seed + episode)
            obs = torch.tensor(env.reset(), dtype=torch.float32)
            hidden = model.init_hidden()
            trajectory: list = []
            total_reward = 0.0
            for _ in range(max_steps):
                q, hidden = model.forward_step(obs.unsqueeze(0), hidden)
                if random.random() < epsilon:
                    action = random.randrange(num_actions)
                else:
                    action = int(q.argmax(dim=-1).item())
                next_obs_raw, reward, done, _ = env.step(action)
                next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)
                trajectory.append((obs.clone(), action, reward, next_obs.clone(), done))
                total_reward += reward
                obs = next_obs
                if done:
                    break
            env.close()
            if len(trajectory) >= seq_len:
                replay.append(trajectory)
            episode_td: list[float] = []
            episode_loss: list[float] = []
            if len(replay) >= _MIN_REPLAY:
                sampled = random.choice(list(replay))
                window = sampled[:seq_len]
                h = model.init_hidden()
                th = target.init_hidden()
                loss_terms = []
                for obs_t, act_t, rew_t, nobs_t, done_t in window:
                    q, h = model.forward_step(obs_t.unsqueeze(0), h)
                    with torch.no_grad():
                        qn, th = target.forward_step(nobs_t.unsqueeze(0), th)
                        y = torch.tensor([rew_t], dtype=torch.float32) + gamma * (1.0 - float(done_t)) * qn.max(1).values
                    chosen = q[:, act_t]
                    episode_td.append(float((chosen - y).detach()))
                    loss_terms.append(F.smooth_l1_loss(chosen, y))
                if loss_terms:
                    loss = torch.stack(loss_terms).mean()
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    for tp, sp in zip(target.parameters(), model.parameters()):
                        tp.data.mul_(0.95).add_(sp.data, alpha=0.05)
                    episode_loss.append(float(loss.detach()))
            epsilon = epsilon_for_episode(config, episode)
            td_mean = float(torch.tensor(episode_td).mean()) if episode_td else 0.0
            td_var = float(torch.tensor(episode_td).var(unbiased=False)) if episode_td else 0.0
            loss_val = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
            rewards.append(total_reward)
            td_means.append(td_mean)
            td_vars.append(td_var)
            losses.append(loss_val)
            logger.log(episode=episode, reward=total_reward, td_mean=td_mean, td_var=td_var,
                       loss=loss_val, model="DRQN", noise=config["noise"], seed=seed)
    return _finalize_output(csv_path, rewards, td_means, td_vars, losses)


# ── R2D2 ──────────────────────────────────────────────────────────────────────

def run_r2d2(config_path: str | Path, *, seed: int = 0, episodes_override: int | None = None) -> RunnerOutput:
    """R2D2 with LSTM, sequence replay, burn-in, and target network sync."""
    config = load_runner_config(config_path, seed=seed)
    if episodes_override:
        config["episodes"] = episodes_override
    set_global_seed(seed)

    obs_dim = int(config.get("obs_dim", 2))
    num_actions = int(config.get("num_actions", 2))
    episodes = int(config["episodes"])
    gamma = float(config["gamma"])
    lr = float(config["learning_rate"])
    seq_len = int(config["seq_len"])
    burn_in = max(2, seq_len // 4)          # burn-in = 25% of sequence
    max_steps = int(config["max_steps"])
    epsilon = float(config["epsilon_start"])
    batch_size = int(config["batch_size"])

    r2d2_cfg = ChDQNConfig(obs_dim=obs_dim, num_actions=num_actions)
    model = R2D2Baseline(r2d2_cfg)
    target = R2D2Baseline(r2d2_cfg)
    target.load_state_dict(model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Episode-level sequence replay: each entry is a list of (obs, action, reward, next_obs, done)
    replay: deque = deque(maxlen=2000)

    out_dir = resolve_results_dir(Path(config_path).stem)
    csv_path = out_dir / f"r2d2_seed{seed}.csv"
    rewards: list[float] = []
    td_means: list[float] = []
    td_vars: list[float] = []
    losses: list[float] = []

    target_sync_freq = 10  # hard-sync target every N episodes

    with BenchmarkLogger(csv_path) as logger:
        for episode in range(1, episodes + 1):
            env = make_env(config, seed + episode)
            obs = torch.tensor(env.reset(), dtype=torch.float32)
            hidden = model.init_hidden()
            trajectory: list = []
            total_reward = 0.0
            for _ in range(max_steps):
                q, hidden = model.forward_step(obs.unsqueeze(0), hidden)
                if random.random() < epsilon:
                    action = random.randrange(num_actions)
                else:
                    action = int(q.argmax(dim=-1).item())
                nobs_raw, reward, done, _ = env.step(action)
                nobs = torch.tensor(nobs_raw, dtype=torch.float32)
                trajectory.append((obs.clone(), action, reward, nobs.clone(), done))
                total_reward += reward
                obs = nobs
                if done:
                    break
            env.close()
            if len(trajectory) >= seq_len + burn_in:
                replay.append(trajectory)

            episode_td: list[float] = []
            episode_loss: list[float] = []

            if len(replay) >= _MIN_REPLAY:
                for _ in range(4):  # multiple updates per episode
                    sampled = random.sample(list(replay), min(batch_size, len(replay)))
                    batch_loss = torch.tensor(0.0)
                    batch_td: list[float] = []
                    for seq in sampled:
                        # Sample a contiguous window of length burn_in + seq_len
                        window_len = burn_in + seq_len
                        if len(seq) < window_len:
                            continue
                        start = random.randint(0, len(seq) - window_len)
                        window = seq[start: start + window_len]

                        # Burn-in: warm up LSTM hidden state without gradient
                        with torch.no_grad():
                            h = model.init_hidden()
                            th = target.init_hidden()
                            for obs_t, _, _, _, _ in window[:burn_in]:
                                _, h = model.forward_step(obs_t.unsqueeze(0), h)
                                _, th = target.forward_step(obs_t.unsqueeze(0), th)

                        # Learning window (after burn-in)
                        for obs_t, act_t, rew_t, nobs_t, done_t in window[burn_in:]:
                            q_all, h = model.forward_step(obs_t.unsqueeze(0), h)
                            with torch.no_grad():
                                # Double-DQN: select action from online, value from target
                                online_next, _ = model.forward_step(nobs_t.unsqueeze(0), h)
                                best_a = online_next.argmax(dim=-1, keepdim=True)
                                qn, th = target.forward_step(nobs_t.unsqueeze(0), th)
                                qn_best = qn.gather(1, best_a).squeeze()
                                y = (torch.tensor([rew_t], dtype=torch.float32)
                                     + gamma * (1.0 - float(done_t)) * qn_best)
                            q_chosen = q_all[:, act_t]
                            td_e = float((q_chosen - y).detach())
                            batch_td.append(td_e)
                            batch_loss = batch_loss + F.smooth_l1_loss(q_chosen, y)

                    if batch_td:
                        loss = batch_loss / max(len(batch_td), 1)
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                        episode_td.extend(batch_td)
                        episode_loss.append(float(loss.detach()))

            # Hard target sync every N episodes
            if episode % target_sync_freq == 0:
                target.load_state_dict(model.state_dict())

            epsilon = epsilon_for_episode(config, episode)
            td_mean = float(torch.tensor(episode_td).mean()) if episode_td else 0.0
            td_var = float(torch.tensor(episode_td).var(unbiased=False)) if episode_td else 0.0
            loss_val = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
            rewards.append(total_reward)
            td_means.append(td_mean)
            td_vars.append(td_var)
            losses.append(loss_val)
            logger.log(episode=episode, reward=total_reward, td_mean=td_mean, td_var=td_var,
                       loss=loss_val, model="R2D2", noise=config["noise"], seed=seed)

    return _finalize_output(csv_path, rewards, td_means, td_vars, losses)
