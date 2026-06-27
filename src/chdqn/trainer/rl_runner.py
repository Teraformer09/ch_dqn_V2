from __future__ import annotations

import copy
from dataclasses import dataclass

from chdqn.config import ChDQNConfig
from chdqn.envs.cartpole_pomdp import CartPolePOMDPConfig, CartPolePOMDPEnv
from chdqn.evaluation import EvaluationStats, evaluate_model
from chdqn.logger import CSVMetricLogger, LogRecord
from chdqn.replay import SequenceReplayBuffer
from chdqn.rollout import RolloutCollector
from chdqn.trainer import ChDQNTrainer


@dataclass(slots=True)
class RLTrainSummary:
    episodes: int
    mean_train_reward: float
    final_train_reward: float
    final_td_var: float
    evaluation: EvaluationStats


# ── V0/V1/V2 distinction ──────────────────────────────────────────────────────
#
# V0: temporal smoother active; c_prime = 1.0 (floor=1.0, disables weighting);
#     no meta controller.  Isolates the pure variance-reduction claim.
# V1: smoother active; c_prime computed from reliability; no meta controller.
#     Tests the reliability-weighted Bellman operator.
# V2: smoother active; c_prime from reliability; meta controller active.
#     Full system.
# ─────────────────────────────────────────────────────────────────────────────


def _make_config_for_variant(base: ChDQNConfig, variant: str) -> ChDQNConfig:
    """Return a shallow copy of base with variant-specific overrides."""
    import dataclasses
    cfg = dataclasses.replace(base)
    if variant == "V0":
        cfg.reliability_floor = 1.0   # c_prime always 1 → no weighting
        cfg.use_v2 = False
    elif variant == "V1":
        cfg.use_v2 = False
    elif variant == "V2":
        cfg.use_v2 = True
    return cfg


class CartPoleRLRunner:
    def __init__(self, config: ChDQNConfig, model_type: str = "V2", *, noise_type: str | None = "gaussian") -> None:
        self.config = config
        self.model_type = model_type
        self.noise_type = noise_type

        variant_config = _make_config_for_variant(config, model_type)

        if model_type == "DQN":
            from chdqn.models import DQNBaseline
            self.model = DQNBaseline(obs_dim=config.obs_dim, hidden_dim=config.hidden_width, num_actions=config.num_actions)
        elif model_type == "DRQN":
            from chdqn.models import DRQNBaseline
            self.model = DRQNBaseline(obs_dim=config.obs_dim, hidden_dim=config.hidden_width, num_actions=config.num_actions)
        elif model_type == "R2D2":
            from chdqn.models import R2D2Baseline
            self.model = R2D2Baseline(config)
        else:
            from chdqn.models import ChDQNModel
            self.model = ChDQNModel(variant_config)

        self.trainer = ChDQNTrainer(variant_config)
        self.trainer.model = self.model
        self.trainer.target_model = copy.deepcopy(self.model)
        self.trainer.sgd = self.trainer._init_sgd()
        self.trainer.second_order = self.trainer._init_second_order()

        self.replay = SequenceReplayBuffer(
            capacity=50_000,
            sequence_length=config.sequence_length,
        )
        self.collector = RolloutCollector(variant_config)
        self._variant_config = variant_config

    def make_env(self, episode: int = 0) -> CartPolePOMDPEnv:
        return CartPolePOMDPEnv(
            CartPolePOMDPConfig(
                noise_type=self.noise_type,
                reward_scale=self.config.reward_scale,
                seed=self.config.seed + episode,
                is_non_stationary=self.config.is_non_stationary,
                episode=episode,
            )
        )

    def train(self, *, log_path: str | None = None, seed: int = 0) -> RLTrainSummary:
        rewards: list[float] = []
        last_metrics: dict = {
            "td_mean": 0.0, "td_var": 0.0, "loss": 0.0,
            "latent_gap": 0.0, "reliability": 1.0, "alpha": 0.0, "gamma_film": 1.0,
        }

        logger = CSVMetricLogger(log_path) if log_path is not None else None

        try:
            for episode in range(1, self._variant_config.train_episodes + 1):
                env = self.make_env(episode)
                rollout_stats = self.collector.collect_episode(
                    env,
                    self.model,
                    self.replay,
                    max_steps=self._variant_config.max_steps_per_episode,
                )
                # Average noise_std seen this episode
                noise_stds = [s.get("noise_std", 0.0) for s in rollout_stats.step_data]
                mean_noise_std = sum(noise_stds) / len(noise_stds) if noise_stds else 0.0
                env.close()
                rewards.append(rollout_stats.episode_reward)

                if len(self.replay) >= self._variant_config.min_replay_sequences:
                    for _ in range(self._variant_config.train_updates_per_episode):
                        stats, _, metrics = self.trainer.train_on_replay(
                            self.replay,
                            batch_size=min(self._variant_config.batch_size, len(self.replay)),
                        )
                        if metrics:
                            last_metrics = metrics

                if logger is not None:
                    logger.log(LogRecord(
                        episode=episode,
                        step=episode,
                        reward=rollout_stats.episode_reward,
                        td_mean=float(last_metrics.get("td_mean", 0.0)),
                        td_var=float(last_metrics.get("td_var", 0.0)),
                        loss=float(last_metrics.get("loss", 0.0)),
                        noise_std=mean_noise_std,
                        latent_gap=float(last_metrics.get("latent_gap", 0.0)),
                        alpha_t=float(last_metrics.get("alpha", 0.0)),
                        gamma_t=float(last_metrics.get("gamma_film", 1.0)),
                        c_t=float(last_metrics.get("reliability", 1.0)),
                        model=self.model_type,
                        seed=seed,
                    ))
        finally:
            if logger is not None:
                logger.close()

        evaluation = evaluate_model(
            self.trainer.model,
            self.make_env,
            episodes=self._variant_config.eval_episodes,
            max_steps=self._variant_config.max_steps_per_episode,
        )
        return RLTrainSummary(
            episodes=self._variant_config.train_episodes,
            mean_train_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            final_train_reward=rewards[-1] if rewards else 0.0,
            final_td_var=float(last_metrics.get("td_var", 0.0)),
            evaluation=evaluation,
        )
