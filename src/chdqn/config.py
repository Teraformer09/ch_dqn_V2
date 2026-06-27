from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ChDQNConfig:
    obs_dim: int = 2
    latent_dim: int = 2
    num_actions: int = 2
    sequence_length: int = 10
    hidden_width: int = 16
    gamma: float = 0.99
    reward_scale: float = 1.0
    td_clip: float = 100.0
    lambda_cons: float = 0.05
    lambda_pred: float = 0.05
    learning_rate: float = 0.02
    grad_clip_norm: float = 1.0
    beta1: float = 0.15
    beta2: float = 0.05
    target_tau: float = 0.05
    latent_clip: float = 5.0
    seed: int = 7
    use_layer_norm: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10000
    batch_size: int = 64
    train_episodes: int = 300
    eval_episodes: int = 20
    max_steps_per_episode: int = 200
    min_replay_sequences: int = 100
    train_updates_per_episode: int = 4
    smoothing_window: int = 3
    ema_tau: float = 0.005
    # V1: reliability-weighted Bellman operator
    reliability_floor: float = 0.30
    reliability_scale: float = 1.0
    # V2: adaptive meta-control
    use_v2: bool = False
    meta_hidden: int = 16
    meta_alpha_max: float = 0.2
    film_gamma_min: float = 0.8
    film_gamma_max: float = 1.0
    film_beta_bound: float = 0.5
    meta_variance_weight: float = 0.1
    gamma_smooth_weight: float = 0.01
    # R2D2-lite baseline
    r2d2_sequence_length: int = 20
    r2d2_burn_in: int = 5
    r2d2_hidden_dim: int = 128
    r2d2_use_priority: bool = True
    r2d2_n_step: int = 3
    is_non_stationary: bool = False
    # Anti-collapse: Gaussian noise injected into encoder output during training
    latent_noise_std: float = 0.05


@dataclass(slots=True)
class ReferenceConfig:
    encoder_matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (0.35, 0.30),
        (0.15, 0.50),
    )
    encoder_bias: tuple[float, float] = (0.0, 0.0)
    filter_matrix_h: tuple[tuple[float, float], tuple[float, float]] = (
        (0.40, 0.10),
        (-0.05, 0.50),
    )
    filter_matrix_z: tuple[tuple[float, float], tuple[float, float]] = (
        (1.00, 0.50),
        (-1.00, -1.00),
    )
    filter_bias: tuple[float, float] = (-0.0011, -0.0196)
    q_matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (1.16, 0.0),
        (-0.34, 0.0),
    )
    q_bias: tuple[float, float] = (-0.012, 0.028)
    dry_run_latents: tuple[tuple[float, float], ...] = (
        (0.05, -0.02),
        (0.08, -0.05),
        (0.10, -0.08),
        (0.12, -0.11),
        (0.14, -0.13),
        (0.15, -0.14),
        (0.16, -0.15),
        (0.17, -0.16),
        (0.18, -0.17),
        (0.19, -0.18),
    )
    epoch_q0: tuple[float, ...] = (0.129, 0.147, 0.168, 0.185, 0.198, 0.207, 0.215, 0.221)
    epoch_td: tuple[float, ...] = (-0.10, -0.082, -0.061, -0.044, -0.031, -0.022, -0.014, -0.008)
    noisy_sequence: tuple[tuple[float, float], ...] = (
        (0.04, -0.01),
        (0.09, -0.06),
        (0.07, -0.02),
        (0.11, -0.10),
        (0.13, -0.12),
        (0.14, -0.12),
        (0.15, -0.16),
        (0.16, -0.14),
        (0.18, -0.18),
        (0.20, -0.19),
    )
    noise_cases: tuple[str, ...] = ("gaussian", "uniform", "exponential", "correlated", "mixed")
    metrics_template: dict[str, float] = field(
        default_factory=lambda: {
            "gaussian": 0.65,
            "uniform": 0.35,
            "exponential": -0.25,
            "correlated": 0.90,
            "mixed": 0.75,
        }
    )
