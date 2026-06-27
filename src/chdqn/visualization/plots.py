"""Plotting pipeline for Ch-DQN paper figures.

Groups:
  A  — Core theory validation  (A1: TD variance, A2: latent gap, A3: Q convergence)
  B  — V1 operator behavior    (B1: reliability weight, B2: bias visualization)
  C  — V2 meta dynamics        (C1: α_t, C2: γ_t, C3: α vs TD var, C4: γ vs gap)
  D  — Distribution behavior   (D1: perf vs noise level, D2: TD var by noise type)
  E  — Ablation                (E1: model comparison, E2: remove meta layer)
  F  — Failure visualization   (F1: non-Markovian, F2: bias noise)

Usage:
    from chdqn.plots import generate_all_plots
    generate_all_plots("output/figures/")
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from chdqn.config import ChDQNConfig
from chdqn.noise import correlated_noise, exponential_noise, gaussian_noise, mixed_noise, uniform_noise
from chdqn.reference import reference_sequences
from chdqn.replay import SequenceBatch
from chdqn.trainer import ChDQNTrainer
from chdqn.utils import set_seed

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

_COLORS = {
    "dqn": "#888888",
    "v0": "#4C72B0",
    "v1": "#DD8452",
    "v2": "#55A868",
    "v2_no_film": "#C44E52",
}

_MA_WINDOW = 20  # moving average window for smoothing curves


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")


def _moving_average(values: list[float], window: int = _MA_WINDOW) -> list[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def _make_trainer(use_v2: bool = False, seed: int = 7, floor: float = 0.15) -> ChDQNTrainer:
    cfg = ChDQNConfig(seed=seed, use_v2=use_v2, reliability_floor=floor)
    return ChDQNTrainer(cfg, use_reference_init=True)


def _make_batch(obs: torch.Tensor) -> SequenceBatch:
    return SequenceBatch(
        observations=obs,
        actions=torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.long),
        rewards=torch.full((obs.shape[0], obs.shape[1]), 0.1),
        dones=torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.bool),
    )


def _collect_logs(
    trainer: ChDQNTrainer,
    n_steps: int,
    noise_std: float = 0.02,
    noise_schedule: list[float] | None = None,
) -> dict[str, list[float]]:
    """Run n_steps training steps and collect per-step metrics."""
    clean, noisy = reference_sequences()
    logs: dict[str, list[float]] = {
        "td_var": [], "td_mean": [], "loss": [], "latent_gap": [],
        "reliability": [], "alpha": [], "gamma_film": [], "q_variance": [],
    }
    for step in range(n_steps):
        sigma = noise_schedule[step] if noise_schedule else noise_std
        obs = torch.stack([clean + gaussian_noise(clean.shape, std=sigma), noisy])
        stats, _, metrics = trainer.train_step(_make_batch(obs))
        for k in logs:
            logs[k].append(float(metrics.get(k, getattr(stats, k.replace("_film", ""), 0.0))))
    return logs


def _multi_seed_logs(
    use_v2: bool,
    n_steps: int,
    seeds: Sequence[int],
    noise_std: float = 0.02,
    noise_schedule: list[float] | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """Collect logs across multiple seeds; return {key: (mean, std)} per step."""
    all_logs: dict[str, list[list[float]]] = {}
    for seed in seeds:
        trainer = _make_trainer(use_v2=use_v2, seed=seed)
        logs = _collect_logs(trainer, n_steps, noise_std=noise_std, noise_schedule=noise_schedule)
        for k, v in logs.items():
            all_logs.setdefault(k, []).append(v)

    result = {}
    for k, runs in all_logs.items():
        mat = torch.tensor(runs, dtype=torch.float32)
        result[k] = (mat.mean(dim=0).tolist(), mat.std(dim=0).tolist())
    return result


def _save_fig(fig, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_plot(ax, x, mean, std, label, color, alpha_shade=0.2):
    mean_sm = _moving_average(mean)
    std_sm = _moving_average(std)
    ax.plot(x, mean_sm, label=label, color=color, linewidth=1.5)
    ax.fill_between(
        x,
        [m - s for m, s in zip(mean_sm, std_sm)],
        [m + s for m, s in zip(mean_sm, std_sm)],
        alpha=alpha_shade, color=color,
    )


# ---------------------------------------------------------------------------
# Group A — Core theory validation
# ---------------------------------------------------------------------------

def plot_td_variance_comparison(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """A1: TD variance vs training steps for V0, V1, V2."""
    _require_mpl()
    output_dir = Path(output_dir)

    configs = [
        ("v0", False, 1.0),   # floor=1.0 disables V1 weighting
        ("v1", False, 0.15),
        ("v2", True, 0.15),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, use_v2, floor in configs:
        logs_by_seed: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed, floor=floor)
            logs = _collect_logs(trainer, n_steps)
            logs_by_seed.append(logs["td_var"])
        mat = torch.tensor(logs_by_seed, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label.upper(), color=_COLORS[label])

    ax.set_title("A1 — TD Variance vs Training Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(δ_t)")
    ax.legend()
    ax.set_yscale("log")
    _save_fig(fig, output_dir, "A1_td_variance.png")


def plot_latent_gap(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """A2: ||h_t - h_tilde_t|| vs training steps."""
    _require_mpl()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, use_v2 in [("v0", False), ("v1", False), ("v2", True)]:
        logs_by_seed: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed)
            logs = _collect_logs(trainer, n_steps)
            logs_by_seed.append(logs["latent_gap"])
        mat = torch.tensor(logs_by_seed, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label.upper(), color=_COLORS[label])

    ax.set_title("A2 — Latent Gap ||h_t - h̃_t|| vs Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latent Gap")
    ax.legend()
    _save_fig(fig, output_dir, "A2_latent_gap.png")


def plot_q_convergence(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """A3: Q-value variance vs training steps (proxy for Q convergence)."""
    _require_mpl()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, use_v2 in [("v0", False), ("v1", False), ("v2", True)]:
        q_vars: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed)
            logs = _collect_logs(trainer, n_steps)
            q_vars.append(logs["q_variance"])
        mat = torch.tensor(q_vars, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label.upper(), color=_COLORS[label])

    ax.set_title("A3 — Q-Value Variance vs Training Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(Q)")
    ax.legend()
    _save_fig(fig, output_dir, "A3_q_convergence.png")


# ---------------------------------------------------------------------------
# Group B — V1 operator behavior
# ---------------------------------------------------------------------------

def plot_reliability_weight(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """B1: c_t' (reliability weight) vs training steps."""
    _require_mpl()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    # Make non-stationary noise: low then spike then recovery
    noise_sched = [0.01] * (n_steps // 3) + [0.2] * (n_steps // 3) + [0.02] * (n_steps - 2 * (n_steps // 3))

    for label, use_v2 in [("v1", False), ("v2", True)]:
        rels: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed)
            logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
            rels.append(logs["reliability"])
        mat = torch.tensor(rels, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label.upper(), color=_COLORS[label])

    ax.axvline(n_steps // 3, color="red", linestyle="--", alpha=0.5, label="Noise spike")
    ax.axvline(2 * n_steps // 3, color="green", linestyle="--", alpha=0.5, label="Recovery")
    ax.set_ylim([0, 1.1])
    ax.set_title("B1 — Reliability Weight c_t' vs Steps (non-stationary noise)")
    ax.set_xlabel("Step")
    ax.set_ylabel("c_t'")
    ax.legend()
    _save_fig(fig, output_dir, "B1_reliability_weight.png")


def plot_bias_visualization(
    output_dir: str | Path,
    n_steps: int = 150,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """B2: TD mean (proxy for Q bias) — V1 slightly lower than clean, showing conservative estimation."""
    _require_mpl()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, noise_std in [("v0 (no floor)", 0.0), ("v1 (floor=0.15)", 0.05)]:
        td_means: list[list[float]] = []
        for seed in seeds:
            cfg = ChDQNConfig(seed=seed, reliability_floor=noise_std)
            trainer = ChDQNTrainer(cfg, use_reference_init=True)
            logs = _collect_logs(trainer, n_steps, noise_std=0.05)
            td_means.append(logs["td_mean"])
        mat = torch.tensor(td_means, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        color = _COLORS["v0"] if "no floor" in label else _COLORS["v1"]
        _shade_plot(ax, x, mean, std, label=label, color=color)

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title("B2 — TD Mean (Q bias) vs Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean TD Error")
    ax.legend()
    _save_fig(fig, output_dir, "B2_bias_visualization.png")


# ---------------------------------------------------------------------------
# Group C — V2 meta dynamics
# ---------------------------------------------------------------------------

def plot_alpha_t(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """C1: α_t (memory update rate) vs steps with noise regime overlay."""
    _require_mpl()
    output_dir = Path(output_dir)

    noise_sched = [0.01] * (n_steps // 3) + [0.2] * (n_steps // 3) + [0.02] * (n_steps - 2 * (n_steps // 3))

    alphas: list[list[float]] = []
    for seed in seeds:
        trainer = _make_trainer(use_v2=True, seed=seed)
        logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
        alphas.append(logs["alpha"])
    mat = torch.tensor(alphas, dtype=torch.float32)
    mean = mat.mean(dim=0).tolist()
    std = mat.std(dim=0).tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))
    _shade_plot(ax, x, mean, std, label="α_t", color=_COLORS["v2"])
    ax.axvline(n_steps // 3, color="red", linestyle="--", alpha=0.5, label="Noise spike")
    ax.axvline(2 * n_steps // 3, color="green", linestyle="--", alpha=0.5, label="Recovery")
    ax.set_ylim([0, 0.22])
    ax.set_title("C1 — Meta α_t (Memory Update Rate) vs Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("α_t")
    ax.legend()
    _save_fig(fig, output_dir, "C1_alpha_t.png")


def plot_gamma_t(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """C2: γ_t (FiLM scale) vs steps — should dip during noise spike."""
    _require_mpl()
    output_dir = Path(output_dir)

    noise_sched = [0.01] * (n_steps // 3) + [0.2] * (n_steps // 3) + [0.02] * (n_steps - 2 * (n_steps // 3))

    gammas: list[list[float]] = []
    for seed in seeds:
        trainer = _make_trainer(use_v2=True, seed=seed)
        logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
        gammas.append(logs["gamma_film"])
    mat = torch.tensor(gammas, dtype=torch.float32)
    mean = mat.mean(dim=0).tolist()
    std = mat.std(dim=0).tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))
    _shade_plot(ax, x, mean, std, label="γ_t", color=_COLORS["v2"])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="No modulation (γ=1)")
    ax.axvline(n_steps // 3, color="red", linestyle="--", alpha=0.5, label="Noise spike")
    ax.axvline(2 * n_steps // 3, color="green", linestyle="--", alpha=0.5, label="Recovery")
    ax.set_ylim([0.4, 1.6])
    ax.set_title("C2 — FiLM Scale γ_t vs Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("γ_t")
    ax.legend()
    _save_fig(fig, output_dir, "C2_gamma_t.png")


def plot_alpha_vs_td_variance(
    output_dir: str | Path,
    n_steps: int = 300,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """C3: α_t vs TD variance scatter — expected monotone increasing relationship."""
    _require_mpl()
    output_dir = Path(output_dir)

    all_alpha: list[float] = []
    all_td_var: list[float] = []

    for seed in seeds:
        trainer = _make_trainer(use_v2=True, seed=seed)
        # Sweep through different noise levels to get variety
        noise_sched = [0.01 + 0.3 * (step / n_steps) for step in range(n_steps)]
        logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
        all_alpha.extend(logs["alpha"])
        all_td_var.extend(logs["td_var"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_td_var, all_alpha, alpha=0.15, s=8, color=_COLORS["v2"])
    ax.set_title("C3 — α_t vs TD Variance (causal relationship)")
    ax.set_xlabel("Var(δ_t)")
    ax.set_ylabel("α_t")
    ax.set_xlim(left=0)
    ax.set_ylim([0, 0.22])
    _save_fig(fig, output_dir, "C3_alpha_vs_td_variance.png")


def plot_gamma_vs_latent_gap(
    output_dir: str | Path,
    n_steps: int = 300,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """C4: γ_t vs latent gap — expected inverse relationship."""
    _require_mpl()
    output_dir = Path(output_dir)

    all_gamma: list[float] = []
    all_gap: list[float] = []

    for seed in seeds:
        trainer = _make_trainer(use_v2=True, seed=seed)
        noise_sched = [0.01 + 0.3 * (step / n_steps) for step in range(n_steps)]
        logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
        all_gamma.extend(logs["gamma_film"])
        all_gap.extend(logs["latent_gap"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_gap, all_gamma, alpha=0.15, s=8, color=_COLORS["v2"])
    ax.set_title("C4 — γ_t vs Latent Gap (modulation response)")
    ax.set_xlabel("||h_t - h̃_t||")
    ax.set_ylabel("γ_t")
    ax.set_xlim(left=0)
    ax.set_ylim([0.4, 1.6])
    _save_fig(fig, output_dir, "C4_gamma_vs_latent_gap.png")


# ---------------------------------------------------------------------------
# Group D — Distribution behavior
# ---------------------------------------------------------------------------

def plot_performance_vs_noise_level(
    output_dir: str | Path,
    n_steps: int = 100,
    seeds: Sequence[int] = (1, 3, 7),
) -> None:
    """D1: Final TD variance vs noise level for V0, V1, V2."""
    _require_mpl()
    output_dir = Path(output_dir)

    noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    results: dict[str, list[float]] = {"v0": [], "v1": [], "v2": []}

    for sigma in noise_levels:
        for label, use_v2, floor in [("v0", False, 1.0), ("v1", False, 0.15), ("v2", True, 0.15)]:
            final_td_vars = []
            for seed in seeds:
                trainer = _make_trainer(use_v2=use_v2, seed=seed, floor=floor)
                logs = _collect_logs(trainer, n_steps, noise_std=sigma)
                final_td_vars.append(sum(logs["td_var"][-10:]) / 10)
            results[label].append(sum(final_td_vars) / len(final_td_vars))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [("v0", _COLORS["v0"]), ("v1", _COLORS["v1"]), ("v2", _COLORS["v2"])]:
        ax.plot(noise_levels, results[label], marker="o", label=label.upper(), color=color, linewidth=2)

    ax.set_title("D1 — Final TD Variance vs Noise Level")
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Final Var(δ_t)")
    ax.set_xscale("log")
    ax.legend()
    _save_fig(fig, output_dir, "D1_performance_vs_noise.png")


def plot_td_variance_by_noise_type(
    output_dir: str | Path,
    n_steps: int = 100,
    seeds: Sequence[int] = (1, 3, 7),
) -> None:
    """D2: Final TD variance broken down by noise type for V1 and V2."""
    _require_mpl()
    output_dir = Path(output_dir)

    noise_fns = {
        "Gaussian": lambda s: gaussian_noise(s, std=0.05),
        "Uniform": lambda s: uniform_noise(s, bound=0.05),
        "Exponential": lambda s: exponential_noise(s, scale=0.03),
        "Correlated": lambda s: correlated_noise(s, rho=0.8, std=0.02),
        "Mixed": lambda s: mixed_noise(s, std=0.02),
    }

    noise_names = list(noise_fns.keys())
    results: dict[str, list[float]] = {"v1": [], "v2": []}
    clean, noisy = reference_sequences()

    for noise_name, noise_fn in noise_fns.items():
        for label, use_v2 in [("v1", False), ("v2", True)]:
            final_vars = []
            for seed in seeds:
                trainer = _make_trainer(use_v2=use_v2, seed=seed)
                td_vars = []
                for _ in range(n_steps):
                    obs = torch.stack([clean + noise_fn(clean.shape), noisy])
                    _, _, metrics = trainer.train_step(_make_batch(obs))
                    td_vars.append(metrics["td_var"])
                final_vars.append(sum(td_vars[-10:]) / 10)
            results[label].append(sum(final_vars) / len(final_vars))

    x = range(len(noise_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x], results["v1"], width, label="V1", color=_COLORS["v1"], alpha=0.85)
    ax.bar([xi + width / 2 for xi in x], results["v2"], width, label="V2", color=_COLORS["v2"], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(noise_names)
    ax.set_title("D2 — Final TD Variance by Noise Type")
    ax.set_ylabel("Final Var(δ_t)")
    ax.legend()
    _save_fig(fig, output_dir, "D2_td_variance_by_noise.png")


# ---------------------------------------------------------------------------
# Group E — Ablation
# ---------------------------------------------------------------------------

def plot_ablation_comparison(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """E1: V0 vs V1 vs V2(mem only) vs V2(mem+FiLM) on TD variance."""
    _require_mpl()
    output_dir = Path(output_dir)

    ablations = [
        ("V0 (no floor)", False, 1.0, False),
        ("V1 (floor+target)", False, 0.15, False),
        ("V2 (mem+FiLM)", True, 0.15, True),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    colors = [_COLORS["v0"], _COLORS["v1"], _COLORS["v2"]]
    for (label, use_v2, floor, _), color in zip(ablations, colors):
        td_vars_runs: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed, floor=floor)
            logs = _collect_logs(trainer, n_steps, noise_std=0.05)
            td_vars_runs.append(logs["td_var"])
        mat = torch.tensor(td_vars_runs, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label, color=color)

    ax.set_title("E1 — Ablation: TD Variance by Model Variant")
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(δ_t)")
    ax.set_yscale("log")
    ax.legend()
    _save_fig(fig, output_dir, "E1_ablation_comparison.png")


def plot_remove_meta_effect(
    output_dir: str | Path,
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """E2: V2 with vs without meta controller under noise spike."""
    _require_mpl()
    output_dir = Path(output_dir)

    noise_sched = [0.01] * (n_steps // 2) + [0.2] * (n_steps - n_steps // 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, use_v2, color in [("V1 (no meta)", False, _COLORS["v1"]), ("V2 (meta)", True, _COLORS["v2"])]:
        td_vars_runs: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=use_v2, seed=seed)
            logs = _collect_logs(trainer, n_steps, noise_schedule=noise_sched)
            td_vars_runs.append(logs["td_var"])
        mat = torch.tensor(td_vars_runs, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label, color=color)

    ax.axvline(n_steps // 2, color="red", linestyle="--", alpha=0.6, label="Noise spike")
    ax.set_title("E2 — Meta Controller Effect: TD Variance under Noise Spike")
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(δ_t)")
    ax.legend()
    _save_fig(fig, output_dir, "E2_remove_meta_effect.png")


# ---------------------------------------------------------------------------
# Group F — Failure cases
# ---------------------------------------------------------------------------

def plot_failure_nonmarkov(
    output_dir: str | Path,
    n_steps: int = 150,
    seeds: Sequence[int] = (1, 3, 7),
) -> None:
    """F1: Non-Markovian environment — show V2 limitation vs short-window smoother."""
    _require_mpl()
    output_dir = Path(output_dir)

    clean, _ = reference_sequences()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, lag, color in [
        ("V2 (Markov, lag=0)", 0, _COLORS["v2"]),
        ("V2 (lag=3)", 3, _COLORS["v1"]),
        ("V2 (lag=6)", 6, _COLORS["v0"]),
    ]:
        td_vars_runs: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=True, seed=seed)
            td_vars = []
            for _ in range(n_steps):
                # Non-Markovian: shift observations by lag
                obs_shifted = torch.roll(clean, shifts=lag, dims=0)
                obs = torch.stack([obs_shifted + gaussian_noise(clean.shape, std=0.02), clean])
                _, _, metrics = trainer.train_step(_make_batch(obs))
                td_vars.append(metrics["td_var"])
            td_vars_runs.append(td_vars)
        mat = torch.tensor(td_vars_runs, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label, color=color)

    ax.set_title("F1 — Failure: Non-Markovian Structure (long-lag dependencies)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(δ_t)")
    ax.legend()
    _save_fig(fig, output_dir, "F1_nonmarkov_failure.png")


def plot_failure_bias_noise(
    output_dir: str | Path,
    n_steps: int = 150,
    seeds: Sequence[int] = (1, 3, 7),
) -> None:
    """F2: Exponential (positively biased) noise — Q underestimation visible."""
    _require_mpl()
    output_dir = Path(output_dir)

    clean, _ = reference_sequences()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(n_steps))

    for label, noise_fn, color in [
        ("Gaussian (no bias)", lambda: gaussian_noise(clean.shape, std=0.02), _COLORS["v1"]),
        ("Exponential (biased)", lambda: exponential_noise(clean.shape, scale=0.02), _COLORS["v0"]),
    ]:
        td_means_runs: list[list[float]] = []
        for seed in seeds:
            trainer = _make_trainer(use_v2=False, seed=seed)
            td_means = []
            for _ in range(n_steps):
                obs = torch.stack([clean + noise_fn(), clean])
                _, _, metrics = trainer.train_step(_make_batch(obs))
                td_means.append(metrics["td_mean"])
            td_means_runs.append(td_means)
        mat = torch.tensor(td_means_runs, dtype=torch.float32)
        mean = mat.mean(dim=0).tolist()
        std = mat.std(dim=0).tolist()
        _shade_plot(ax, x, mean, std, label=label, color=color)

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title("F2 — Failure: Biased Noise Causes Q Underestimation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean TD Error (Q bias proxy)")
    ax.legend()
    _save_fig(fig, output_dir, "F2_bias_noise_failure.png")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all_plots(
    output_dir: str | Path = "output/figures",
    n_steps: int = 200,
    seeds: Sequence[int] = (1, 3, 7, 13, 42),
) -> None:
    """Generate all paper figures (A1–A3, B1–B2, C1–C4, D1–D2, E1–E2, F1–F2).

    Args:
        output_dir: Directory to save PNG files.
        n_steps: Training steps per run (increase for smoother curves).
        seeds: Random seeds for multi-seed averaging.
    """
    _require_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plots] Generating figures -> {output_dir}")
    print("[plots] Group A: Core theory validation")
    plot_td_variance_comparison(output_dir, n_steps=n_steps, seeds=seeds)
    print("  A1 done")
    plot_latent_gap(output_dir, n_steps=n_steps, seeds=seeds)
    print("  A2 done")
    plot_q_convergence(output_dir, n_steps=n_steps, seeds=seeds)
    print("  A3 done")

    print("[plots] Group B: V1 operator behavior")
    plot_reliability_weight(output_dir, n_steps=n_steps, seeds=seeds)
    print("  B1 done")
    plot_bias_visualization(output_dir, n_steps=min(n_steps, 150), seeds=seeds)
    print("  B2 done")

    print("[plots] Group C: V2 meta dynamics")
    plot_alpha_t(output_dir, n_steps=n_steps, seeds=seeds)
    print("  C1 done")
    plot_gamma_t(output_dir, n_steps=n_steps, seeds=seeds)
    print("  C2 done")
    plot_alpha_vs_td_variance(output_dir, n_steps=n_steps + 100, seeds=seeds)
    print("  C3 done")
    plot_gamma_vs_latent_gap(output_dir, n_steps=n_steps + 100, seeds=seeds)
    print("  C4 done")

    print("[plots] Group D: Distribution behavior")
    plot_performance_vs_noise_level(output_dir, n_steps=min(n_steps, 100), seeds=seeds[:3])
    print("  D1 done")
    plot_td_variance_by_noise_type(output_dir, n_steps=min(n_steps, 100), seeds=seeds[:3])
    print("  D2 done")

    print("[plots] Group E: Ablation")
    plot_ablation_comparison(output_dir, n_steps=n_steps, seeds=seeds)
    print("  E1 done")
    plot_remove_meta_effect(output_dir, n_steps=n_steps, seeds=seeds)
    print("  E2 done")

    print("[plots] Group F: Failure cases")
    plot_failure_nonmarkov(output_dir, n_steps=min(n_steps, 150), seeds=seeds[:3])
    print("  F1 done")
    plot_failure_bias_noise(output_dir, n_steps=min(n_steps, 150), seeds=seeds[:3])
    print("  F2 done")

    print(f"[plots] All 14 figures saved to {output_dir}")
