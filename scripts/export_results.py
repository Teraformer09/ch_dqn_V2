"""Regenerate all 8 publication-quality plots from existing results/ CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.plotting.plot_ablation import plot_ablation
from experiments.plotting.plot_nonstationary import plot_nonstationary_recovery
from experiments.plotting.plot_rewards import plot_reward_curves
from experiments.plotting.plot_signals import plot_alpha, plot_latent_gap, plot_reliability
from experiments.plotting.plot_td_variance import plot_td_variance_curves

_RESULTS_DIR = PROJECT_ROOT / "results"
_OUT_DIR = PROJECT_ROOT / "output" / "figures"

_NOISE_TYPES = ["gaussian", "uniform", "exponential", "correlated", "mixed"]
_ALL_MODELS = ["DQN", "DRQN", "R2D2", "V0", "V1", "V2"]
_SIGNAL_MODELS = ["V0", "V1", "V2"]
_R2D2_MODELS = ["DQN", "DRQN", "R2D2", "V2"]
_SEEDS = [0, 42, 123]


def _csvs_for(noise: str, models: list[str]) -> list[Path]:
    found = []
    for model in models:
        for seed in _SEEDS:
            p = _RESULTS_DIR / f"{model}_{noise}_seed{seed}.csv"
            if p.exists():
                found.append(p)
    return found


def generate_plots() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for noise in _NOISE_TYPES:
        all_csvs = _csvs_for(noise, _ALL_MODELS)
        if not all_csvs:
            print(f"No CSVs found for noise={noise}, skipping.")
            continue

        suffix = noise
        # Plot 1: reward comparison
        plot_reward_curves(all_csvs, _OUT_DIR / f"reward_{suffix}.png")
        # Plot 2: TD variance comparison
        plot_td_variance_curves(all_csvs, _OUT_DIR / f"td_variance_{suffix}.png")

    # Plots 3-5 and 7: signal plots from gaussian
    gaussian_signal_csvs = _csvs_for("gaussian", _SIGNAL_MODELS)
    if gaussian_signal_csvs:
        plot_latent_gap(gaussian_signal_csvs, _OUT_DIR / "latent_gap.png")
        plot_reliability(gaussian_signal_csvs, _OUT_DIR / "reliability.png")
        plot_alpha(gaussian_signal_csvs, _OUT_DIR / "alpha.png")
        plot_ablation(gaussian_signal_csvs, _OUT_DIR / "ablation.png")

    # Plot 8: R2D2 comparison
    r2d2_csvs = _csvs_for("gaussian", _R2D2_MODELS)
    if r2d2_csvs:
        plot_reward_curves(r2d2_csvs, _OUT_DIR / "r2d2_comparison.png")

    # Plot 6: non-stationary recovery
    ns_csvs: list[Path] = []
    for model in _ALL_MODELS:
        for seed in _SEEDS:
            p = _RESULTS_DIR / f"{model}_nonstationary_seed{seed}.csv"
            if p.exists():
                ns_csvs.append(p)
    if ns_csvs:
        plot_nonstationary_recovery(ns_csvs, _OUT_DIR / "nonstationary_recovery.png", switch_episode=100)
        plot_reward_curves(ns_csvs, _OUT_DIR / "nonstationary_rewards.png")

    print(f"Plots written to {_OUT_DIR}")


if __name__ == "__main__":
    generate_plots()
