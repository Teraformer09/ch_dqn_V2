from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chdqn.evaluation import aggregate_csv_runs
from chdqn.visualization.plot_ablation import plot_ablation
from chdqn.visualization.plot_nonstationary import plot_nonstationary_recovery
from chdqn.visualization.plot_rewards import plot_reward_curves
from chdqn.visualization.plot_signals import plot_alpha, plot_latent_gap, plot_reliability
from chdqn.visualization.plot_td_variance import plot_td_variance_curves
from chdqn.trainer import (
    run_dqn,
    run_drqn,
    run_r2d2,
    run_v0,
    run_v1,
    run_v2,
)
from chdqn.utils import resolve_results_dir

_CONFIG_DIR = PROJECT_ROOT / "configs" / "environments"

# 5 noise configs (all cartpole_pomdp env)
_NOISE_CONFIGS = [
    "cartpole_gaussian",
    "cartpole_uniform",
    "cartpole_exponential",
    "cartpole_correlated",
    "cartpole_mixed",
]
_NONSTATIONARY_CONFIG = "cartpole_nonstationary"

_RUNNERS = {
    "DQN":  run_dqn,
    "DRQN": run_drqn,
    "R2D2": run_r2d2,
    "V0":   run_v0,
    "V1":   run_v1,
    "V2":   run_v2,
}

_SEEDS = [0, 42, 123]


def _run_config(
    config_stem: str,
    *,
    seeds: list[int],
    episodes_override: int | None = None,
) -> dict[str, list[Path]]:
    """Run all 6 models for a given config. Returns csv_paths per model."""
    config_path = _CONFIG_DIR / f"{config_stem}.yaml"
    csv_by_model: dict[str, list[Path]] = {m: [] for m in _RUNNERS}
    for model_name, runner in _RUNNERS.items():
        for seed in seeds:
            print(f"  [{config_stem}] {model_name} seed={seed} ...")
            try:
                output = runner(config_path, seed=seed, episodes_override=episodes_override)
                csv_by_model[model_name].append(output.csv_path)
            except Exception as exc:
                print(f"    WARNING: {model_name} seed={seed} failed: {exc}")
    return csv_by_model


def _all_csvs(csv_by_model: dict[str, list[Path]]) -> list[Path]:
    return [p for paths in csv_by_model.values() for p in paths]


def main(episodes_override: int | None = None) -> None:
    aggregate_summary: dict[str, dict] = {}

    # ── Noise configs: all 6 models ────────────────────────────────────────────
    noise_csv_by_config: dict[str, dict[str, list[Path]]] = {}
    for config_stem in _NOISE_CONFIGS:
        print(f"\n=== Config: {config_stem} ===")
        csv_by_model = _run_config(config_stem, seeds=_SEEDS, episodes_override=episodes_override)
        noise_csv_by_config[config_stem] = csv_by_model
        all_csvs = _all_csvs(csv_by_model)
        out_dir = resolve_results_dir(config_stem)

        # Plot 1: Reward comparison (all 6 models)
        plot_reward_curves(all_csvs, out_dir / "reward_comparison.png")
        # Plot 2: TD variance (all 6 models)
        plot_td_variance_curves(all_csvs, out_dir / "td_variance_comparison.png")

        if all_csvs:
            bundle = aggregate_csv_runs(all_csvs)
            aggregate_summary[config_stem] = {
                "mean_reward": bundle.metrics.mean_reward,
                "reward_std": bundle.metrics.reward_std,
                "mean_td": bundle.metrics.mean_td,
                "td_variance": bundle.metrics.td_variance,
            }

    # ── Latent signal plots from gaussian config (V0/V1/V2 CSVs) ──────────────
    gaussian_maps = noise_csv_by_config.get("cartpole_gaussian", {})
    signal_csvs = [
        p
        for m in ("V0", "V1", "V2")
        for p in gaussian_maps.get(m, [])
    ]
    if signal_csvs:
        gauss_dir = resolve_results_dir("cartpole_gaussian")
        plot_latent_gap(signal_csvs, gauss_dir / "latent_gap.png")       # Plot 3
        plot_reliability(signal_csvs, gauss_dir / "reliability.png")     # Plot 4
        plot_alpha(signal_csvs, gauss_dir / "alpha.png")                 # Plot 5
        plot_ablation(signal_csvs, gauss_dir / "ablation.png")           # Plot 7

    # Plot 8: R2D2 comparison — DQN/DRQN/R2D2/V2 from gaussian config
    r2d2_csvs = [
        p
        for m in ("DQN", "DRQN", "R2D2", "V2")
        for p in gaussian_maps.get(m, [])
    ]
    if r2d2_csvs:
        gauss_dir = resolve_results_dir("cartpole_gaussian")
        plot_reward_curves(r2d2_csvs, gauss_dir / "r2d2_comparison.png")

    # ── Non-stationary config ─────────────────────────────────────────────────
    print(f"\n=== Config: {_NONSTATIONARY_CONFIG} ===")
    ns_csv_by_model = _run_config(
        _NONSTATIONARY_CONFIG, seeds=_SEEDS, episodes_override=episodes_override
    )
    ns_all_csvs = _all_csvs(ns_csv_by_model)
    ns_dir = resolve_results_dir(_NONSTATIONARY_CONFIG)

    if ns_all_csvs:
        # Plot 6: Non-stationary recovery
        plot_nonstationary_recovery(ns_all_csvs, ns_dir / "nonstationary_recovery.png", switch_episode=100)
        plot_reward_curves(ns_all_csvs, ns_dir / "reward_comparison.png")
        plot_td_variance_curves(ns_all_csvs, ns_dir / "td_variance_comparison.png")
        bundle = aggregate_csv_runs(ns_all_csvs)
        aggregate_summary[_NONSTATIONARY_CONFIG] = {
            "mean_reward": bundle.metrics.mean_reward,
            "reward_std": bundle.metrics.reward_std,
            "mean_td": bundle.metrics.mean_td,
            "td_variance": bundle.metrics.td_variance,
        }

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary_path = PROJECT_ROOT / "results" / "benchmarks" / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(aggregate_summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(aggregate_summary, indent=2))


if __name__ == "__main__":
    main()
