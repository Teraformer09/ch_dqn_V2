"""Ablation runner for DQN/DRQN/R2D2/V0/V1/V2 variants.

This script reuses the repo's ChDQNTrainer and baseline model classes (if available)
and runs short experiments to collect efficiency, complexity, and adaptability
metrics, then computes 0-10 scores for each axis.

Run with:

# from the repo root
export PYTHONPATH=src
python -m experiments.run_ablation --output_dir=output/ablation --epochs=6 --iters=200

Notes:
- This script is designed for quick, small-scale ablations. For publication-grade
  results increase epochs/iterations and run multiple random seeds.
- If baseline model classes are not exposed exactly as imported below, adjust
  imports to the correct names (check src/chdqn/models/*.py).
"""

from __future__ import annotations

import argparse
import time
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch

from chdqn.config import ChDQNConfig
from chdqn.experiment import _build_batch, _store_batch_sequences, reference_sequences
from chdqn.replay import SequenceReplayBuffer
from chdqn.trainer import ChDQNTrainer

# Try importing baseline model classes; fall back to None if not present.
try:
    from chdqn.models.dqn import DQNBaseline  # type: ignore
except Exception:
    DQNBaseline = None  # type: ignore
try:
    from chdqn.models.drqn import DRQNBaseline  # type: ignore
except Exception:
    DRQNBaseline = None  # type: ignore
try:
    from chdqn.models.r2d2 import R2D2Baseline  # type: ignore
except Exception:
    R2D2Baseline = None  # type: ignore


def param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def score_efficiency(time_per_iter: float, baseline: float) -> float:
    # Lower time => higher score. baseline is a reference time_per_iter to normalize.
    val = baseline / time_per_iter
    # clamp and scale to 0-10
    score = max(0.0, min(10.0, 10.0 * (val / (1.5))))
    return score


def score_complexity(param_cnt: int) -> float:
    # Lower params => lower complexity. We want to invert to a 0-10 complexity value
    # where 0 = trivial, 10 = very complex. We'll set heuristics.
    if param_cnt < 1e4:
        return 3.0
    if param_cnt < 1e5:
        return 5.0
    if param_cnt < 5e5:
        return 7.0
    return 9.0


def score_adaptability(td_var: float, latent_gap: float, baseline_td_var: float, baseline_gap: float) -> float:
    # Lower td_var and lower latent_gap are better. Compare to baseline values.
    td_factor = baseline_td_var / (td_var + 1e-12)
    gap_factor = baseline_gap / (latent_gap + 1e-12)
    val = 0.6 * td_factor + 0.4 * gap_factor
    # scale
    score = max(0.0, min(10.0, 10.0 * (val / 2.0)))
    return score


def run_variant(variant_name: str, config: ChDQNConfig, epochs: int, iterations_per_epoch: int, device: str) -> Dict[str, Any]:
    print(f"Running variant: {variant_name}")
    config = ChDQNConfig(**config.__dict__)
    config.seed = config.seed or 7

    trainer = ChDQNTrainer(config, use_reference_init=True)

    # If a baseline model class was requested, replace trainer.model/target_model
    if variant_name == "DQN" and DQNBaseline is not None:
        trainer.model = DQNBaseline(config)
        trainer.target_model = DQNBaseline(config)
        trainer.target_model.load_state_dict(trainer.model.state_dict())
    if variant_name == "DRQN" and DRQNBaseline is not None:
        trainer.model = DRQNBaseline(config)
        trainer.target_model = DRQNBaseline(config)
        trainer.target_model.load_state_dict(trainer.model.state_dict())
    if variant_name == "R2D2" and R2D2Baseline is not None:
        trainer.model = R2D2Baseline(config)
        trainer.target_model = R2D2Baseline(config)
        trainer.target_model.load_state_dict(trainer.model.state_dict())

    # Configure stage-specific flags
    if variant_name == "V0":
        config.use_v2 = False
        config.lambda_cons = max(config.lambda_cons, 0.01)
    elif variant_name == "V1":
        config.use_v2 = False
        config.reliability_floor = max(config.reliability_floor, 0.2)
    elif variant_name == "V2":
        config.use_v2 = True

    # Replay buffer and reference sequences
    replay = SequenceReplayBuffer(capacity=512, sequence_length=config.sequence_length)
    clean, noisy = reference_sequences()

    iteration_times: List[float] = []
    td_vars: List[float] = []
    latent_gaps: List[float] = []

    total_iters = epochs * iterations_per_epoch
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for iteration in range(1, iterations_per_epoch + 1):
            t0 = time.time()
            # cycle through noise types similarly to run_reference_experiment
            noise_cycle = ("gaussian", "uniform", "correlated", "mixed", "exponential")
            noise_type = noise_cycle[(iteration - 1) % len(noise_cycle)]
            batch = _build_batch(clean, noisy, noise_type)
            _store_batch_sequences(replay, batch)
            stats, _, metrics = trainer.train_on_replay(replay, batch_size=min(4, len(replay)))
            t1 = time.time()
            iteration_times.append(t1 - t0)
            td_vars.append(metrics.get("td_var", 0.0))
            latent_gaps.append(metrics.get("latent_gap", 0.0))
    total_time = time.time() - start_time

    mean_time = sum(iteration_times) / len(iteration_times)
    mean_td_var = sum(td_vars) / len(td_vars)
    mean_latent_gap = sum(latent_gaps) / len(latent_gaps)

    # Parameter count (rough complexity proxy)
    params = param_count(trainer.model)

    result = {
        "variant": variant_name,
        "params": params,
        "mean_time_per_iteration": mean_time,
        "total_time": total_time,
        "mean_td_var": mean_td_var,
        "mean_latent_gap": mean_latent_gap,
    }
    print(f"Finished {variant_name}: time/iter={mean_time:.4f}s, params={params}, td_var={mean_td_var:.6f}, gap={mean_latent_gap:.6f}")
    return result


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    base_cfg = ChDQNConfig()
    base_cfg.sequence_length = 10
    base_cfg.batch_size = 32
    base_cfg.train_episodes = 50

    variants = ["DQN", "DRQN", "R2D2", "V0", "V1", "V2"]

    results: List[Dict[str, Any]] = []
    for v in variants:
        try:
            res = run_variant(v, base_cfg, epochs=args.epochs, iterations_per_epoch=args.iters, device=device)
            results.append(res)
        except Exception as e:
            print(f"Variant {v} failed: {e}")

    # Determine baseline references for scoring
    # Use the slowest time as baseline_time and worst td/latent as baseline for normalization
    times = [r["mean_time_per_iteration"] for r in results if r.get("mean_time_per_iteration")]
    t_baseline = max(times) if times else 1.0
    td_vars = [r["mean_td_var"] for r in results if r.get("mean_td_var")]
    gap_vals = [r["mean_latent_gap"] for r in results if r.get("mean_latent_gap")]
    td_baseline = max(td_vars) if td_vars else 1.0
    gap_baseline = max(gap_vals) if gap_vals else 1.0

    scored: List[Dict[str, Any]] = []
    for r in results:
        eff = score_efficiency(r["mean_time_per_iteration"], t_baseline)
        comp = score_complexity(r["params"])
        adapt = score_adaptability(r["mean_td_var"], r["mean_latent_gap"], td_baseline, gap_baseline)
        r_out = r.copy()
        r_out.update({"efficiency_score": eff, "complexity_score": comp, "adaptability_score": adapt})
        scored.append(r_out)

    # Write CSV
    csv_path = output_dir / "ablation_results.csv"
    fieldnames = [
        "variant",
        "params",
        "mean_time_per_iteration",
        "total_time",
        "mean_td_var",
        "mean_latent_gap",
        "efficiency_score",
        "complexity_score",
        "adaptability_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("Ablation complete. Results written to:", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/ablation")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    main(args)
