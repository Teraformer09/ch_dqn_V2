"""Top-level script: run all 6 models across 5 noise types + non-stationary, 3 seeds each."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from chdqn.config import ChDQNConfig
from chdqn.trainer_rl import CartPoleRLRunner

_MODELS = ["DQN", "DRQN", "R2D2", "V0", "V1", "V2"]
_NOISE_TYPES = ["gaussian", "uniform", "exponential", "correlated", "mixed"]
_SEEDS = [0, 42, 123]

_BASE_CONFIG = ChDQNConfig(
    train_episodes=300,
    max_steps_per_episode=200,
    batch_size=32,
    gamma=0.99,
    learning_rate=1e-3,
    epsilon_decay=10_000,
    min_replay_sequences=100,
    is_non_stationary=False,
)


def run_all_experiments() -> None:
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    import dataclasses

    # Stationary sweeps
    for noise in _NOISE_TYPES:
        for model_name in _MODELS:
            for seed in _SEEDS:
                print(f"Running {model_name}  noise={noise}  seed={seed} ...")
                cfg = dataclasses.replace(_BASE_CONFIG, seed=seed, is_non_stationary=False)
                runner = CartPoleRLRunner(cfg, model_type=model_name, noise_type=noise)
                log_path = out_dir / f"{model_name}_{noise}_seed{seed}.csv"
                runner.train(log_path=str(log_path), seed=seed)

    # Non-stationary sweep
    for model_name in _MODELS:
        for seed in _SEEDS:
            print(f"Running {model_name}  noise=nonstationary  seed={seed} ...")
            cfg = dataclasses.replace(_BASE_CONFIG, seed=seed, is_non_stationary=True)
            runner = CartPoleRLRunner(cfg, model_type=model_name, noise_type="gaussian")
            log_path = out_dir / f"{model_name}_nonstationary_seed{seed}.csv"
            runner.train(log_path=str(log_path), seed=seed)

    print("\nDone. CSVs written to", out_dir)


if __name__ == "__main__":
    run_all_experiments()
