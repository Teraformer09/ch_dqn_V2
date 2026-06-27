"""Run only the CSV files needed for behavior tests in parallel with thread optimization."""
from __future__ import annotations
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import torch here to set threads before anything else
import torch
torch.set_num_threads(1)

from chdqn.config import ChDQNConfig
from chdqn.trainer_rl import CartPoleRLRunner
import dataclasses

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

# Prioritized RUNS
RUNS = [
    # (model, noise, seed, is_nonstationary)
    ("V2",  "nonstationary", 0,   True),
    ("V1",  "nonstationary", 0,   True),
    ("V2",  "gaussian",      0,   False),
    ("V1",  "gaussian",      0,   False),
    ("V0",  "gaussian",      0,   False),
    ("DQN", "gaussian",      0,   False), 
    
    # Rest of the seeds for ranking and stability
    ("V2",  "gaussian",      42,  False),
    ("V2",  "gaussian",      123, False),
    ("DQN", "gaussian",      42,  False),
    ("DQN", "gaussian",      123, False),
    ("V1",  "gaussian",      42,  False),
    ("V1",  "gaussian",      123, False),
    
    # Other noise types
    ("V1",  "exponential",   0,   False),
    ("V1",  "correlated",    0,   False),
    ("V2",  "mixed",         0,   False),
]

def run_single(args):
    # Set thread limit in child process too
    import torch
    torch.set_num_threads(1)
    
    model_name, noise, seed, nonstat = args
    noise_type = "gaussian" if nonstat else noise
    suffix = "nonstationary" if nonstat else noise
    out_dir = PROJECT_ROOT / "results"
    out_path = out_dir / f"{model_name}_{suffix}_seed{seed}.csv"
    
    # Skip if file already looks complete (at least 301 lines)
    if out_path.exists():
        with open(out_path, 'r') as f:
            lines = sum(1 for _ in f)
        if lines >= 301:
            print(f"  SKIP {out_path.name} (complete)")
            return
        else:
            print(f"  Removing incomplete {out_path.name}")
            out_path.unlink()
        
    print(f"Starting {model_name}  noise={suffix}  seed={seed} ...", flush=True)
    cfg = dataclasses.replace(_BASE_CONFIG, seed=seed, is_non_stationary=nonstat)
    runner = CartPoleRLRunner(cfg, model_type=model_name, noise_type=noise_type)
    runner.train(log_path=str(out_path), seed=seed)
    print(f"Finished {model_name}  noise={suffix}  seed={seed}.", flush=True)

def main():
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    
    # Use 6 workers.
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(run_single, RUNS)

    print("\nAll targeted runs complete.")

if __name__ == "__main__":
    main()
