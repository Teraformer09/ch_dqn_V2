from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chdqn.experiment import run_reference_experiment


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    summary = run_reference_experiment(output_dir=results_dir, epochs=20, iterations_per_epoch=500, seed=7)
    print("Reference experiment completed.")
    print(f"Total iterations: {summary.total_iterations}")
    print(f"Final loss: {summary.final_loss:.6f}")
    print(f"Final td_mean: {summary.final_td_mean:.6f}")
    print(f"Final td_var: {summary.final_td_var:.6f}")
    print(f"Final latent_gap: {summary.final_latent_gap:.6f}")


if __name__ == "__main__":
    main()
