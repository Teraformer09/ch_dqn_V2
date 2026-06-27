from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .config import ChDQNConfig
from .noise import correlated_noise, exponential_noise, gaussian_noise, mixed_noise, uniform_noise
from .reference import reference_sequences
from .replay import SequenceBatch, SequenceReplayBuffer
from .trainer import ChDQNTrainer
from .utils import set_seed


@dataclass(slots=True)
class ExperimentSummary:
    epochs: int
    iterations_per_epoch: int
    total_iterations: int
    mean_loss: float
    mean_td_mean: float
    mean_td_var: float
    final_loss: float
    final_td_mean: float
    final_td_var: float
    final_latent_gap: float


def _noise_for_iteration(noise_type: str, shape: torch.Size) -> torch.Tensor:
    if noise_type == "gaussian":
        return gaussian_noise(shape, std=0.01)
    if noise_type == "uniform":
        return uniform_noise(shape, bound=0.02)
    if noise_type == "correlated":
        return correlated_noise(shape, rho=0.8, std=0.01)
    if noise_type == "exponential":
        return exponential_noise(shape, scale=0.02)
    if noise_type == "mixed":
        return mixed_noise(shape, std=0.01)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def _build_batch(clean: torch.Tensor, noisy: torch.Tensor, noise_type: str) -> SequenceBatch:
    noise_clean = _noise_for_iteration(noise_type, clean.shape)
    noise_noisy = _noise_for_iteration(noise_type, noisy.shape)
    observations = torch.stack([clean + noise_clean, noisy + noise_noisy], dim=0)
    actions = torch.zeros((2, clean.shape[0]), dtype=torch.long)
    rewards = torch.full((2, clean.shape[0]), 0.1, dtype=torch.float32)
    dones = torch.zeros((2, clean.shape[0]), dtype=torch.bool)
    return SequenceBatch(observations=observations, actions=actions, rewards=rewards, dones=dones)


def _store_batch_sequences(replay: SequenceReplayBuffer, batch: SequenceBatch) -> None:
    for idx in range(batch.observations.shape[0]):
        replay.add(
            SequenceBatch(
                observations=batch.observations[idx],
                actions=batch.actions[idx],
                rewards=batch.rewards[idx],
                dones=batch.dones[idx],
            )
        )


def _write_iteration_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "iteration",
        "global_iteration",
        "noise_type",
        "loss",
        "td_mean",
        "td_var",
        "td_loss",
        "consistency_loss",
        "prediction_loss",
        "latent_gap",
        "q_variance",
        "h_variance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_epoch_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "mean_loss",
        "mean_td_mean",
        "mean_td_var",
        "mean_latent_gap",
        "final_loss",
        "final_td_mean",
        "final_td_var",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: ExperimentSummary) -> None:
    lines = [
        "# Example Results",
        "",
        f"- Epochs: {summary.epochs}",
        f"- Iterations per epoch: {summary.iterations_per_epoch}",
        f"- Total iterations: {summary.total_iterations}",
        f"- Mean loss: {summary.mean_loss:.6f}",
        f"- Mean td_mean: {summary.mean_td_mean:.6f}",
        f"- Mean td_var: {summary.mean_td_var:.6f}",
        f"- Final loss: {summary.final_loss:.6f}",
        f"- Final td_mean: {summary.final_td_mean:.6f}",
        f"- Final td_var: {summary.final_td_var:.6f}",
        f"- Final latent_gap: {summary.final_latent_gap:.6f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(iteration_rows: list[dict[str, float]], epoch_rows: list[dict[str, float]], output_dir: Path) -> None:
    x = [row["global_iteration"] for row in iteration_rows]
    loss = [row["loss"] for row in iteration_rows]
    td_var = [row["td_var"] for row in iteration_rows]
    latent_gap = [row["latent_gap"] for row in iteration_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(x, loss, linewidth=1.0)
    plt.title("Loss Across 20 Epochs x 500 Iterations")
    plt.xlabel("Global Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, td_var, linewidth=1.0)
    plt.title("TD Variance Across 20 Epochs x 500 Iterations")
    plt.xlabel("Global Iteration")
    plt.ylabel("TD Variance")
    plt.tight_layout()
    plt.savefig(output_dir / "td_variance_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, latent_gap, linewidth=1.0)
    plt.title("Latent Gap Across 20 Epochs x 500 Iterations")
    plt.xlabel("Global Iteration")
    plt.ylabel("Latent Gap")
    plt.tight_layout()
    plt.savefig(output_dir / "latent_gap_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    epochs = [row["epoch"] for row in epoch_rows]
    epoch_mean_loss = [row["mean_loss"] for row in epoch_rows]
    epoch_mean_td_var = [row["mean_td_var"] for row in epoch_rows]
    plt.plot(epochs, epoch_mean_loss, label="Mean Loss", linewidth=2.0)
    plt.plot(epochs, epoch_mean_td_var, label="Mean TD Variance", linewidth=2.0)
    plt.title("Epoch-Level Summary")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "epoch_summary.png", dpi=150)
    plt.close()


def run_reference_experiment(
    *,
    output_dir: str | Path,
    epochs: int = 20,
    iterations_per_epoch: int = 500,
    seed: int = 7,
) -> ExperimentSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    config = ChDQNConfig(seed=seed)
    trainer = ChDQNTrainer(config, use_reference_init=True)
    replay = SequenceReplayBuffer(capacity=256, sequence_length=config.sequence_length)
    clean, noisy = reference_sequences()

    iteration_rows: list[dict[str, float]] = []
    epoch_rows: list[dict[str, float]] = []
    noise_cycle = ("gaussian", "uniform", "correlated", "mixed", "exponential")

    for epoch in range(1, epochs + 1):
        epoch_metrics = []
        for iteration in range(1, iterations_per_epoch + 1):
            noise_type = noise_cycle[(iteration - 1) % len(noise_cycle)]
            batch = _build_batch(clean, noisy, noise_type)
            _store_batch_sequences(replay, batch)
            stats, _, metrics = trainer.train_on_replay(replay, batch_size=min(2, len(replay)))

            row = {
                "epoch": epoch,
                "iteration": iteration,
                "global_iteration": (epoch - 1) * iterations_per_epoch + iteration,
                "noise_type": noise_type,
                "loss": stats.loss,
                "td_mean": metrics["td_mean"],
                "td_var": metrics["td_var"],
                "td_loss": stats.td_loss,
                "consistency_loss": stats.consistency_loss,
                "prediction_loss": stats.prediction_loss,
                "latent_gap": stats.latent_gap,
                "q_variance": stats.q_variance,
                "h_variance": stats.h_variance,
            }
            iteration_rows.append(row)
            epoch_metrics.append(row)

        epoch_rows.append(
            {
                "epoch": epoch,
                "mean_loss": sum(item["loss"] for item in epoch_metrics) / len(epoch_metrics),
                "mean_td_mean": sum(item["td_mean"] for item in epoch_metrics) / len(epoch_metrics),
                "mean_td_var": sum(item["td_var"] for item in epoch_metrics) / len(epoch_metrics),
                "mean_latent_gap": sum(item["latent_gap"] for item in epoch_metrics) / len(epoch_metrics),
                "final_loss": epoch_metrics[-1]["loss"],
                "final_td_mean": epoch_metrics[-1]["td_mean"],
                "final_td_var": epoch_metrics[-1]["td_var"],
            }
        )

    summary = ExperimentSummary(
        epochs=epochs,
        iterations_per_epoch=iterations_per_epoch,
        total_iterations=epochs * iterations_per_epoch,
        mean_loss=sum(item["loss"] for item in iteration_rows) / len(iteration_rows),
        mean_td_mean=sum(item["td_mean"] for item in iteration_rows) / len(iteration_rows),
        mean_td_var=sum(item["td_var"] for item in iteration_rows) / len(iteration_rows),
        final_loss=iteration_rows[-1]["loss"],
        final_td_mean=iteration_rows[-1]["td_mean"],
        final_td_var=iteration_rows[-1]["td_var"],
        final_latent_gap=iteration_rows[-1]["latent_gap"],
    )

    _write_iteration_csv(output_path / "iteration_results_20epochs_500iters.csv", iteration_rows)
    _write_epoch_csv(output_path / "epoch_results_20epochs.csv", epoch_rows)
    _write_summary(output_path / "EXAMPLE_RESULTS.md", summary)
    _make_plots(iteration_rows, epoch_rows, output_path)
    return summary
