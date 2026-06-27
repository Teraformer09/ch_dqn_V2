from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path


class BenchmarkLogger:
    """Episode-level benchmark logger with full V0/V1/V2 signal schema."""

    FIELDS = [
        "timestamp", "episode", "reward",
        "td_mean", "td_var", "loss",
        "latent_gap", "reliability", "alpha", "gamma_film",
        "model", "noise", "seed",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def log(
        self,
        *,
        episode: int,
        reward: float,
        td_mean: float,
        td_var: float,
        loss: float,
        latent_gap: float = 0.0,
        reliability: float = 1.0,
        alpha: float = 0.0,
        gamma_film: float = 1.0,
        model: str,
        noise: str,
        seed: int,
    ) -> None:
        self.writer.writerow({
            "timestamp": datetime.now(UTC).isoformat(),
            "episode": episode,
            "reward": reward,
            "td_mean": td_mean,
            "td_var": td_var,
            "loss": loss,
            "latent_gap": latent_gap,
            "reliability": reliability,
            "alpha": alpha,
            "gamma_film": gamma_film,
            "model": model,
            "noise": noise,
            "seed": seed,
        })
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "BenchmarkLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
