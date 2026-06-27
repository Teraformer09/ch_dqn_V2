from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class LogRecord:
    episode: int
    step: int
    reward: float
    td_mean: float
    td_var: float
    loss: float
    noise_std: float = 0.0
    latent_gap: float = 0.0
    alpha_t: float = 0.0
    gamma_t: float = 1.0
    c_t: float = 1.0
    model: str = ""
    seed: int = 0


class CSVMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self.fieldnames = [
            "timestamp", "episode", "step", "reward", "td_mean", "td_var",
            "loss", "noise_std", "latent_gap", "alpha_t", "gamma_t", "c_t",
            "model", "seed",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, record: LogRecord) -> None:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "episode": record.episode,
            "step": record.step,
            "reward": record.reward,
            "td_mean": record.td_mean,
            "td_var": record.td_var,
            "loss": record.loss,
            "noise_std": record.noise_std,
            "latent_gap": record.latent_gap,
            "alpha_t": record.alpha_t,
            "gamma_t": record.gamma_t,
            "c_t": record.c_t,
            "model": record.model,
            "seed": record.seed,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVMetricLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
