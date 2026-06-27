from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from chdqn.evaluation.metrics import BenchmarkMetrics, build_metrics


@dataclass(slots=True)
class EvaluationBundle:
    metrics: BenchmarkMetrics
    rewards: list[float]
    td_errors: list[float]
    losses: list[float]


def evaluate_run(rewards: list[float], td_errors: list[float], losses: list[float]) -> EvaluationBundle:
    return EvaluationBundle(metrics=build_metrics(rewards, td_errors, losses), rewards=rewards, td_errors=td_errors, losses=losses)


def aggregate_csv_runs(csv_paths: list[str | Path]) -> EvaluationBundle:
    rewards: list[float] = []
    td_errors: list[float] = []
    losses: list[float] = []
    for path in csv_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rewards.append(float(row["reward"]))
                td_errors.append(float(row["td_mean"]))
                losses.append(float(row["loss"]))
    return evaluate_run(rewards, td_errors, losses)
