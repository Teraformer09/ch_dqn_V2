from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_rewards import _load_csvs, _model_color

_ABLATION_MODELS = ["V0", "V1", "V2"]
_LABELS = {
    "V0": "V0 — smoother only (c=1)",
    "V1": "V1 — + reliability weighting",
    "V2": "V2 — + meta controller",
}


def plot_ablation(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    window: int = 10,
    dpi: int = 150,
) -> None:
    """V0/V1/V2 ablation: reward curves with CI shading on one figure."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return

    if "model" not in df.columns:
        df["model"] = "unknown"

    fig, ax = plt.subplots(figsize=(9, 5))

    present = [m for m in _ABLATION_MODELS if m in df["model"].unique()]
    for idx, model in enumerate(present):
        mdf = df[df["model"] == model]
        grouped = mdf.groupby("episode")["reward"]
        mean = grouped.mean().rolling(window, min_periods=1).mean()
        std  = grouped.std(ddof=0).rolling(window, min_periods=1).mean().fillna(0)
        eps = mean.index
        color = _model_color(model, idx)
        label = _LABELS.get(model, model)
        ax.plot(eps, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.18, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Ablation: V0 vs V1 vs V2", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
