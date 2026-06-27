from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_rewards import _load_csvs, _model_color


def plot_td_variance_curves(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    window: int = 10,
    log_scale: bool = False,
    dpi: int = 150,
) -> None:
    """Plot per-episode TD variance with +/-1 sigma CI shading, one line per model."""
    df = _load_csvs(csv_paths)
    if df.empty or "td_var" not in df.columns:
        return

    if "model" not in df.columns:
        df["model"] = "unknown"

    fig, ax = plt.subplots(figsize=(9, 5))

    models = sorted(df["model"].unique())
    for idx, model in enumerate(models):
        mdf = df[df["model"] == model]
        grouped = mdf.groupby("episode")["td_var"]
        mean = grouped.mean().rolling(window, min_periods=1).mean()
        std  = grouped.std(ddof=0).rolling(window, min_periods=1).mean().fillna(0)
        eps = mean.index
        color = _model_color(model, idx)
        ax.plot(eps, mean, label=model, color=color, linewidth=1.8)
        ax.fill_between(eps, (mean - std).clip(lower=0), mean + std, alpha=0.18, color=color)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("TD Variance", fontsize=12)
    ax.set_title("TD Variance per Episode (+/-1 sigma, rolling window)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
