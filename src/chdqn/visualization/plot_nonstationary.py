from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_rewards import _load_csvs, _model_color


def plot_nonstationary_recovery(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    switch_episode: int = 100,
    window: int = 10,
    dpi: int = 150,
) -> None:
    """Reward curves with vertical marker at noise-switch episode."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return

    if "model" not in df.columns:
        df["model"] = "unknown"

    fig, ax = plt.subplots(figsize=(10, 5))

    models = sorted(df["model"].unique())
    for idx, model in enumerate(models):
        mdf = df[df["model"] == model]
        grouped = mdf.groupby("episode")["reward"]
        mean = grouped.mean().rolling(window, min_periods=1).mean()
        std  = grouped.std(ddof=0).rolling(window, min_periods=1).mean().fillna(0)
        eps = mean.index
        color = _model_color(model, idx)
        ax.plot(eps, mean, label=model, color=color, linewidth=1.8)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.15, color=color)

    ymin, ymax = ax.get_ylim()
    ax.axvline(x=switch_episode, color="crimson", linestyle="--", linewidth=1.5,
               label=f"noise shift (ep {switch_episode})")
    ax.annotate(
        "sigma: 0.01 -> 0.2",
        xy=(switch_episode, ymax * 0.95),
        xytext=(switch_episode + 5, ymax * 0.95),
        fontsize=9,
        color="crimson",
    )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Non-Stationary Recovery (noise shift at episode 100)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
