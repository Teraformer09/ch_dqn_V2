from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_COLORS = {
    "DQN":  "#1f77b4",
    "DRQN": "#ff7f0e",
    "R2D2": "#2ca02c",
    "V0":   "#d62728",
    "V1":   "#9467bd",
    "V2":   "#8c564b",
}

_FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


_COL_ALIASES = {
    # CSVMetricLogger names → BenchmarkLogger canonical names
    "c_t":      "reliability",
    "alpha_t":  "alpha",
    "gamma_t":  "gamma_film",
}


def _load_csvs(csv_paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                df = df.rename(columns=_COL_ALIASES)
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _model_color(model: str, idx: int) -> str:
    return MODEL_COLORS.get(model, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def plot_reward_curves(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    window: int = 10,
    dpi: int = 150,
) -> None:
    """Plot per-episode mean reward with +/-1 sigma CI shading, one line per model."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return

    if "model" not in df.columns:
        df["model"] = "unknown"
    if "seed" not in df.columns:
        df["seed"] = 0

    fig, ax = plt.subplots(figsize=(9, 5))

    models = sorted(df["model"].unique())
    for idx, model in enumerate(models):
        mdf = df[df["model"] == model]
        grouped = mdf.groupby("episode")["reward"]
        mean = grouped.mean().rolling(window, min_periods=1).mean()
        std  = grouped.std(ddof=0).rolling(window, min_periods=1).mean().fillna(0)
        eps = mean.index
        color = _model_color(model, idx)
        ax.plot(eps, mean, label=model, color=color, linewidth=1.8)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.18, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Mean Episode Reward (+/-1 sigma, rolling window)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
