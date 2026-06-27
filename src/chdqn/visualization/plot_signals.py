from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_rewards import _load_csvs, _model_color

_SIGNAL_MODELS = {"V0", "V1", "V2"}


def _plot_signal(
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
    *,
    window: int = 10,
    dpi: int = 150,
) -> None:
    if col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    models = [m for m in sorted(df["model"].unique()) if m in _SIGNAL_MODELS]
    if not models:
        plt.close(fig)
        return

    for idx, model in enumerate(models):
        mdf = df[df["model"] == model]
        grouped = mdf.groupby("episode")[col]
        mean = grouped.mean().rolling(window, min_periods=1).mean()
        std  = grouped.std(ddof=0).rolling(window, min_periods=1).mean().fillna(0)
        # Skip models where the signal is always zero (e.g. V0 has no reliability change)
        if mean.abs().max() < 1e-9:
            continue
        eps = mean.index
        color = _model_color(model, idx)
        ax.plot(eps, mean, label=model, color=color, linewidth=1.8)
        ax.fill_between(eps, (mean - std), mean + std, alpha=0.18, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latent_gap(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Plot ||h_t - h_tilde|| over episodes for V0/V1/V2."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return
    if "model" not in df.columns:
        df["model"] = "unknown"
    _plot_signal(
        df, "latent_gap",
        ylabel="||h_t - h_tilde||",
        title="Latent Gap over Episodes (V0/V1/V2)",
        out_path=out_path,
        dpi=dpi,
    )


def plot_reliability(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Plot reliability c_t over episodes for V0/V1/V2."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return
    if "model" not in df.columns:
        df["model"] = "unknown"
    _plot_signal(
        df, "reliability",
        ylabel="Reliability c_t",
        title="Reliability (c_t) over Episodes",
        out_path=out_path,
        dpi=dpi,
    )


def plot_alpha(
    csv_paths: list[str | Path],
    out_path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Plot meta-controller alpha_t over episodes (V2 only)."""
    df = _load_csvs(csv_paths)
    if df.empty:
        return
    if "model" not in df.columns:
        df["model"] = "unknown"
    _plot_signal(
        df, "alpha",
        ylabel="alpha_t (meta weight)",
        title="Meta-Controller Alpha over Episodes (V2)",
        out_path=out_path,
        dpi=dpi,
    )
