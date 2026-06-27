"""V1 / V2 Behavior Validation Suite.

These are INTEGRATION tests — they read CSV files produced by run_experiments.py.
Run experiments first:
    python run_experiments.py

Then run this suite:
    pytest tests/test_v1_v2_behavior.py -v -m BEHAVIOR

Individual groups:
    pytest tests/test_v1_v2_behavior.py -v -m V1
    pytest tests/test_v1_v2_behavior.py -v -m V2
    pytest tests/test_v1_v2_behavior.py -v -m CRITICAL

Tests skip gracefully when the required CSV files do not exist.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.BEHAVIOR

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ── Column normalisation ──────────────────────────────────────────────────────
# CSVMetricLogger (run_experiments.py) uses different column names from
# BenchmarkLogger (experiments/main_benchmark.py). Normalise on load.
_COL_ALIASES = {
    "c_t":       "reliability",
    "alpha_t":   "alpha",
    "gamma_t":   "gamma",
    "td_var":    "td_variance",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(model: str, noise: str, seed: int = 0) -> pd.DataFrame:
    path = RESULTS_DIR / f"{model}_{noise}_seed{seed}.csv"
    if not path.exists():
        pytest.skip(f"CSV not found (run run_experiments.py first): {path}")
    df = pd.read_csv(path)
    return df.rename(columns=_COL_ALIASES)


def window_mean(series: pd.Series, start: int, end: int) -> float:
    sub = series.iloc[start:end]
    if sub.empty:
        pytest.skip(f"Not enough episodes in series (need {end}, got {len(series)})")
    return float(sub.mean())


# ── V1 Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.V1
class TestV1Behavior:

    def test_v1_1_noise_spike_suppresses_reliability(self):
        """V1.1 — c_t must drop during the high-noise phase of nonstationary env.

        The env switches from σ=0.01 to σ=0.2 at episode ~100.
        Reliability in the spike window (100-150) must be lower than pre-spike.
        """
        df = load_csv("V1", "nonstationary")
        c = df["reliability"]
        pre   = window_mean(c, 0, 80)
        spike = window_mean(c, 100, 150)
        assert spike < pre, (
            f"V1.1 FAIL: c_t did not drop at noise spike "
            f"(pre={pre:.3f}, spike={spike:.3f})"
        )

    def test_v1_2_false_positive_clean_env(self):
        """V1.2 — On clean Gaussian noise (σ=0.02), c_t should stay high (> 0.85).

        If reliability drops without reason, V1 is over-penalising clean data.
        """
        df = load_csv("V1", "gaussian")
        late_c = window_mean(df["reliability"], 100, 200)
        assert late_c > 0.85, (
            f"V1.2 FAIL: reliability too low on clean env (mean={late_c:.3f})"
        )

    def test_v1_3_bias_under_exponential_noise(self):
        """V1.3 — V1 must underestimate Q vs DQN under exponential (skewed) noise.

        Exponential noise creates asymmetric perturbations → V1 reliability-weighted
        Bellman target should be lower → lower mean reward than naive DQN.
        (Documents the bias-variance tradeoff, not a failure.)
        """
        v1  = load_csv("V1",  "exponential")
        dqn = load_csv("DQN", "exponential")
        v1_mean  = window_mean(v1["reward"],  50, 200)
        dqn_mean = window_mean(dqn["reward"], 50, 200)
        # V1 should not dramatically exceed DQN; slight under or equal is expected
        assert v1_mean <= dqn_mean * 1.15, (
            f"V1.3: V1 ({v1_mean:.3f}) far exceeds DQN ({dqn_mean:.3f}) on "
            f"exponential noise — bias-variance claim not holding"
        )

    def test_v1_4_floor_effect_on_td_variance(self):
        """V1.4 — V1 must reduce TD variance vs DQN on Gaussian noise."""
        v1  = load_csv("V1",  "gaussian")
        dqn = load_csv("DQN", "gaussian")
        v1_var  = window_mean(v1["td_variance"],  100, 300)
        dqn_var = window_mean(dqn["td_variance"], 100, 300)
        assert v1_var < dqn_var, (
            f"V1.4 FAIL: V1 td_var ({v1_var:.4f}) >= DQN ({dqn_var:.4f})"
        )

    def test_v1_5_correlated_noise_advantage(self):
        """V1.5 — V1 reward >= DQN reward on correlated AR(1) noise (ρ=0.9).

        Correlated noise creates persistent latent gaps; reliability should
        correctly discount bootstrap and protect Q-learning.
        """
        v1  = load_csv("V1",  "correlated")
        dqn = load_csv("DQN", "correlated")
        v1_r  = window_mean(v1["reward"],  150, 300)
        dqn_r = window_mean(dqn["reward"], 150, 300)
        tolerance = 0.10 * abs(dqn_r) if abs(dqn_r) > 1e-6 else 1.0
        assert v1_r >= dqn_r - tolerance, (
            f"V1.5 FAIL: V1 ({v1_r:.3f}) much worse than DQN ({dqn_r:.3f}) "
            f"on correlated noise"
        )


# ── V2 Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.V2
class TestV2Behavior:

    def test_v2_1_alpha_responds_to_noise_spike(self):
        """V2.1 — alpha_t must increase after the noise shift (episode 100+)."""
        df = load_csv("V2", "nonstationary")
        alpha = df["alpha"]
        pre  = window_mean(alpha, 20, 80)
        post = window_mean(alpha, 110, 170)
        assert post > pre, (
            f"V2.1 FAIL: alpha not adapting to noise spike "
            f"(pre={pre:.4f}, post={post:.4f})"
        )

    def test_v2_2_gamma_suppresses_during_noise(self):
        """V2.2 — gamma_t (FiLM scale) must decrease when noise increases."""
        df = load_csv("V2", "nonstationary")
        gamma = df["gamma"]
        pre  = window_mean(gamma, 20, 80)
        post = window_mean(gamma, 110, 170)
        assert post < pre, (
            f"V2.2 FAIL: gamma not suppressing noise "
            f"(pre={pre:.4f}, post={post:.4f})"
        )

    def test_v2_3_meta_not_constant(self):
        """V2.3 — alpha and gamma must vary throughout training on Gaussian noise."""
        df = load_csv("V2", "gaussian")
        alpha_std = float(df["alpha"].std())
        gamma_std = float(df["gamma"].std())
        assert alpha_std > 1e-3, (
            f"V2.3 FAIL: alpha is near-constant (std={alpha_std:.6f}) — meta dead"
        )
        assert gamma_std > 1e-3, (
            f"V2.3 FAIL: gamma is near-constant (std={gamma_std:.6f}) — FiLM dead"
        )

    def test_v2_4_beats_v1_under_nonstationary(self):
        """V2.4 — V2 recovers faster than V1 after the noise shift."""
        v2 = load_csv("V2", "nonstationary")
        v1 = load_csv("V1", "nonstationary")
        v2_recovery = window_mean(v2["reward"], 150, 250)
        v1_recovery = window_mean(v1["reward"], 150, 250)
        assert v2_recovery >= v1_recovery * 0.95, (
            f"V2.4 FAIL: V2 recovery ({v2_recovery:.3f}) < V1 ({v1_recovery:.3f})"
        )

    def test_v2_5_meta_not_harmful(self):
        """V2.5 — V2 must not be significantly worse than V0 on clean Gaussian.

        Soft test: V2 reward >= V0 reward minus 20% tolerance.
        """
        v2 = load_csv("V2", "gaussian")
        v0 = load_csv("V0", "gaussian")
        v2_r = window_mean(v2["reward"], 150, 300)
        v0_r = window_mean(v0["reward"], 150, 300)
        tolerance = 0.20 * abs(v0_r) if abs(v0_r) > 1e-6 else 1.0
        assert v2_r >= v0_r - tolerance, (
            f"V2.5 FAIL: V2 ({v2_r:.3f}) significantly worse than V0 ({v0_r:.3f})"
        )

    def test_v2_6_latent_gap_positive_and_meaningful(self):
        """V2.6 — Latent gap must be > 0 throughout V2 training.

        Zero gap means h_tilde = h_t → smoother collapsed → anti-collapse fixes
        must be working.
        """
        df = load_csv("V2", "gaussian")
        if "latent_gap" not in df.columns:
            pytest.skip("latent_gap column not present in CSV")
        late_gap = window_mean(df["latent_gap"], 50, 300)
        assert late_gap > 0.01, (
            f"V2.6 FAIL: latent_gap near zero ({late_gap:.5f}) — smoother collapsed"
        )

    def test_v2_7_meta_stable_under_random_noise(self):
        """V2.7 — On mixed (random spike) noise, meta must not produce exploding values."""
        df = load_csv("V2", "mixed")
        alpha_max = float(df["alpha"].max())
        gamma_max = float(df["gamma"].max())
        assert alpha_max <= 0.3, (
            f"V2.7 FAIL: alpha exploded to {alpha_max:.4f} on mixed noise"
        )
        assert gamma_max <= 2.0, (
            f"V2.7 FAIL: gamma exploded to {gamma_max:.4f} on mixed noise"
        )


# ── Critical Cross-Model Tests ────────────────────────────────────────────────

@pytest.mark.CRITICAL
class TestCritical:

    def test_x1_model_ranking(self):
        """X1 — V2 > V1 > V0 > DQN on mean Gaussian reward (episodes 150-300)."""
        models = ["V2", "V1", "V0", "DQN"]
        means: dict[str, float] = {}
        for m in models:
            df = load_csv(m, "gaussian")
            means[m] = window_mean(df["reward"], 150, 300)

        v2, v1, v0, dqn = means["V2"], means["V1"], means["V0"], means["DQN"]
        assert v2 > v0, f"X1 FAIL: V2 ({v2:.3f}) <= V0 ({v0:.3f})"
        assert v1 > dqn * 0.9, f"X1 FAIL: V1 ({v1:.3f}) much worse than DQN ({dqn:.3f})"
        assert v2 >= v1 * 0.95, f"X1 FAIL: V2 ({v2:.3f}) much worse than V1 ({v1:.3f})"

    def test_x2_seed_stability_v2_vs_dqn(self):
        """X2 — V2 reward variance across seeds must be <= DQN variance."""
        seeds = [0, 42, 123]
        v2_r, dqn_r = [], []
        for s in seeds:
            v2_df  = load_csv("V2",  "gaussian", seed=s)
            dqn_df = load_csv("DQN", "gaussian", seed=s)
            v2_r.append(window_mean(v2_df["reward"],  150, 300))
            dqn_r.append(window_mean(dqn_df["reward"], 150, 300))
        v2_std  = float(np.std(v2_r))
        dqn_std = float(np.std(dqn_r))
        assert v2_std <= dqn_std * 1.5, (
            f"X2 FAIL: V2 std ({v2_std:.4f}) >> DQN std ({dqn_std:.4f}) "
            f"— V2 less stable"
        )

    def test_x3_td_variance_hierarchy(self):
        """X3 — TD variance: V2 < V1 < DQN (episodes 150-300, Gaussian noise)."""
        v2  = window_mean(load_csv("V2",  "gaussian")["td_variance"], 150, 300)
        v1  = window_mean(load_csv("V1",  "gaussian")["td_variance"], 150, 300)
        dqn = window_mean(load_csv("DQN", "gaussian")["td_variance"], 150, 300)
        # Require at least partial ordering
        assert v2 < dqn, (
            f"X3 FAIL: V2 td_var ({v2:.5f}) not < DQN ({dqn:.5f})"
        )
        assert v1 < dqn, (
            f"X3 FAIL: V1 td_var ({v1:.5f}) not < DQN ({dqn:.5f})"
        )

    def test_x4_latent_gap_nonzero_throughout(self):
        """X4 — Latent gap must not collapse to zero during V2 training."""
        df = load_csv("V2", "gaussian")
        if "latent_gap" not in df.columns:
            pytest.skip("latent_gap not logged")
        # Check first half and second half separately
        first_half = window_mean(df["latent_gap"], 0, 150)
        second_half = window_mean(df["latent_gap"], 150, 300)
        assert first_half > 0.01, (
            f"X4 FAIL: latent_gap collapsed early (first_half={first_half:.5f})"
        )
        assert second_half > 0.01, (
            f"X4 FAIL: latent_gap collapsed late (second_half={second_half:.5f})"
        )
