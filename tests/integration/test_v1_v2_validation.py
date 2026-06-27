import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Adjust RESULTS_DIR to be relative to this file
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Project-specific column normalization
_COL_ALIASES = {
    "c_t":       "reliability",
    "alpha_t":   "alpha",
    "gamma_t":   "gamma",
    "td_var":    "td_variance",
}

# -------------------------------
# Helpers
# -------------------------------

def load_csv(model_name, env, seed=0):
    path = os.path.join(RESULTS_DIR, f"{model_name}_{env}_seed{seed}.csv")
    if not os.path.exists(path):
        pytest.skip(f"Missing file: {path}")
    df = pd.read_csv(path)
    # Align with test expectations
    return df.rename(columns=_COL_ALIASES)


def mean_range(series, start, end):
    return series.iloc[start:end].mean()


def std(series):
    return float(np.std(series))


# -------------------------------
# 🔴 V1 TESTS
# -------------------------------

@pytest.mark.V1
def test_v1_reacts_to_noise_spike():
    """
    c_t must drop when noise spikes
    """
    df = load_csv("V1", "nonstationary", seed=0)

    c = df["reliability"]

    pre = mean_range(c, 0, 80)
    spike = mean_range(c, 100, 150)

    assert spike < pre, f"V1 failed: c_t did not drop (pre={pre}, spike={spike})"


@pytest.mark.V1
def test_v1_td_variance_reduction():
    """
    V1 must reduce TD variance vs DQN
    """
    v1 = load_csv("V1", "gaussian", seed=0)
    dqn = load_csv("DQN", "gaussian", seed=0)

    assert v1["td_variance"].mean() < dqn["td_variance"].mean()


@pytest.mark.V1
def test_v1_bias_exists():
    """
    V1 should underestimate Q (bias effect)
    """
    v1 = load_csv("V1", "exponential", seed=0)
    dqn = load_csv("DQN", "exponential", seed=0)

    assert v1["reward"].mean() < dqn["reward"].mean(), \
        "V1 bias not observed (unexpected)"


# -------------------------------
# 🔴 V2 TESTS
# -------------------------------

@pytest.mark.V2
def test_v2_alpha_adapts():
    """
    alpha must increase after noise spike
    """
    df = load_csv("V2", "nonstationary", seed=0)

    alpha = df["alpha"]

    pre = mean_range(alpha, 0, 80)
    post = mean_range(alpha, 100, 150)

    assert post > pre, f"alpha not adapting (pre={pre}, post={post})"


@pytest.mark.V2
def test_v2_gamma_suppresses_noise():
    """
    gamma should decrease during noisy regime
    """
    df = load_csv("V2", "nonstationary", seed=0)

    gamma = df["gamma"]

    pre = mean_range(gamma, 0, 80)
    post = mean_range(gamma, 100, 150)

    assert post < pre, f"gamma not suppressing noise (pre={pre}, post={post})"


@pytest.mark.V2
def test_v2_meta_not_constant():
    """
    alpha and gamma must not be constant
    """
    df = load_csv("V2", "gaussian", seed=0)

    alpha_std = std(df["alpha"])
    gamma_std = std(df["gamma"])

    assert alpha_std > 1e-3, "alpha is constant → meta dead"
    assert gamma_std > 1e-3, "gamma is constant → meta dead"


@pytest.mark.V2
def test_v2_beats_v1_under_shift():
    """
    V2 should recover faster than V1
    """
    v2 = load_csv("V2", "nonstationary", seed=0)
    v1 = load_csv("V1", "nonstationary", seed=0)

    # recovery = average reward after spike
    v2_recovery = mean_range(v2["reward"], 150, 200)
    v1_recovery = mean_range(v1["reward"], 150, 200)

    assert v2_recovery > v1_recovery, \
        f"V2 not better than V1 (V2={v2_recovery}, V1={v1_recovery})"


# -------------------------------
# 🔴 CROSS TESTS (CRITICAL)
# -------------------------------

@pytest.mark.CRITICAL
def test_model_ranking():
    """
    V2 > V1 > V0 > DQN
    """
    v2 = load_csv("V2", "gaussian", seed=0)["reward"].mean()
    v1 = load_csv("V1", "gaussian", seed=0)["reward"].mean()
    v0 = load_csv("V0", "gaussian", seed=0)["reward"].mean()
    dqn = load_csv("DQN", "gaussian", seed=0)["reward"].mean()

    assert v2 > v1 > v0 > dqn, \
        f"Ranking failed: V2={v2}, V1={v1}, V0={v0}, DQN={dqn}"


@pytest.mark.CRITICAL
def test_td_variance_hierarchy():
    """
    Var(δ): V2 < V1 < DQN
    """
    v2 = load_csv("V2", "gaussian", seed=0)["td_variance"].mean()
    v1 = load_csv("V1", "gaussian", seed=0)["td_variance"].mean()
    dqn = load_csv("DQN", "gaussian", seed=0)["td_variance"].mean()

    assert v2 < v1 < dqn, \
        f"Variance hierarchy broken: V2={v2}, V1={v1}, DQN={dqn}"


@pytest.mark.CRITICAL
def test_seed_stability():
    """
    V2 must be more stable across seeds than DQN
    """
    seeds = [0, 42, 123]

    v2_rewards = []
    dqn_rewards = []

    for s in seeds:
        v2_rewards.append(load_csv("V2", "gaussian", s)["reward"].mean())
        dqn_rewards.append(load_csv("DQN", "gaussian", s)["reward"].mean())

    assert np.std(v2_rewards) < np.std(dqn_rewards), \
        "V2 not more stable than DQN"
