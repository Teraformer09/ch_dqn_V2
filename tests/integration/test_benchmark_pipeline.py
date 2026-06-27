from __future__ import annotations

from pathlib import Path

import pytest

from chdqn.evaluation import aggregate_csv_runs, evaluate_run
from chdqn.utils import BenchmarkLogger
from scripts.main_benchmark import main as run_main_benchmark
from chdqn.visualization.plot_rewards import plot_reward_curves
from chdqn.visualization.plot_td_variance import plot_td_variance_curves
from chdqn.trainer import load_runner_config, run_chdqn, run_dqn, run_drqn
from chdqn.utils import load_yaml_config, resolve_results_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "environments"


@pytest.mark.parametrize(
    "config_name",
    ["cartpole_gaussian.yaml", "cartpole_uniform.yaml", "cartpole_exponential.yaml"],
)
def test_yaml_config_loads(config_name: str):
    config = load_yaml_config(CONFIG_DIR / config_name)
    assert config["env"] == "cartpole_pomdp"
    assert config["seq_len"] >= 3


def test_invalid_yaml_is_rejected(tmp_path: Path):
    path = tmp_path / "broken.yaml"
    path.write_text("gamma: -1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml_config(path)


def test_runner_config_injects_seed():
    config = load_runner_config(CONFIG_DIR / "cartpole_gaussian.yaml", seed=42)
    assert config["seed"] == 42


def test_benchmark_logger_writes_csv(tmp_path: Path):
    path = tmp_path / "run.csv"
    with BenchmarkLogger(path) as logger:
        logger.log(episode=1, reward=1.0, td_mean=0.1, td_var=0.01, loss=0.2, model="chdqn", noise="gaussian", seed=0)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "reward" in text
    assert "chdqn" in text


def test_metrics_bundle_computes_values():
    bundle = evaluate_run([1.0, 2.0, 3.0], [0.1, 0.0, -0.1], [0.5, 0.2, 0.1])
    assert bundle.metrics.mean_reward > 0.0
    assert bundle.metrics.td_variance >= 0.0


def test_plot_functions_create_files(tmp_path: Path):
    csv1 = tmp_path / "m1.csv"
    csv2 = tmp_path / "m2.csv"
    content = "timestamp,episode,reward,td_mean,td_var,loss,model,noise,seed\n0,1,1.0,0.1,0.01,0.2,m,gaussian,0\n0,2,2.0,0.05,0.005,0.1,m,gaussian,0\n"
    csv1.write_text(content, encoding="utf-8")
    csv2.write_text(content, encoding="utf-8")
    reward_plot = plot_reward_curves([csv1, csv2], tmp_path / "reward.png")
    td_plot = plot_td_variance_curves([csv1, csv2], tmp_path / "td.png")
    assert reward_plot.exists()
    assert td_plot.exists()


def test_run_chdqn_smoke():
    output = run_chdqn(CONFIG_DIR / "cartpole_gaussian.yaml", seed=0, episodes_override=2)
    assert output.csv_path.exists()
    assert len(output.rewards) == 2


def test_run_dqn_smoke():
    output = run_dqn(CONFIG_DIR / "cartpole_gaussian.yaml", seed=0, episodes_override=2)
    assert output.csv_path.exists()
    assert len(output.rewards) == 2


def test_run_drqn_smoke():
    output = run_drqn(CONFIG_DIR / "cartpole_gaussian.yaml", seed=0, episodes_override=2)
    assert output.csv_path.exists()
    assert len(output.rewards) == 2


def test_aggregate_csv_runs_from_smoke_outputs():
    outputs = [
        run_chdqn(CONFIG_DIR / "cartpole_gaussian.yaml", seed=1, episodes_override=2),
        run_dqn(CONFIG_DIR / "cartpole_gaussian.yaml", seed=1, episodes_override=2),
    ]
    bundle = aggregate_csv_runs([output.csv_path for output in outputs])
    assert bundle.metrics.mean_reward >= 0.0


def test_resolve_results_dir_creates_directory():
    path = resolve_results_dir("pytest_smoke")
    assert path.exists()
    assert path.is_dir()


def test_main_benchmark_small_run_creates_summary():
    run_main_benchmark(episodes_override=1)
    summary_path = PROJECT_ROOT / "results" / "benchmarks" / "benchmark_summary.json"
    assert summary_path.exists()


@pytest.mark.parametrize("noise_name", ["cartpole_gaussian.yaml", "cartpole_uniform.yaml", "cartpole_exponential.yaml"])
def test_all_configs_produce_chdqn_csv(noise_name: str):
    output = run_chdqn(CONFIG_DIR / noise_name, seed=2, episodes_override=1)
    assert output.csv_path.exists()
