from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError("Config must be a non-empty mapping.")
    required = ["env", "noise", "episodes", "max_steps", "batch_size", "seq_len", "gamma", "learning_rate"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not (0.0 < float(data["gamma"]) <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    if int(data["seq_len"]) < 3:
        raise ValueError("seq_len must be >= 3.")
    if int(data["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0.")
    if float(data["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be > 0.")
    if data["noise"] not in {"gaussian", "uniform", "exponential", "correlated", "mixed", "delay"}:
        raise ValueError("Unsupported noise type.")
    return data


def resolve_results_dir(name: str) -> Path:
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
