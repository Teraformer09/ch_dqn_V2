from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ReferenceConfig


@dataclass(slots=True)
class ReferenceDryRun:
    observation: torch.Tensor
    encoded: torch.Tensor
    latent: torch.Tensor
    q_values: torch.Tensor


def get_reference_config() -> ReferenceConfig:
    return ReferenceConfig()


def get_reference_dry_run() -> ReferenceDryRun:
    return ReferenceDryRun(
        observation=torch.tensor([0.2, 0.05], dtype=torch.float32),
        encoded=torch.tensor([0.085, 0.055], dtype=torch.float32),
        latent=torch.tensor([0.1114, -0.1596], dtype=torch.float32),
        q_values=torch.tensor([0.046, 0.011], dtype=torch.float32),
    )


def reference_epoch_curve() -> dict[str, torch.Tensor]:
    config = get_reference_config()
    return {
        "q": torch.tensor(config.epoch_q0, dtype=torch.float32),
        "td": torch.tensor(config.epoch_td, dtype=torch.float32),
    }


def reference_sequences() -> tuple[torch.Tensor, torch.Tensor]:
    config = get_reference_config()
    clean = torch.tensor(config.dry_run_latents, dtype=torch.float32)
    noisy = torch.tensor(config.noisy_sequence, dtype=torch.float32)
    return clean, noisy
