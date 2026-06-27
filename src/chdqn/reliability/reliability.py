from __future__ import annotations
import torch

def apply_bias_floor(reliability: torch.Tensor, floor: float) -> torch.Tensor:
    """V1 fix: c_t' = ε + (1-ε)*c_t.
    Ensures c_prime ≥ ε even when h_t and h_tilde diverge, bounding
    systematic underestimation bias while preserving contraction."""
    return floor + (1.0 - floor) * reliability



def compute_reliability(h: torch.Tensor, h_tilde: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    diff = torch.norm(h - h_tilde, dim=-1)
    # scale modulates sensitivity: higher scale -> c_t drops faster with diff
    return torch.exp(-scale * (diff ** 2)).detach()


