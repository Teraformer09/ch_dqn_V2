from __future__ import annotations
import torch

def consistency_loss(h_t: torch.Tensor, h_tilde: torch.Tensor) -> torch.Tensor:
    """Train the smoother (h_tilde) toward the forward belief state (h_t).

    h_t is detached so gradient flows only through h_tilde (the smoother),
    preventing the forward model from collapsing into the smoother target.
    """
    return torch.mean((h_tilde - h_t.detach()) ** 2)


