from __future__ import annotations
import torch
from torch import nn
from chdqn.config import ChDQNConfig
from chdqn.utils import build_smoothing_input

class TemporalSmoother(nn.Module):
    def __init__(self, config: ChDQNConfig) -> None:
        super().__init__()
        self.window = config.smoothing_window
        # h-only window: z_t removed so smoother cannot cheat by reconstructing
        # the forward state from the current observation.
        self.expected_input_dim = (2 * self.window + 1) * config.latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.expected_input_dim, config.hidden_width),
            nn.ReLU(),
            nn.Linear(config.hidden_width, config.latent_dim),
        )

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.expected_input_dim:
            return x
        if x.shape[-1] > self.expected_input_dim:
            raise ValueError("Smoother input larger than expected windowed input dimension.")
        padding = torch.zeros(*x.shape[:-1], self.expected_input_dim - x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=-1)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if len(args) == 1:
            x = args[0]
        elif len(args) == 2:
            # Convenience: smoother(h_prev, h_next) for direct pairwise calls
            x = torch.cat(args, dim=-1)
        else:
            raise ValueError("TemporalSmoother expects one window tensor or two h tensors (prev, next).")
        x = self._pad_input(x)
        return self.net(x)


