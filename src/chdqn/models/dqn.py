from __future__ import annotations

import torch
from torch import nn


class DQNBaseline(nn.Module):
    def __init__(self, obs_dim: int = 2, hidden_dim: int = 32, num_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward_step(self, observation: torch.Tensor, hidden: any = None) -> any:
        # Consistency with sequence models
        from dataclasses import dataclass
        @dataclass
        class StepOut:
            q: torch.Tensor
            h: any = None
        return StepOut(q=self.forward(observation), h=None)
