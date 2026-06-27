from __future__ import annotations

import torch
import torch.nn as nn
from chdqn.config import ChDQNConfig

class R2D2Baseline(nn.Module):
    def __init__(self, config: ChDQNConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChDQNConfig()
        self.obs_dim = self.config.obs_dim
        self.hidden_dim = self.config.r2d2_hidden_dim
        self.num_actions = self.config.num_actions

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.q_head = nn.Linear(self.hidden_dim, self.num_actions)

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch_size, self.hidden_dim),
            torch.zeros(1, batch_size, self.hidden_dim),
        )

    def forward(self, observations: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # observations shape: (batch, seq, obs_dim)
        z = self.encoder(observations)
        q, hidden = self.lstm(z, hidden)
        q = self.q_head(q)
        return q, hidden

    def forward_step(self, observation: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # observation shape: (batch, obs_dim)
        z = self.encoder(observation.unsqueeze(1))
        q, hidden = self.lstm(z, hidden)
        q = self.q_head(q.squeeze(1))
        return q, hidden
