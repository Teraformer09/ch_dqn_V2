from __future__ import annotations

import torch
from torch import nn


class DRQNBaseline(nn.Module):
    def __init__(self, obs_dim: int = 2, hidden_dim: int = 32, num_actions: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (batch, seq, dim)
        z = torch.relu(self.encoder(obs))
        q, hidden = self.gru(z, hidden)
        q = self.head(q)
        return q, hidden

    def forward_step(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (batch, dim)
        z = torch.relu(self.encoder(obs))
        # GRU expects (seq, batch, dim) or batch_first (batch, seq, dim)
        q, hidden = self.gru(z.unsqueeze(1), hidden)
        q = self.head(q.squeeze(1))
        return q, hidden
