from __future__ import annotations
import torch
from torch import nn
from chdqn.config import ChDQNConfig, ReferenceConfig
from chdqn.utils import clamp_observation

class LinearEncoder(nn.Module):
    def __init__(self, config: ChDQNConfig, ref: ReferenceConfig | None = None) -> None:
        super().__init__()
        self.latent_noise_std = config.latent_noise_std
        self.linear = nn.Linear(config.obs_dim, config.latent_dim)
        if ref is not None:
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(ref.encoder_matrix, dtype=torch.float32))
                self.linear.bias.copy_(torch.tensor(ref.encoder_bias, dtype=torch.float32))
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = clamp_observation(obs)
        z = self.linear(obs)
        if self.training and self.latent_noise_std > 0.0:
            z = z + torch.randn_like(z) * self.latent_noise_std
        return z


