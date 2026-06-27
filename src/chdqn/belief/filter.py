from __future__ import annotations
import torch
from torch import nn
from chdqn.config import ChDQNConfig, ReferenceConfig

class GRULatentFilter(nn.Module):
    def __init__(self, config: ChDQNConfig, ref: ReferenceConfig | None = None) -> None:
        super().__init__()
        self.use_reference_mode = ref is not None
        if self.use_reference_mode:
            self.recurrent = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
            self.observation = nn.Linear(config.latent_dim, config.latent_dim, bias=True)
            with torch.no_grad():
                self.recurrent.weight.copy_(torch.tensor(ref.filter_matrix_h, dtype=torch.float32))
                self.observation.weight.copy_(torch.tensor(ref.filter_matrix_z, dtype=torch.float32))
                self.observation.bias.copy_(torch.tensor(ref.filter_bias, dtype=torch.float32))
        else:
            self.gru = nn.GRUCell(config.latent_dim, config.latent_dim)
        # LayerNorm applied before tanh to prevent scale collapse.
        # Disabled in reference mode to preserve exact reference values in tests.
        self.layer_norm = (
            nn.LayerNorm(config.latent_dim)
            if config.use_layer_norm and not self.use_reference_mode
            else None
        )

    @torch.no_grad()
    def enforce_contractivity(self) -> None:
        if self.use_reference_mode:
            spectral_norm = torch.linalg.matrix_norm(self.recurrent.weight, ord=2)
            if spectral_norm > 1.0:
                self.recurrent.weight.div_(spectral_norm)

    def forward(self, prev_h: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        if self.use_reference_mode:
            self.enforce_contractivity()
            h_raw = self.recurrent(prev_h) + self.observation(z_t)
        else:
            h_raw = self.gru(z_t, prev_h)
        if self.layer_norm is not None:
            h_raw = self.layer_norm(h_raw)
        h_t = torch.tanh(h_raw)
        assert torch.isfinite(h_t).all()
        return h_t


