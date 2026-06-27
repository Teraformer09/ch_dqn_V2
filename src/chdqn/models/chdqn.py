from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from chdqn.config import ChDQNConfig, ReferenceConfig
from chdqn.utils import build_smoothing_input, clamp_observation
from chdqn.encoder import LinearEncoder
from chdqn.belief import GRULatentFilter, TemporalSmoother


class MetaController(nn.Module):
    """V2 adaptive meta-control: f_φ(|δ_t|, g_t) → (α_t, γ_t, β_t).

    Outputs are hard-constrained:
        α_t ∈ (0, α_max)          — memory update rate
        γ_t ∈ [γ_min, γ_max]      — FiLM scale
        β_t ∈ (-β_bound, β_bound) — FiLM shift
    """

    def __init__(self, config: ChDQNConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, config.meta_hidden),
            nn.ReLU(),
            nn.Linear(config.meta_hidden, 3),
        )
        self.alpha_max = config.meta_alpha_max
        self.gamma_min = config.film_gamma_min
        self.gamma_range = config.film_gamma_max - config.film_gamma_min
        self.beta_bound = config.film_beta_bound
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(
        self, td_abs: torch.Tensor, latent_gap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([td_abs, latent_gap], dim=-1)
        raw = self.net(x)
        alpha = torch.sigmoid(raw[..., :1]) * self.alpha_max
        gamma = torch.sigmoid(raw[..., 1:2]) * self.gamma_range + self.gamma_min
        beta = torch.tanh(raw[..., 2:3]) * self.beta_bound
        return alpha, gamma, beta


class DuelingQHead(nn.Module):
    def __init__(self, config: ChDQNConfig, ref: ReferenceConfig | None = None) -> None:
        super().__init__()
        self.num_actions = config.num_actions
        self.value = nn.Linear(config.latent_dim, 1)
        self.advantage = nn.Linear(config.latent_dim, config.num_actions)
        if ref is not None:
            with torch.no_grad():
                self.value.weight.copy_(torch.tensor([[0.41, 0.0]], dtype=torch.float32))
                self.value.bias.copy_(torch.tensor([0.008], dtype=torch.float32))
                self.advantage.weight.copy_(torch.tensor([[0.75, 0.0], [-0.75, 0.0]], dtype=torch.float32))
                self.advantage.bias.copy_(torch.tensor([-0.020, 0.020], dtype=torch.float32))
        else:
            nn.init.xavier_uniform_(self.value.weight)
            nn.init.zeros_(self.value.bias)
            nn.init.xavier_uniform_(self.advantage.weight)
            nn.init.zeros_(self.advantage.bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        value = self.value(h_t)
        advantage = self.advantage(h_t)
        centered = advantage - advantage.mean(dim=-1, keepdim=True)
        return value + centered


@dataclass(slots=True)
class ForwardPass:
    z: torch.Tensor
    h: torch.Tensor
    q: torch.Tensor
    h_smooth: torch.Tensor | None = None


class ChDQNModel(nn.Module):
    def __init__(self, config: ChDQNConfig, *, use_reference_init: bool = False) -> None:
        super().__init__()
        ref = ReferenceConfig() if use_reference_init else None
        self.config = config
        self.encoder = LinearEncoder(config, ref)
        self.filter = GRULatentFilter(config, ref)
        self.smoother = TemporalSmoother(config)
        self.q_head = DuelingQHead(config, ref)
        self.meta: MetaController | None = MetaController(config) if config.use_v2 else None
        # V2 memory: cat([h_mod, M_old]) needs projection back to latent_dim for q_head
        self.v2_input_proj: nn.Linear | None = None
        if config.use_v2:
            self.v2_input_proj = nn.Linear(2 * config.latent_dim, config.latent_dim)
            nn.init.xavier_uniform_(self.v2_input_proj.weight)
            nn.init.zeros_(self.v2_input_proj.bias)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.latent_dim)

    def forward_step(self, obs: torch.Tensor, prev_h: torch.Tensor) -> ForwardPass:
        z_t = self.encoder(obs)
        h_t = self.filter(prev_h, z_t)
        q_t = self.q_head(h_t)
        return ForwardPass(z=z_t, h=h_t, q=q_t)

    def forward_sequence(self, observations: torch.Tensor) -> ForwardPass:
        # observations shape: [batch, time, obs_dim] or [time, obs_dim]
        was_unbatched = observations.ndim == 2
        if was_unbatched:
            observations = observations.unsqueeze(0)

        batch_size, seq_len, _ = observations.shape
        hidden = self.init_hidden(batch_size)

        z_list = []
        h_list = []
        q_list = []

        for t in range(seq_len):
            obs_t = observations[:, t, :]
            out = self.forward_step(obs_t, hidden)
            hidden = out.h
            z_list.append(out.z)
            h_list.append(out.h)
            q_list.append(out.q)

        z = torch.stack(z_list, dim=1)  # [batch, time, latent_dim]
        h = torch.stack(h_list, dim=1)  # [batch, time, latent_dim]
        q = torch.stack(q_list, dim=1)  # [batch, time, num_actions]

        h_smooth = self.smooth_sequence(h)

        if was_unbatched:
            return ForwardPass(
                z=z.squeeze(0),
                h=h.squeeze(0),
                q=q.squeeze(0),
                h_smooth=h_smooth.squeeze(0),
            )
        return ForwardPass(z=z, h=h, q=q, h_smooth=h_smooth)

    def smooth_sequence(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: [batch, time, latent_dim]
        batch_size, seq_len, _ = h.shape
        if seq_len < 3:
            raise ValueError("Need at least three timesteps for smoothing.")
            
        smoothed = []
        for t in range(seq_len):
            # build_smoothing_input needs to be updated to handle batches
            u_t_list = []
            for b in range(batch_size):
                u_t_list.append(build_smoothing_input(h[b], t, self.config.smoothing_window))
            u_t = torch.stack(u_t_list, dim=0) # [batch, windowed_dim]
            smoothed.append(self.smoother(u_t))
            
        return torch.stack(smoothed, dim=1) # [batch, time, latent_dim]

    def policy(self, h_t: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        q = self.q_head(h_t).clamp(-10.0, 10.0)
        return torch.softmax(q / temperature, dim=-1)
