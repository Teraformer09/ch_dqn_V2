from __future__ import annotations
import torch

def double_dqn_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    gamma: float,
    clip_value: float,
    c_prime: torch.Tensor | None = None,
) -> torch.Tensor:
    """Double DQN target.
    When c_prime is supplied (V1 mode) the bootstrap is reliability-weighted:
        y_t = r + γ * c_t' * Q(h_{t+1})
    This implements the reliability-weighted Bellman operator with bounded bias."""
    next_actions = online_next_q.argmax(dim=-1, keepdim=True)
    next_values = target_next_q.gather(-1, next_actions).squeeze(-1)
    if c_prime is not None:
        next_values = c_prime * next_values
    target = rewards + gamma * (1.0 - dones.float()) * next_values
    return target.clamp(-clip_value, clip_value)


