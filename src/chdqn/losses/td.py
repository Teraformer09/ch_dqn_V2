from __future__ import annotations
from dataclasses import dataclass
import torch
from chdqn.utils import huber
from chdqn.reliability import apply_bias_floor, compute_reliability
from chdqn.belief.consistency import consistency_loss


@dataclass(slots=True)
class LossOutput:
    td_loss: torch.Tensor
    consistency_loss: torch.Tensor
    prediction_loss: torch.Tensor
    total_loss: torch.Tensor
    td_error: torch.Tensor
    target: torch.Tensor
    latent_gap: torch.Tensor
    reliability: torch.Tensor
    c_prime: torch.Tensor       # V1: bias-floored reliability used for contraction
    h_t: torch.Tensor
    alpha_t: torch.Tensor | None = None   # V2: meta memory update rate
    gamma_t: torch.Tensor | None = None   # V2: FiLM scale
    beta_t: torch.Tensor | None = None    # V2: FiLM shift


def weighted_td_loss(
    q_pred: torch.Tensor, target: torch.Tensor, c_prime: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    td_error = q_pred - target
    # Reliability weighting: trust samples more when belief is reliable
    weighted = c_prime * huber(td_error)
    return torch.mean(weighted), td_error


def compute_losses(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    target: torch.Tensor,
    predicted_next_h: torch.Tensor,
    target_next_h: torch.Tensor,
    h_t: torch.Tensor,
    h_tilde: torch.Tensor,
    lambda_cons: float,
    lambda_pred: float,
    gamma: float = 0.99,
    reliability_scale: float = 1.0,
    precomputed_reliability: torch.Tensor | None = None,
    alpha_t: torch.Tensor | None = None,
    gamma_t: torch.Tensor | None = None,
    beta_t: torch.Tensor | None = None,
) -> LossOutput:
    chosen_q = q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

    if precomputed_reliability is not None:
        c_prime = precomputed_reliability
        raw_reliability = precomputed_reliability
    else:
        raw_reliability = compute_reliability(h_t, h_tilde, scale=reliability_scale)
        c_prime = apply_bias_floor(raw_reliability, floor=0.05)

    # Bellman contraction check: γ * max(c_prime) must be < 1
    # Note: floor + (1-floor)*c_t is always <= 1.0, so γ < 1 ensures contraction.
    assert (gamma * c_prime).max().item() < 1.0, (
        f"Bellman contraction violated: γ={gamma}, max(c_prime)={c_prime.max().item():.4f}"
    )

    td_loss, td_error = weighted_td_loss(chosen_q, target, c_prime)
    cons_loss = consistency_loss(h_t, h_tilde)
    pred_loss = torch.mean((predicted_next_h - target_next_h.detach()) ** 2)
    latent_gap = torch.norm(h_t - h_tilde, dim=-1).mean()
    total_loss = td_loss + lambda_cons * cons_loss + lambda_pred * pred_loss
    return LossOutput(
        td_loss=td_loss,
        consistency_loss=cons_loss,
        prediction_loss=pred_loss,
        total_loss=total_loss,
        td_error=td_error,
        target=target,
        latent_gap=latent_gap,
        reliability=raw_reliability.detach(),
        c_prime=c_prime.detach(),
        h_t=h_t,
        alpha_t=alpha_t.detach() if alpha_t is not None else None,
        gamma_t=gamma_t.detach() if gamma_t is not None else None,
        beta_t=beta_t.detach() if beta_t is not None else None,
    )
