from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from chdqn.config import ChDQNConfig
from chdqn.ema import EMASmoother
from chdqn.losses import LossOutput, compute_losses
from chdqn.reliability import apply_bias_floor, compute_reliability, double_dqn_target
from chdqn.models import ChDQNModel
from chdqn.optimization import SecondOrderOptimizer
from chdqn.replay import SequenceBatch, SequenceReplayBuffer, SubsequenceBatch
from chdqn.utils import build_smoothing_input, pairwise_variance, set_seed, soft_update


@dataclass(slots=True)
class TrainStepStats:
    loss: float
    td_loss: float
    consistency_loss: float
    prediction_loss: float
    td_error_mean: float
    td_error_var: float
    q_variance: float
    h_variance: float
    latent_gap: float
    reliability_mean: float
    alpha_mean: float = 0.0    # V2: mean memory update rate
    gamma_mean: float = 1.0    # V2: mean FiLM scale


class ChDQNTrainer:
    def __init__(self, config: ChDQNConfig, *, use_reference_init: bool = False) -> None:
        set_seed(config.seed)
        self.config = config
        # Initial model might be replaced by CartPoleRLRunner
        self.model = ChDQNModel(config, use_reference_init=use_reference_init)
        self.target_model = ChDQNModel(config, use_reference_init=use_reference_init)
        self.target_model.load_state_dict(self.model.state_dict())
        self.prediction = nn.Linear(config.latent_dim, config.latent_dim)
        nn.init.eye_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        
        # Components that might be missing in baselines
        self.ema_smoother = EMASmoother(self.model.smoother, tau=config.ema_tau) if hasattr(self.model, "smoother") else None
        self.step_count = 0

        # V2: persistent memory state M_t (updated once per training step)
        self.memory_state: torch.Tensor = torch.zeros(1, config.latent_dim)
        self.gamma_prev: torch.Tensor | None = None   # for gamma smoothness penalty

        self.sgd = self._init_sgd()
        self.second_order = self._init_second_order()

    def _init_sgd(self):
        params = []
        from chdqn.models import DQNBaseline
        from chdqn.models import DRQNBaseline
        from chdqn.models import R2D2Baseline
        if isinstance(self.model, (DQNBaseline, DRQNBaseline, R2D2Baseline)):
            params += list(self.model.parameters())
        else:
            if hasattr(self.model, "encoder"): params += list(self.model.encoder.parameters())
            if hasattr(self.model, "smoother"): params += list(self.model.smoother.parameters())
            if hasattr(self.model, "lstm"): params += list(self.model.lstm.parameters())
            if self.config.use_v2 and hasattr(self.model, "meta") and self.model.meta is not None:
                params += list(self.model.meta.parameters())
            if self.config.use_v2 and hasattr(self.model, "v2_input_proj") and self.model.v2_input_proj is not None:
                params += list(self.model.v2_input_proj.parameters())
        params += list(self.prediction.parameters())
        return torch.optim.Adam(params, lr=self.config.learning_rate)

    def _init_second_order(self):
        params = {}
        if hasattr(self.model, "filter"):
            params.update({f"filter.{name}": p for name, p in self.model.filter.named_parameters()})
        if hasattr(self.model, "q_head"):
            params.update({f"q_head.{name}": p for name, p in self.model.q_head.named_parameters()})
        return SecondOrderOptimizer(params, lr=self.config.learning_rate, beta1=self.config.beta1, beta2=self.config.beta2)

    def train_on_replay(self, replay: SequenceReplayBuffer, batch_size: int) -> tuple[TrainStepStats, any, dict[str, float]]:
        batch = replay.sample_subsequences(batch_size)
        return self.train_step(batch)

    def train_step(self, batch: SequenceBatch | SubsequenceBatch) -> tuple[TrainStepStats, any, dict[str, float]]:
        self.step_count += 1
        self.sgd.zero_grad(set_to_none=True)
        if hasattr(self.model, "filter"):
            for param in self.model.filter.parameters():
                if param.grad is not None: param.grad.zero_()
        if hasattr(self.model, "q_head"):
            for param in self.model.q_head.parameters():
                if param.grad is not None: param.grad.zero_()

        if isinstance(batch, SequenceBatch):
            batch = self._sequence_batch_to_subsequences(batch)

        # Vectorized processing for ChDQNModel
        if isinstance(self.model, ChDQNModel):
            obs_seq = batch.observations_seq
            time_idx = batch.time_index.long()
            
            # Forward all sequences at once
            forward = self.model.forward_sequence(obs_seq)
            target_forward = self.target_model.forward_sequence(obs_seq)
            
            # Extract values at specific time indices
            # forward.h: [batch, time, latent_dim]
            # time_idx: [batch]
            batch_indices = torch.arange(obs_seq.shape[0])
            h_t = forward.h[batch_indices, time_idx]
            next_h = forward.h[batch_indices, time_idx + 1]
            q_t = forward.q[batch_indices, time_idx]
            q_next_online = forward.q[batch_indices, time_idx + 1]
            q_next_target = target_forward.q[batch_indices, time_idx + 1]
            h_tilde = forward.h_smooth[batch_indices, time_idx]

            # V1: Reliability and contraction
            raw_reliability = compute_reliability(h_t, h_tilde.detach(), scale=self.config.reliability_scale)
            c_prime = apply_bias_floor(raw_reliability, self.config.reliability_floor)

            # V2: Meta-control modulation
            alpha_t = gamma_t = beta_t = None
            if self.config.use_v2 and self.model.meta is not None:
                g_t = 5.0 * torch.norm(h_t - h_tilde.detach(), dim=-1, keepdim=True)
                q_a_prelim = q_t.gather(-1, batch.a_t.long().unsqueeze(-1))
                # Double DQN target for prelim delta
                with torch.no_grad():
                    next_actions_pre = q_next_online.argmax(dim=-1, keepdim=True)
                    next_vals_pre = q_next_target.gather(-1, next_actions_pre).squeeze(-1)
                    target_pre = batch.r_t + self.config.gamma * (1.0 - batch.done_t.float()) * c_prime * next_vals_pre
                delta_abs = 5.0 * (q_a_prelim - target_pre.unsqueeze(-1)).abs().detach()

                _, gamma_t, beta_t = self.model.meta(delta_abs, g_t)

                # Alpha: deterministic gap-proportional function (avoids learned inversion)
                # Empirically: end-to-end learned alpha inverts under weak SGD gradient.
                # Direct mapping: large gap -> high alpha (more memory adaptation needed).
                alpha_t = (0.01 + 0.19 * torch.sigmoid(g_t - g_t.detach().mean())).clamp(
                    0.01, self.config.meta_alpha_max
                )

                # Exploration fix (gamma only — alpha now deterministic)
                if self.step_count < 200:
                    gamma_t = (gamma_t + 0.1 * torch.randn_like(gamma_t)).clamp(
                        self.config.film_gamma_min, self.config.film_gamma_max
                    )

                h_mod = gamma_t * h_t + beta_t
                
                # Update memory state per sample (with gradients for alpha_t)
                M_old = self.memory_state.detach().expand(obs_seq.shape[0], -1)
                M_next = (1.0 - alpha_t) * M_old + alpha_t * h_mod

                # Update global persistent memory state with batch average
                self.memory_state = M_next.mean(dim=0, keepdim=True).detach()

                # Final Q computation for current step using the updated memory
                h_concat = torch.cat([h_mod, M_next], dim=-1)
                h_final = self.model.v2_input_proj(h_concat)
                q_t = self.model.q_head(h_final)

            # Compute targets and losses
            target = double_dqn_target(
                rewards=batch.r_t,
                dones=batch.done_t,
                online_next_q=q_next_online,
                target_next_q=q_next_target,
                gamma=self.config.gamma,
                clip_value=self.config.td_clip,
                c_prime=c_prime,
            )
            
            predicted_next_h = self.prediction(h_t)
            loss_out = compute_losses(
                q_values=q_t,
                actions=batch.a_t,
                target=target,
                predicted_next_h=predicted_next_h,
                target_next_h=next_h,
                h_t=h_t,
                h_tilde=h_tilde,
                lambda_cons=self.config.lambda_cons,
                lambda_pred=self.config.lambda_pred,
                gamma=self.config.gamma,
                reliability_scale=self.config.reliability_scale,
                precomputed_reliability=c_prime,
                alpha_t=alpha_t,
                gamma_t=gamma_t,
                beta_t=beta_t,
            )
            total_loss = loss_out.total_loss

            # V2: explicit meta loss — direct variance-reduction signal for alpha_t/gamma_t/beta_t
            if self.config.use_v2 and alpha_t is not None:
                q_a = q_t.gather(-1, batch.a_t.long().unsqueeze(-1)).squeeze(-1)
                td_err_meta = q_a - target.detach()
                # Variance term: train gamma_t/beta_t (via meta network) to reduce TD variance
                # Alpha is now deterministic — no gradient flows through it to meta params
                meta_loss = self.config.meta_variance_weight * td_err_meta.var(unbiased=False)
                total_loss = total_loss + meta_loss

                if self.gamma_prev is not None and gamma_t.shape == self.gamma_prev.shape:
                    gamma_smooth = ((gamma_t - self.gamma_prev) ** 2).mean()
                    total_loss = total_loss + self.config.gamma_smooth_weight * gamma_smooth
                self.gamma_prev = gamma_t.detach()
        else:
            # Baselines
            return self._baseline_train_step(batch)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.sgd.step()
        self.second_order.step()
        if self.ema_smoother: self.ema_smoother.update()
        soft_update(self.target_model, self.model, self.config.target_tau)

        stats = TrainStepStats(
            loss=float(total_loss.detach()),
            td_loss=float(loss_out.td_loss.detach()),
            consistency_loss=float(loss_out.consistency_loss.detach()),
            prediction_loss=float(loss_out.prediction_loss.detach()),
            td_error_mean=float(loss_out.td_error.mean().detach()),
            td_error_var=float(loss_out.td_error.var(unbiased=False).detach()),
            q_variance=float(target.var(unbiased=False).detach()),
            h_variance=float(h_t.var(unbiased=False).detach()),
            latent_gap=float(loss_out.latent_gap.detach()),
            reliability_mean=float(loss_out.reliability.mean().detach()),
            alpha_mean=float(alpha_t.mean().detach()) if alpha_t is not None else 0.0,
            gamma_mean=float(gamma_t.mean().detach()) if gamma_t is not None else 1.0,
        )
        metrics = {
            "td_mean": stats.td_error_mean,
            "td_var": stats.td_error_var,
            "loss": stats.loss,
            "latent_gap": stats.latent_gap,
            "reliability": stats.reliability_mean,
            "alpha": stats.alpha_mean,
            "gamma_film": stats.gamma_mean,
        }
        return stats, loss_out, metrics

    def _baseline_train_step(self, batch: SubsequenceBatch) -> tuple[TrainStepStats, any, dict[str, float]]:
        # Simplified DQN/DRQN training
        o_t = batch.o_t
        a_t = batch.a_t.long()
        r_t = batch.r_t
        done_t = batch.done_t
        o_next = batch.o_next

        # Forward
        if hasattr(self.model, "init_hidden"):
            # Recurrent baseline
            hidden = self.model.init_hidden(o_t.shape[0])
            # Pass (batch, 1, dim) to recurrent forward
            q_t, _ = self.model.forward(o_t.unsqueeze(1), hidden)
            q_t = q_t.squeeze(1)
            with torch.no_grad():
                target_hidden = self.target_model.init_hidden(o_next.shape[0])
                q_next, _ = self.target_model.forward(o_next.unsqueeze(1), target_hidden)
                q_next = q_next.squeeze(1)
        else:
            # Simple MLP DQN
            q_t = self.model(o_t)
            with torch.no_grad():
                q_next = self.target_model(o_next)

        q_a = q_t.gather(-1, a_t.unsqueeze(-1)).squeeze(-1)
        max_q_next = q_next.max(dim=-1)[0]
        target = r_t + self.config.gamma * (1.0 - done_t.float()) * max_q_next
        
        td_error = target - q_a
        loss = td_error.pow(2).mean()
        
        loss.backward()
        self.sgd.step()
        soft_update(self.target_model, self.model, self.config.target_tau)
        
        metrics = {
            "td_mean": float(td_error.mean().detach()),
            "td_var": float(td_error.var().detach()),
            "loss": float(loss.detach()),
        }
        return None, None, metrics

    def _sequence_batch_to_subsequences(self, batch: SequenceBatch) -> SubsequenceBatch:
        temp_buffer = SequenceReplayBuffer(capacity=batch.observations.shape[0], sequence_length=batch.observations.shape[1])
        for idx in range(batch.observations.shape[0]):
            temp_buffer.add(
                SequenceBatch(
                    observations=batch.observations[idx],
                    actions=batch.actions[idx],
                    rewards=batch.rewards[idx],
                    dones=batch.dones[idx],
                )
            )
        return temp_buffer.sample_subsequences(batch.observations.shape[0])

    def _loss_for_subsequence(self, batch: SubsequenceBatch, idx: int, *, use_smoother_in_inference: bool = False) -> LossOutput:
        assert not use_smoother_in_inference
        observations = batch.observations_seq[idx]
        time_index = int(batch.time_index[idx].item())

        forward = self.model.forward_sequence(observations)
        target_forward = self.target_model.forward_sequence(observations)
        h_t = forward.h[time_index].unsqueeze(0)
        next_h = forward.h[time_index + 1].unsqueeze(0)
        q_t = forward.q[time_index].unsqueeze(0)
        q_next_online = forward.q[time_index + 1].unsqueeze(0)
        q_next_target = target_forward.q[time_index + 1].unsqueeze(0)

        # --- V1: compute h_tilde BEFORE target so c_prime can weight the bootstrap ---
        # Online smoother (with gradient): trained via consistency loss toward h_t.detach().
        # Asymmetric info: smoother sees only h window (no z_t), preventing identity collapse.
        u_t = build_smoothing_input(forward.h, time_index, self.config.smoothing_window).unsqueeze(0)
        h_tilde = self.model.smoother(u_t)

        # Reliability uses detached h_tilde so no gradient flows into c_prime.
        raw_reliability = compute_reliability(h_t, h_tilde.detach())
        c_prime = apply_bias_floor(raw_reliability, self.config.reliability_floor)

        # y_t = r + γ * c_t' * Q(h_{t+1})  — reliability-weighted Bellman target
        target = double_dqn_target(
            rewards=batch.r_t[idx].view(1),
            dones=batch.done_t[idx].view(1),
            online_next_q=q_next_online,
            target_next_q=q_next_target,
            gamma=self.config.gamma,
            clip_value=self.config.td_clip,
            c_prime=c_prime,
        )

        # --- V2: meta-control modulation ---
        alpha_t = gamma_t = beta_t = None
        if self.config.use_v2 and self.model.meta is not None:
            # FIX 2: amplify meta inputs so small changes in h/TD produce decisive signals
            g_t = 5.0 * torch.norm(h_t - h_tilde.detach(), dim=-1, keepdim=True)    # [1, 1]
            q_a_prelim = q_t.gather(-1, batch.a_t[idx].long().view(1, 1))
            delta_abs = 5.0 * (q_a_prelim - target.unsqueeze(-1)).abs().detach()     # [1, 1]

            alpha_t, gamma_t, beta_t = self.model.meta(delta_abs, g_t)

            # FIX 6: early meta exploration — perturb outputs to generate gradient signal
            # while the meta network is still learning what inputs mean.
            # Reduced from 2000 to 200 steps to allow for convergence within 300 episodes.
            if self.step_count < 200:
                alpha_t = (alpha_t + 0.05).clamp(0.0, self.config.meta_alpha_max)
                gamma_t = (gamma_t + 0.1 * torch.randn_like(gamma_t)).clamp(
                    self.config.film_gamma_min, self.config.film_gamma_max
                )

            # FiLM modulation: h_t^mod = γ_t * h_t + β_t
            h_mod = gamma_t * h_t + beta_t

            # Memory update: M_{t+1} = (1-α_t)*M_t + α_t*h_mod
            M_old = self.memory_state.detach()
            self.memory_state = (
                (1.0 - alpha_t.detach()) * M_old + alpha_t.detach() * h_mod.detach()
            )

            # FIX 5: cat h_mod and M_old → projection → Q head
            # (h_mod + M_old is trivial after warmup; cat preserves distinct information)
            h_concat = torch.cat([h_mod, M_old], dim=-1)    # [1, 2*latent_dim]
            h_final = self.model.v2_input_proj(h_concat)    # [1, latent_dim]
            q_t = self.model.q_head(h_final)

        predicted_next_h = self.prediction(h_t)
        lambda_cons = self.config.lambda_cons
        return compute_losses(
            q_values=q_t,
            actions=batch.a_t[idx].view(1),
            target=target,
            predicted_next_h=predicted_next_h,
            target_next_h=next_h,
            h_t=h_t,
            h_tilde=h_tilde,
            lambda_cons=lambda_cons,
            lambda_pred=self.config.lambda_pred,
            gamma=self.config.gamma,
            precomputed_reliability=c_prime,
            alpha_t=alpha_t,
            gamma_t=gamma_t,
            beta_t=beta_t,
        )

    def evaluate_sequence_variance(self, observations: torch.Tensor) -> dict[str, float]:
        forward = self.model.forward_sequence(observations)
        return {
            "latent_forward_var": float(pairwise_variance(forward.h).detach()),
            "latent_smooth_var": float(pairwise_variance(forward.h_smooth).detach()),
            "q_var": float(pairwise_variance(forward.q).detach()),
        }
