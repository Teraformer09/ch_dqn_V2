from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import ChDQNConfig
from .replay import SequenceReplayBuffer


@dataclass(slots=True)
class RolloutStats:
    episode_reward: float
    episode_length: int
    epsilon: float
    step_data: list[dict]


class RolloutCollector:
    def __init__(self, config: ChDQNConfig) -> None:
        self.config = config
        self.global_step = 0

    def current_epsilon(self) -> float:
        decay_ratio = min(1.0, self.global_step / max(1, self.config.epsilon_decay))
        return self.config.epsilon_start + decay_ratio * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, q_values: torch.Tensor, action_dim: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(action_dim)
        return int(q_values.argmax(dim=-1).item())

    def collect_trajectory(self, env, model, epsilon: float, max_steps: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], RolloutStats]:
        observation = torch.as_tensor(env.reset(), dtype=torch.float32)
        hidden = model.init_hidden() if hasattr(model, "init_hidden") else None
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        step_data: list[dict] = []
        total_reward = 0.0

        for t in range(max_steps):
            out = model.forward_step(observation.unsqueeze(0), hidden)
            hidden = out.h if hasattr(out, "h") else out[1]
            q_values = out.q if hasattr(out, "q") else out[0]
            
            action = self.select_action(q_values, self.config.num_actions, epsilon)
            step_result = env.step(action)
            
            # Unpack step result
            if len(step_result) == 3:
                next_observation, reward, done = step_result
                info = {}
            else:
                next_observation, reward, done, info = step_result

            observations.append(observation.detach().clone())
            actions.append(torch.tensor(action, dtype=torch.long))
            rewards.append(torch.tensor(float(reward), dtype=torch.float32))
            dones.append(torch.tensor(bool(done), dtype=torch.bool))

            step_record = {
                "step": t,
                "reward": float(reward),
                "q_values": q_values.detach().cpu().numpy(),
                "noise_std": info.get("noise_std", 0.0),
            }
            # Capture V1/V2 specific metrics if available
            if hasattr(out, "c_prime"): step_record["c_t"] = out.c_prime.item()
            if hasattr(out, "alpha"): step_record["alpha_t"] = out.alpha.item()
            if hasattr(out, "gamma_film"): step_record["gamma_t"] = out.gamma_film.item()
            if hasattr(out, "latent_gap"): step_record["latent_gap"] = out.latent_gap.item()
            
            step_data.append(step_record)

            total_reward += float(reward)
            self.global_step += 1
            observation = torch.as_tensor(next_observation, dtype=torch.float32)

            if done:
                break

        stats = RolloutStats(
            episode_reward=total_reward, 
            episode_length=len(observations), 
            epsilon=epsilon,
            step_data=step_data
        )
        return observations, actions, rewards, dones, stats

    def collect_episode(self, env, model, replay: SequenceReplayBuffer, *, max_steps: int = 200) -> RolloutStats:
        epsilon = self.current_epsilon()
        observations, actions, rewards, dones, stats = self.collect_trajectory(env, model, epsilon=epsilon, max_steps=max_steps)
        if len(observations) >= replay.sequence_length:
            replay.add_trajectory(
                observations=torch.stack(observations),
                actions=torch.stack(actions),
                rewards=torch.stack(rewards),
                dones=torch.stack(dones),
            )
        return stats
