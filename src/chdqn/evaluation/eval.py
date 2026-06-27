from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EvaluationStats:
    mean_reward: float
    std_reward: float
    mean_episode_length: float


def evaluate_model(model, env_factory, episodes: int = 20, max_steps: int = 200) -> EvaluationStats:
    rewards = []
    lengths = []
    for _ in range(episodes):
        env = env_factory()
        observation = torch.as_tensor(env.reset(), dtype=torch.float32)
        hidden = model.init_hidden() if hasattr(model, "init_hidden") else None
        total_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            out = model.forward_step(observation.unsqueeze(0), hidden)
            hidden = out.h if hasattr(out, "h") else out[1]
            q_values = out.q if hasattr(out, "q") else out[0]
            action = int(q_values.argmax(dim=-1).item())
            
            step_result = env.step(action)
            if len(step_result) == 3:
                next_observation, reward, done = step_result
            else:
                next_observation, reward, done, _ = step_result
                
            observation = torch.as_tensor(next_observation, dtype=torch.float32)
            total_reward += float(reward)
            steps += 1
            if done:
                break
        rewards.append(total_reward)
        lengths.append(steps)
        env.close()
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    length_tensor = torch.tensor(lengths, dtype=torch.float32)
    return EvaluationStats(
        mean_reward=float(reward_tensor.mean()),
        std_reward=float(reward_tensor.std(unbiased=False)),
        mean_episode_length=float(length_tensor.mean()),
    )
