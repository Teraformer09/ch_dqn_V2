from __future__ import annotations
from dataclasses import dataclass
import torch








@dataclass(slots=True)
class SecondOrderState:
    velocity: dict[str, torch.Tensor]
    previous_velocity: dict[str, torch.Tensor]


class SecondOrderOptimizer:
    def __init__(self, params: dict[str, torch.nn.Parameter], lr: float, beta1: float, beta2: float) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = SecondOrderState(
            velocity={name: torch.zeros_like(param) for name, param in params.items()},
            previous_velocity={name: torch.zeros_like(param) for name, param in params.items()},
        )

    @torch.no_grad()
    def reset(self) -> None:
        for name in self.params:
            self.state.velocity[name].zero_()
            self.state.previous_velocity[name].zero_()

    @torch.no_grad()
    def step(self) -> None:
        for name, param in self.params.items():
            if param.grad is None:
                continue
            prev_v = self.state.velocity[name].clone()
            v = self.beta1 * self.state.velocity[name] + param.grad
            a = v - prev_v
            param.add_(-self.lr * v - self.beta2 * a)
            self.state.previous_velocity[name] = prev_v
            self.state.velocity[name] = v

