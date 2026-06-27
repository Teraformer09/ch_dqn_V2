from __future__ import annotations

import copy

import torch


class EMASmoother:
    def __init__(self, model: torch.nn.Module, tau: float = 0.005) -> None:
        self.model = model
        self.target = copy.deepcopy(model)
        self.tau = tau
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self) -> None:
        for online, target in zip(self.model.parameters(), self.target.parameters()):
            target.data.copy_(self.tau * online.data + (1.0 - self.tau) * target.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target(x)
