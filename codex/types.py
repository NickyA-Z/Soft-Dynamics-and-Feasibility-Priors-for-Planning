from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ObjectiveBreakdown:
    total: torch.Tensor
    goal: torch.Tensor
    dsm: torch.Tensor
    contrastive: torch.Tensor
    feasibility: torch.Tensor
    action: torch.Tensor
    action_smoothness: torch.Tensor
    latent_acceleration: torch.Tensor
    reference: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in vars(self).items()
        }


@dataclass
class PlanResult:
    latents: torch.Tensor
    actions: torch.Tensor
    objective: float
    breakdown: dict[str, float]
    iterations: int
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class FixedTransitionResult:
    action: torch.Tensor
    energy: float
    breakdown: dict[str, float]
    iterations: int
