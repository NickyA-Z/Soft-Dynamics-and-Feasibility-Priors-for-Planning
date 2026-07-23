from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ActionEvaluation:
    """Differentiable objective and rollout produced for raw action parameters."""

    loss: torch.Tensor
    actions: torch.Tensor
    latents: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class ActionSearchResult:
    """Best evaluated action trajectory found by an action-search method."""

    raw_actions: torch.Tensor
    actions: torch.Tensor
    latents: torch.Tensor
    objective: float
    diagnostics: dict[str, float]
    chain_index: int
    step_index: int


EvaluateActions = Callable[[torch.Tensor], ActionEvaluation]
