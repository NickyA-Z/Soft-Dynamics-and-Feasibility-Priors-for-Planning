from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def action_recovery_metrics(planner: torch.Tensor, expert: torch.Tensor) -> dict[str, float]:
    """Compare a recovered action with expert and zero-action baselines."""
    zero = torch.zeros_like(expert)
    return {
        "mse_planner_expert": float(F.mse_loss(planner, expert).detach()),
        "mse_zero_expert": float(F.mse_loss(zero, expert).detach()),
        "cosine_planner_expert": float(F.cosine_similarity(
            planner.reshape(1, -1), expert.reshape(1, -1)
        ).detach()),
    }


def next_state_error(
    action: torch.Tensor,
    expert_next_state: torch.Tensor,
    simulate_from_same_state: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """Measure environment next-state MSE from an independently reset state."""
    predicted = simulate_from_same_state(action)
    return float(F.mse_loss(predicted, expert_next_state).detach())
