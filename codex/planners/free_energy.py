from __future__ import annotations

import torch

from codex.optimization.solver import TrajectoryOptimizer
from codex.planners.common import interpolate_latents, zero_actions
from codex.types import PlanResult


class FreeEnergyPlanner:
    """Jointly optimize actions and free future latents without world dynamics."""

    def __init__(self, optimizer: TrajectoryOptimizer, horizon: int, action_dim: int) -> None:
        self.optimizer = optimizer
        self.horizon = horizon
        self.action_dim = action_dim

    def plan(self, current_latent: torch.Tensor, goal_latent: torch.Tensor,
             *, initial_actions: torch.Tensor | None = None,
             reference_latents: torch.Tensor | None = None,
             latent_history: torch.Tensor | None = None) -> PlanResult:
        latents = interpolate_latents(current_latent, goal_latent, self.horizon)
        actions = initial_actions if initial_actions is not None else zero_actions(
            self.horizon, self.action_dim, current_latent
        )
        return self.optimizer.solve(
            latents, actions, goal_latent, reference_latents=reference_latents,
            optimize_latents=True, prefix_history=latent_history,
        )
