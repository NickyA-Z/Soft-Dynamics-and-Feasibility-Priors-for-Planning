from __future__ import annotations

import torch

from codex.optimization.solver import TrajectoryOptimizer
from codex.types import PlanResult


class OpenLoopPlanner:
    """Recover actions for an entire fixed latent trajectory; latents never move."""

    def __init__(self, optimizer: TrajectoryOptimizer) -> None:
        self.optimizer = optimizer

    def plan(self, fixed_latents: torch.Tensor, initial_actions: torch.Tensor) -> PlanResult:
        return self.optimizer.solve(
            fixed_latents,
            initial_actions,
            fixed_latents[-1],
            reference_latents=fixed_latents,
            optimize_latents=False,
        )
