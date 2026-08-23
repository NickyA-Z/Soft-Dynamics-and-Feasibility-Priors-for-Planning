from __future__ import annotations

import torch

from codex.config import PlannerConfig
from codex.models.feasibility import FeasibilityScorer
from codex.models.world_model import WorldModelAdapter
from codex.objectives.combined import PlanningObjective
from codex.optimization.solver import TrajectoryOptimizer
from codex.planners.fixed_transition import FixedTransitionPlanner
from codex.planners.free_energy import FreeEnergyPlanner
from codex.planners.open_loop import OpenLoopPlanner


def build_planners(
    dino_adapter,
    checkpoint: str,
    config: PlannerConfig,
    device: str | torch.device,
) -> dict[str, object]:
    """Construct all experiment stages around one frozen checkpoint."""
    world = WorldModelAdapter(dino_adapter)
    feasibility = FeasibilityScorer.from_checkpoint(checkpoint, device)
    objective = PlanningObjective(
        feasibility, world, config.objective, config.history_length
    )
    optimizer = TrajectoryOptimizer(
        objective, config.optimizer, config.action_low, config.action_high
    )
    return {
        "world": world,
        "feasibility": feasibility,
        "fixed_transition": FixedTransitionPlanner(
            feasibility, config.objective, config.optimizer,
            config.action_low, config.action_high,
        ),
        "open_loop": OpenLoopPlanner(optimizer),
        "free_energy": FreeEnergyPlanner(optimizer, config.horizon, world.action_dim),
    }
