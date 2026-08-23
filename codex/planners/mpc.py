from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from codex.models.world_model import WorldModelAdapter
from codex.planners.free_energy import FreeEnergyPlanner
from codex.types import PlanResult


@dataclass
class MPCStep:
    observation: Any
    action: torch.Tensor
    plan: PlanResult


class MPCPlanner:
    """Replan from the encoded actual state after every executed action.

    No suffix of an original oracle video is attached to an off-trajectory
    state. A reference may be supplied by ``reference_factory``, but it is
    regenerated from the latest observation on every iteration.
    """

    def __init__(self, planner: FreeEnergyPlanner, world: WorldModelAdapter,
                 history_length: int) -> None:
        self.planner = planner
        self.world = world
        self.history_length = history_length

    def run(
        self,
        initial_observation: Any,
        goal_observation: Any,
        execute: Callable[[torch.Tensor], Any],
        steps: int,
        *,
        reference_factory: Callable[[Any, Any, int], torch.Tensor | None] | None = None,
    ) -> list[MPCStep]:
        observation = initial_observation
        goal = self.world.encode_observation(goal_observation).detach()
        history: list[torch.Tensor] = []
        results: list[MPCStep] = []
        warm_actions = None
        for _ in range(steps):
            current = self.world.encode_observation(observation).detach()
            history.append(current)
            history = history[-self.history_length:]
            latent_history = torch.stack(history)
            reference = (
                reference_factory(observation, goal_observation, self.planner.horizon)
                if reference_factory is not None else None
            )
            plan = self.planner.plan(
                current, goal, initial_actions=warm_actions,
                reference_latents=reference, latent_history=latent_history,
            )
            action = plan.actions[0]
            results.append(MPCStep(observation, action.detach().clone(), plan))
            observation = execute(action)
            warm_actions = torch.cat([plan.actions[1:], plan.actions[-1:]], dim=0).detach()
        return results
