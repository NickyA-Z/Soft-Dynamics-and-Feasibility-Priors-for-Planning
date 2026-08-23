from __future__ import annotations

from typing import Any

import torch

from local.gvpwm.adapters.dino_wm import DinoWorldModelAdapter


class WorldModelAdapter:
    """Thin planning facade around the existing DINO-WM adapter.

    Encoding and goal-loss behavior are delegated to
    ``local.gvpwm.adapters.dino_wm.DinoWorldModelAdapter`` rather than copied.
    DINO-WM transition prediction is intentionally not exposed to the
    feasibility-only planner.
    """

    def __init__(self, adapter: DinoWorldModelAdapter) -> None:
        self.adapter = adapter

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def history_length(self) -> int:
        return self.adapter.history_length

    @property
    def action_dim(self) -> int:
        return self.adapter.action_dim

    def encode_observation(self, observation: Any) -> torch.Tensor:
        return self.adapter.encode_observation(observation)

    def encode_sequence(self, observations: Any) -> torch.Tensor:
        return self.adapter.encode_sequence(observations)

    def goal_loss(self, latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.adapter.goal_loss(latent, goal)
