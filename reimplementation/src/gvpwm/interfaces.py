from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .losses import goal_mse, scale_invariant_alignment
from .utils import ensure_history_length


@dataclass
class VideoPlan:
    data: Any
    encoded: bool = False


class VideoPlanSource(ABC):
    @abstractmethod
    def generate(
        self,
        initial_observation: Any,
        goal_observation: Any,
        horizon: int,
    ) -> VideoPlan:
        raise NotImplementedError


@dataclass
class CollocationResult:
    latents: torch.Tensor
    actions: torch.Tensor
    objective: float
    augmented_lagrangian: float
    dynamics_residual_norm: float
    multipliers: torch.Tensor
    rho: float
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class MPCStepResult:
    step_index: int
    planned_actions: torch.Tensor
    executed_actions: torch.Tensor
    collocation_objective: float
    dynamics_residual_norm: float


@dataclass
class MPCResult:
    video_latents: torch.Tensor
    executed_actions: torch.Tensor
    executed_latents: torch.Tensor
    steps: list[MPCStepResult]


class WorldModelAdapter(nn.Module, ABC):
    def __init__(
        self,
        history_length: int,
        action_dim: int,
        action_low: torch.Tensor | float,
        action_high: torch.Tensor | float,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.action_dim = int(action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim == 0:
            low = low.repeat(self.action_dim)
        if high.ndim == 0:
            high = high.repeat(self.action_dim)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    @property
    def device(self) -> torch.device:
        return self.action_low.device

    @abstractmethod
    def encode_observation(self, observation: Any) -> torch.Tensor:
        raise NotImplementedError

    def encode_sequence(self, observations: Any) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return torch.stack(
                [self.encode_observation(obs) for obs in observations],
                dim=0,
            )
        return torch.stack(
            [self.encode_observation(obs) for obs in observations],
            dim=0,
        )

    @abstractmethod
    def predict_next_latent(
        self,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def initialize_latents_from_video(
        self,
        current_latent: torch.Tensor,
        video_latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = video_latents.clone()
        latents[0] = current_latent
        return latents

    def video_alignment_loss(
        self,
        latent: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return scale_invariant_alignment(latent, reference)

    def goal_loss(
        self,
        latent: torch.Tensor,
        goal_latent: torch.Tensor,
    ) -> torch.Tensor:
        return goal_mse(latent, goal_latent)

    def rollout(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        planned_actions: torch.Tensor,
    ) -> torch.Tensor:
        latent_history = ensure_history_length(
            latent_context,
            self.history_length,
            pad_mode="repeat_first",
        )
        if self.history_length > 1:
            action_context = ensure_history_length(
                past_action_context,
                self.history_length - 1,
                pad_mode="zeros",
            )
        else:
            action_context = past_action_context.new_zeros((0, self.action_dim))

        predicted_latents: list[torch.Tensor] = []
        for index in range(planned_actions.shape[0]):
            state_window = latent_history[-self.history_length :]
            action_window = torch.cat(
                [action_context, planned_actions[index : index + 1]],
                dim=0,
            )[-self.history_length :]
            next_latent = self.predict_next_latent(state_window, action_window)
            predicted_latents.append(next_latent)
            latent_history = torch.cat([latent_history, next_latent.unsqueeze(0)], dim=0)
            if self.history_length > 1:
                action_context = action_window[-(self.history_length - 1) :]
        if not predicted_latents:
            return latent_context.new_empty((0, *latent_context.shape[1:]))
        return torch.stack(predicted_latents, dim=0)
