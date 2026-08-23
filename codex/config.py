from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


DSMMode = Literal["paper_clean", "noisy_probe"]
OptimizationMode = Literal["joint", "alternating"]


@dataclass
class ObjectiveConfig:
    lambda_goal: float = 100.0
    lambda_feasibility: float = 1.0
    lambda_dsm: float = 1.0
    lambda_contrastive: float = 1.0
    lambda_action: float = 1e-4
    lambda_action_smoothness: float = 1e-4
    lambda_latent_acceleration: float = 1e-4
    lambda_reference: float = 0.0
    dsm_noise_level: float = 0.2
    contrastive_noise_level: float = 0.2
    dsm_mode: DSMMode = "paper_clean"

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name.startswith("lambda_") and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.dsm_noise_level < 0 or self.contrastive_noise_level < 0:
            raise ValueError("Noise levels must be non-negative")


@dataclass
class OptimizerConfig:
    mode: OptimizationMode = "alternating"
    rounds: int = 20
    joint_steps: int = 400
    latent_steps_per_round: int = 10
    action_steps_per_round: int = 10
    latent_learning_rate: float = 1e-2
    action_learning_rate: float = 3e-2
    gradient_clip_norm: float | None = 10.0
    early_stop_patience: int = 50
    improvement_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if min(
            self.rounds,
            self.joint_steps,
            self.latent_steps_per_round,
            self.action_steps_per_round,
        ) < 1:
            raise ValueError("All optimization step counts must be positive")
        if self.latent_learning_rate <= 0 or self.action_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")


@dataclass
class PlannerConfig:
    horizon: int = 5
    history_length: int = 3
    action_low: float | torch.Tensor = -3.0
    action_high: float | torch.Tensor = 3.0
    seed: int = 0
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.history_length < 1:
            raise ValueError("horizon and history_length must be positive")
        low = torch.as_tensor(self.action_low)
        high = torch.as_tensor(self.action_high)
        if low.shape != high.shape:
            raise ValueError("action_low and action_high must have matching shapes")
        if not torch.all(low < high):
            raise ValueError("Every action_low coordinate must be smaller than action_high")
