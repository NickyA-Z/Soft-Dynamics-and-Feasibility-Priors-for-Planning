from __future__ import annotations

import torch

from codex.config import ObjectiveConfig
from codex.models.feasibility import FeasibilityScorer
from codex.models.world_model import WorldModelAdapter
from codex.types import ObjectiveBreakdown
from .task import action_l2, action_smoothness, latent_acceleration, reference_alignment


def history_window(latents: torch.Tensor, end: int, length: int) -> torch.Tensor:
    """Build a left-padded latent history ending immediately before ``end``.

    This is adapted from ``extension.feasibility2.dataset.make_history_windows``;
    padding repeats the earliest available state, matching dataset creation.
    """
    history = latents[max(0, end - length):end]
    if history.shape[0] == 0:
        raise ValueError("Cannot construct history before the first latent")
    if history.shape[0] < length:
        padding = history[:1].expand(length - history.shape[0], *history.shape[1:])
        history = torch.cat([padding, history], dim=0)
    return history


def conditioned_history_window(
    latents: torch.Tensor,
    end: int,
    length: int,
    prefix_history: torch.Tensor | None,
) -> torch.Tensor:
    """Build history using real pre-plan states before candidate states.

    The left-padding convention is adapted from
    ``extension.feasibility2.dataset.make_history_windows``. At MPC step zero,
    this prevents the current state from being repeated when real history is
    available.
    """
    if prefix_history is None:
        return history_window(latents, end, length)
    if prefix_history.shape[1:] != latents.shape[1:]:
        raise ValueError("prefix_history latent shape does not match candidates")
    # prefix includes the current state, as does latents[0], so add only future
    # candidate states before the transition being scored.
    available = torch.cat([prefix_history, latents[1:end]], dim=0)
    history = available[-length:]
    if history.shape[0] < length:
        history = torch.cat(
            [history[:1].expand(length - history.shape[0], *history.shape[1:]), history],
            dim=0,
        )
    return history


class PlanningObjective:
    def __init__(
        self,
        feasibility: FeasibilityScorer,
        world: WorldModelAdapter,
        config: ObjectiveConfig,
        history_length: int,
    ) -> None:
        self.feasibility = feasibility
        self.world = world
        self.config = config
        self.history_length = history_length

    def __call__(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        reference_latents: torch.Tensor | None = None,
        dsm_noise: torch.Tensor | None = None,
        prefix_history: torch.Tensor | None = None,
    ) -> ObjectiveBreakdown:
        if latents.shape[0] != actions.shape[0] + 1:
            raise ValueError("Expected one more latent than action")
        dsm_terms = []
        contrastive_terms = []
        for t in range(actions.shape[0]):
            history = conditioned_history_window(
                latents, t + 1, self.history_length, prefix_history
            )
            noise = None if dsm_noise is None else dsm_noise[t]
            dsm_terms.append(
                self.feasibility.dsm_energy(
                    history,
                    actions[t],
                    latents[t + 1],
                    noise_level=self.config.dsm_noise_level,
                    mode=self.config.dsm_mode,
                    noise=noise,
                )
            )
            contrastive_terms.append(
                self.feasibility.contrastive_energy(
                    history,
                    actions[t],
                    latents[t + 1],
                    noise_level=self.config.contrastive_noise_level,
                )
            )
        dsm = torch.stack(dsm_terms).mean()
        contrastive = torch.stack(contrastive_terms).mean()
        feasibility = self.config.lambda_dsm * dsm + self.config.lambda_contrastive * contrastive
        goal = self.world.goal_loss(latents[-1], goal_latent)
        action = action_l2(actions)
        action_smooth = action_smoothness(actions)
        latent_accel = latent_acceleration(latents)
        reference = reference_alignment(latents, reference_latents)
        total = (
            self.config.lambda_goal * goal
            + self.config.lambda_feasibility * feasibility
            + self.config.lambda_action * action
            + self.config.lambda_action_smoothness * action_smooth
            + self.config.lambda_latent_acceleration * latent_accel
            + self.config.lambda_reference * reference
        )
        return ObjectiveBreakdown(
            total=total,
            goal=goal,
            dsm=dsm,
            contrastive=contrastive,
            feasibility=feasibility,
            action=action,
            action_smoothness=action_smooth,
            latent_acceleration=latent_accel,
            reference=reference,
        )
