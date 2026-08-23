from __future__ import annotations

import torch

from codex.config import ObjectiveConfig, OptimizerConfig
from codex.models.feasibility import FeasibilityScorer
from codex.objectives.combined import history_window
from codex.optimization.actions import actions_to_raw, raw_to_actions
from codex.types import FixedTransitionResult


class FixedTransitionPlanner:
    """Recover an action while holding history and the true next latent fixed."""

    def __init__(self, feasibility: FeasibilityScorer, objective: ObjectiveConfig,
                 optimizer: OptimizerConfig,
                 action_low: float | torch.Tensor,
                 action_high: float | torch.Tensor) -> None:
        self.feasibility = feasibility
        self.objective = objective
        self.optimizer = optimizer
        self.action_low = action_low
        self.action_high = action_high

    def energies(self, history: torch.Tensor, next_latent: torch.Tensor,
                 actions: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
        """Report comparable fixed-transition DSM and contrastive energies."""
        return {name: self._breakdown(history, action, next_latent) for name, action in actions.items()}

    def recover(self, history: torch.Tensor, next_latent: torch.Tensor,
                initial_action: torch.Tensor) -> FixedTransitionResult:
        raw = torch.nn.Parameter(actions_to_raw(initial_action, self.action_low, self.action_high))
        opt = torch.optim.Adam([raw], lr=self.optimizer.action_learning_rate)
        best_value, best_action, best_breakdown, best_step = float("inf"), None, None, 0
        for step in range(1, self.optimizer.joint_steps + 1):
            opt.zero_grad(set_to_none=True)
            action = raw_to_actions(raw, self.action_low, self.action_high)
            dsm, contrastive, total = self._terms(history, action, next_latent)
            total.backward()
            if self.optimizer.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([raw], self.optimizer.gradient_clip_norm)
            opt.step()
            value = float(total.detach().cpu())
            if value < best_value:
                best_value, best_step = value, step
                best_action = action.detach().clone()
                best_breakdown = {"dsm": float(dsm.detach()), "contrastive": float(contrastive.detach())}
        if best_action is None or best_breakdown is None:
            raise RuntimeError("No finite fixed-transition action found")
        return FixedTransitionResult(best_action, best_value, best_breakdown, best_step)

    def _terms(self, history, action, next_latent):
        dsm = self.feasibility.dsm_energy(
            history, action, next_latent,
            noise_level=self.objective.dsm_noise_level,
            mode="paper_clean",
        )
        contrastive = self.feasibility.contrastive_energy(
            history, action, next_latent,
            noise_level=self.objective.contrastive_noise_level,
        )
        total = self.objective.lambda_dsm * dsm + self.objective.lambda_contrastive * contrastive
        return dsm, contrastive, total

    def _breakdown(self, history, action, next_latent):
        with torch.no_grad():
            dsm, contrastive, total = self._terms(history, action, next_latent)
        return {"dsm": float(dsm), "contrastive": float(contrastive), "total": float(total)}
