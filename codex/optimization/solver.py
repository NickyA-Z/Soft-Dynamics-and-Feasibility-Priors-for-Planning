from __future__ import annotations

from dataclasses import dataclass

import torch

from codex.config import OptimizerConfig
from codex.objectives.combined import PlanningObjective
from codex.optimization.actions import actions_to_raw, raw_to_actions
from codex.types import PlanResult


@dataclass
class _Best:
    value: float = float("inf")
    latents: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    breakdown: dict[str, float] | None = None
    iteration: int = 0


class TrajectoryOptimizer:
    """Optimize free future latents and bounded actions with best-iterate recovery."""

    def __init__(
        self,
        objective: PlanningObjective,
        config: OptimizerConfig,
        action_low: float | torch.Tensor,
        action_high: float | torch.Tensor,
    ) -> None:
        self.objective = objective
        self.config = config
        self.action_low = action_low
        self.action_high = action_high

    def solve(
        self,
        initial_latents: torch.Tensor,
        initial_actions: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        reference_latents: torch.Tensor | None = None,
        optimize_latents: bool = True,
        prefix_history: torch.Tensor | None = None,
    ) -> PlanResult:
        if initial_latents.shape[0] != initial_actions.shape[0] + 1:
            raise ValueError("initial_latents must be one longer than initial_actions")
        fixed_start = initial_latents[:1].detach()
        future = torch.nn.Parameter(initial_latents[1:].detach().clone())
        raw_actions = torch.nn.Parameter(
            actions_to_raw(initial_actions.detach(), self.action_low, self.action_high)
        )
        noise = self._fixed_noise(initial_actions.shape[0], initial_latents[0])
        best = _Best()
        if self.config.mode == "joint":
            params = [raw_actions] + ([future] if optimize_latents else [])
            optimizer = torch.optim.Adam(params, lr=self.config.action_learning_rate)
            stale = 0
            for iteration in range(1, self.config.joint_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                breakdown, latents, actions = self._evaluate(
                    fixed_start, future, raw_actions, goal_latent, reference_latents, noise,
                    prefix_history,
                )
                breakdown.total.backward()
                self._clip(params)
                optimizer.step()
                stale = self._record(best, breakdown, latents, actions, iteration, stale)
                if stale >= self.config.early_stop_patience:
                    break
        else:
            iteration = 0
            stale = 0
            latent_optimizer = (
                torch.optim.Adam([future], lr=self.config.latent_learning_rate)
                if optimize_latents else None
            )
            action_optimizer = torch.optim.Adam(
                [raw_actions], lr=self.config.action_learning_rate
            )
            stop = False
            for _ in range(self.config.rounds):
                if latent_optimizer is not None:
                    for _ in range(self.config.latent_steps_per_round):
                        iteration += 1
                        latent_optimizer.zero_grad(set_to_none=True)
                        breakdown, latents, actions = self._evaluate(
                            fixed_start, future, raw_actions.detach(), goal_latent,
                            reference_latents, noise, prefix_history,
                        )
                        breakdown.total.backward()
                        self._clip([future])
                        latent_optimizer.step()
                        stale = self._record(best, breakdown, latents, actions, iteration, stale)
                        if stale >= self.config.early_stop_patience:
                            stop = True
                            break
                if stop:
                    break
                for _ in range(self.config.action_steps_per_round):
                    iteration += 1
                    action_optimizer.zero_grad(set_to_none=True)
                    breakdown, latents, actions = self._evaluate(
                        fixed_start, future.detach(), raw_actions, goal_latent,
                        reference_latents, noise, prefix_history,
                    )
                    breakdown.total.backward()
                    self._clip([raw_actions])
                    action_optimizer.step()
                    stale = self._record(best, breakdown, latents, actions, iteration, stale)
                    if stale >= self.config.early_stop_patience:
                        stop = True
                        break
                if stop:
                    break
        if best.latents is None or best.actions is None or best.breakdown is None:
            raise RuntimeError("Optimizer did not produce a finite candidate")
        return PlanResult(
            latents=best.latents,
            actions=best.actions,
            objective=best.value,
            breakdown=best.breakdown,
            iterations=best.iteration,
        )

    def _evaluate(self, start, future, raw_actions, goal, reference, noise, prefix_history):
        latents = torch.cat([start, future], dim=0)
        actions = raw_to_actions(raw_actions, self.action_low, self.action_high)
        breakdown = self.objective(
            latents, actions, goal,
            reference_latents=reference,
            dsm_noise=noise,
            prefix_history=prefix_history,
        )
        return breakdown, latents, actions

    def _fixed_noise(self, horizon: int, example: torch.Tensor) -> torch.Tensor | None:
        if self.objective.config.dsm_mode != "noisy_probe":
            return None
        generator = torch.Generator(device=example.device)
        generator.manual_seed(0)
        return torch.randn(
            (horizon, *example.shape), generator=generator,
            device=example.device, dtype=example.dtype,
        )

    def _clip(self, params: list[torch.Tensor]) -> None:
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip_norm)

    def _record(self, best, breakdown, latents, actions, iteration, stale):
        value = float(breakdown.total.detach().cpu())
        if not torch.isfinite(breakdown.total):
            return stale + 1
        if value < best.value - self.config.improvement_tolerance:
            best.value = value
            best.latents = latents.detach().clone()
            best.actions = actions.detach().clone()
            best.breakdown = breakdown.detached()
            best.iteration = iteration
            return 0
        return stale + 1
