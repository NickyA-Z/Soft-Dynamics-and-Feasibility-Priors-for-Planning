from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class LangevinResult:
    actions: torch.Tensor
    cost: float
    initial_cost: float
    cost_history: list[float]


def langevin_action_search(
    initial_actions: torch.Tensor,
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_steps: int = 20,
    step_size: float = 1e-3,
    temperature: float = 1e-4,
    num_restarts: int = 1,
    restart_noise_std: float = 0.05,
    grad_clip_norm: float | None = 10.0,
    add_noise: bool = True,
) -> LangevinResult:
    """
    Refine an action sequence using gradient descent plus Langevin noise.

    Update:
        actions <- actions - step_size * grad(cost)
                   + sqrt(2 * step_size * temperature) * noise

    cost_fn must return a scalar tensor and remain differentiable with
    respect to its action argument.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least one.")

    action_low = action_low.to(initial_actions)
    action_high = action_high.to(initial_actions)

    best_actions = initial_actions.detach().clone()
    best_cost = float("inf")
    best_initial_cost = float("inf")
    best_history: list[float] = []

    for restart in range(num_restarts):
        if restart == 0:
            actions = initial_actions.detach().clone()
        else:
            actions = (
                initial_actions
                + restart_noise_std * torch.randn_like(initial_actions)
            ).clamp(action_low, action_high)

        with torch.no_grad():
            restart_initial_cost = float(cost_fn(actions).detach().cpu())

        cost_history = [restart_initial_cost]
        restart_best_actions = actions.detach().clone()
        restart_best_cost = restart_initial_cost

        for _ in range(num_steps):
            actions = actions.detach().requires_grad_(True)
            cost = cost_fn(actions)

            if cost.ndim != 0:
                raise ValueError(
                    f"cost_fn must return a scalar, got shape {tuple(cost.shape)}."
                )

            gradient = torch.autograd.grad(
                cost,
                actions,
                create_graph=False,
                retain_graph=False,
            )[0]

            if not torch.isfinite(gradient).all():
                raise RuntimeError("Non-finite Langevin action gradient.")

            if grad_clip_norm is not None:
                grad_norm = gradient.norm()
                if grad_norm > grad_clip_norm:
                    gradient = gradient * (
                        grad_clip_norm / grad_norm.clamp_min(1e-12)
                    )

            with torch.no_grad():
                actions = actions - step_size * gradient

                if add_noise and temperature > 0:
                    noise_scale = (
                        2.0 * step_size * temperature
                    ) ** 0.5
                    actions = actions + noise_scale * torch.randn_like(actions)

                actions.clamp_(action_low, action_high)

                new_cost = float(cost_fn(actions).detach().cpu())
                cost_history.append(new_cost)

                if new_cost < restart_best_cost:
                    restart_best_cost = new_cost
                    restart_best_actions = actions.clone()

        if restart_best_cost < best_cost:
            best_cost = restart_best_cost
            best_initial_cost = restart_initial_cost
            best_actions = restart_best_actions
            best_history = cost_history

    return LangevinResult(
        actions=best_actions,
        cost=best_cost,
        initial_cost=best_initial_cost,
        cost_history=best_history,
    )