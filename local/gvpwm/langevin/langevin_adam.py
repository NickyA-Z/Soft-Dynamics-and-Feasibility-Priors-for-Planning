from __future__ import annotations

from dataclasses import dataclass

import torch

from .multistart_adam import MultiStartAdamConfig, run_multistart_adam
from .schedules import exponential_schedule
from .types import ActionSearchResult, EvaluateActions


@dataclass
class LangevinAdamConfig:
    langevin_steps: int = 15
    adam_steps: int = 35
    initial_step_size: float = 3e-3
    final_step_size: float = 3e-4
    initial_temperature: float = 1e-3
    final_temperature: float = 1e-6
    grad_clip_norm: float | None = 10.0
    raw_action_limit: float | None = 3.0
    adam_learning_rate: float = 1e-2
    adam_eps: float = 1e-8


def _run_langevin_chain(
    initial_parameter: torch.Tensor,
    evaluate_actions: EvaluateActions,
    config: LangevinAdamConfig,
    chain_index: int,
    generator: torch.Generator | None,
) -> ActionSearchResult:
    raw_actions = torch.nn.Parameter(initial_parameter.detach().clone())

    with torch.no_grad():
        initial_evaluation = evaluate_actions(raw_actions)
    if not torch.isfinite(initial_evaluation.loss):
        raise RuntimeError(f"Non-finite initial loss in Langevin chain {chain_index}")

    best = ActionSearchResult(
        raw_actions=raw_actions.detach().clone(),
        actions=initial_evaluation.actions.detach().clone(),
        latents=initial_evaluation.latents.detach().clone(),
        objective=float(initial_evaluation.loss.detach().cpu()),
        diagnostics=dict(initial_evaluation.diagnostics),
        chain_index=chain_index,
        step_index=-1,
    )

    for step_index in range(config.langevin_steps):
        raw_actions.grad = None
        evaluation = evaluate_actions(raw_actions)

        if not torch.isfinite(evaluation.loss):
            raise RuntimeError(
                f"Non-finite loss in Langevin chain {chain_index}, step {step_index}"
            )
        evaluation.loss.backward()

        if raw_actions.grad is None:
            raise RuntimeError("No gradient reached the action parameters")
        if not torch.isfinite(raw_actions.grad).all():
            raise RuntimeError("Non-finite action gradient")

        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([raw_actions], config.grad_clip_norm)

        step_size = exponential_schedule(
            config.initial_step_size,
            config.final_step_size,
            step_index,
            config.langevin_steps,
        )
        temperature = exponential_schedule(
            config.initial_temperature,
            config.final_temperature,
            step_index,
            config.langevin_steps,
        )
        noise_std = (2.0 * step_size * temperature) ** 0.5

        with torch.no_grad():
            raw_actions.add_(raw_actions.grad, alpha=-step_size)
            if noise_std > 0.0:
                noise = torch.randn(
                    raw_actions.shape,
                    device=raw_actions.device,
                    dtype=raw_actions.dtype,
                    generator=generator,
                )
                raw_actions.add_(noise, alpha=noise_std)
            if config.raw_action_limit is not None:
                raw_actions.clamp_(
                    -config.raw_action_limit,
                    config.raw_action_limit,
                )

            updated_evaluation = evaluate_actions(raw_actions)

        updated_cost = float(updated_evaluation.loss.detach().cpu())
        if updated_cost < best.objective:
            best = ActionSearchResult(
                raw_actions=raw_actions.detach().clone(),
                actions=updated_evaluation.actions.detach().clone(),
                latents=updated_evaluation.latents.detach().clone(),
                objective=updated_cost,
                diagnostics=dict(updated_evaluation.diagnostics),
                chain_index=chain_index,
                step_index=step_index,
            )

    return best


def run_langevin_adam(
    initial_parameters: list[torch.Tensor],
    evaluate_actions: EvaluateActions,
    config: LangevinAdamConfig,
    *,
    generator: torch.Generator | None = None,
) -> ActionSearchResult:
    """Explore with annealed Langevin chains, then refine each best point with Adam."""

    if not initial_parameters:
        raise ValueError("At least one initialization is required")
    if config.langevin_steps < 0 or config.adam_steps < 0:
        raise ValueError("langevin_steps and adam_steps must be non-negative")

    explored_candidates = [
        _run_langevin_chain(
            initial_parameter,
            evaluate_actions,
            config,
            chain_index,
            generator,
        ).raw_actions
        for chain_index, initial_parameter in enumerate(initial_parameters)
    ]

    return run_multistart_adam(
        initial_parameters=explored_candidates,
        evaluate_actions=evaluate_actions,
        config=MultiStartAdamConfig(
            steps=config.adam_steps,
            learning_rate=config.adam_learning_rate,
            adam_eps=config.adam_eps,
            grad_clip_norm=config.grad_clip_norm,
            raw_action_limit=config.raw_action_limit,
        ),
    )
