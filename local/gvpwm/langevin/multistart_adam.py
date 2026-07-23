from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import ActionEvaluation, ActionSearchResult, EvaluateActions


@dataclass
class MultiStartAdamConfig:
    steps: int = 50
    learning_rate: float = 1e-2
    adam_eps: float = 1e-8
    grad_clip_norm: float | None = 10.0
    raw_action_limit: float | None = 3.0


def _snapshot(
    raw_actions: torch.Tensor,
    evaluation: ActionEvaluation,
    chain_index: int,
    step_index: int,
) -> ActionSearchResult:
    return ActionSearchResult(
        raw_actions=raw_actions.detach().clone(),
        actions=evaluation.actions.detach().clone(),
        latents=evaluation.latents.detach().clone(),
        objective=float(evaluation.loss.detach().cpu()),
        diagnostics=dict(evaluation.diagnostics),
        chain_index=chain_index,
        step_index=step_index,
    )


def _keep_better(
    current: ActionSearchResult | None,
    candidate: ActionSearchResult,
) -> ActionSearchResult:
    if current is None or candidate.objective < current.objective:
        return candidate
    return current


def run_multistart_adam(
    initial_parameters: list[torch.Tensor],
    evaluate_actions: EvaluateActions,
    config: MultiStartAdamConfig,
) -> ActionSearchResult:
    """Optimize every initialization with Adam and return the best evaluation."""

    if not initial_parameters:
        raise ValueError("At least one initialization is required")
    if config.steps < 0:
        raise ValueError("steps must be non-negative")

    best: ActionSearchResult | None = None

    for chain_index, initial_parameter in enumerate(initial_parameters):
        raw_actions = torch.nn.Parameter(initial_parameter.detach().clone())
        optimizer = torch.optim.Adam(
            [raw_actions],
            lr=config.learning_rate,
            eps=config.adam_eps,
        )

        with torch.no_grad():
            initial_evaluation = evaluate_actions(raw_actions)
        if not torch.isfinite(initial_evaluation.loss):
            raise RuntimeError(f"Non-finite initial loss in Adam chain {chain_index}")
        best = _keep_better(
            best,
            _snapshot(raw_actions, initial_evaluation, chain_index, -1),
        )

        for step_index in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            evaluation = evaluate_actions(raw_actions)

            if not torch.isfinite(evaluation.loss):
                raise RuntimeError(
                    f"Non-finite loss in Adam chain {chain_index}, step {step_index}"
                )
            evaluation.loss.backward()

            if raw_actions.grad is None:
                raise RuntimeError("No gradient reached the action parameters")
            if not torch.isfinite(raw_actions.grad).all():
                raise RuntimeError("Non-finite action gradient")

            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([raw_actions], config.grad_clip_norm)

            optimizer.step()
            if config.raw_action_limit is not None:
                with torch.no_grad():
                    raw_actions.clamp_(
                        -config.raw_action_limit,
                        config.raw_action_limit,
                    )

            with torch.no_grad():
                updated_evaluation = evaluate_actions(raw_actions)
            best = _keep_better(
                best,
                _snapshot(raw_actions, updated_evaluation, chain_index, step_index),
            )

    if best is None:
        raise RuntimeError("Multi-start Adam produced no valid result")
    return best
