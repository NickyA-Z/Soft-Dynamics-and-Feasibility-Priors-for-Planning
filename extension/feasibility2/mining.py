"""Planner-facing helpers for exporting ranking-violating transitions."""

from __future__ import annotations

import torch


@torch.no_grad()
def make_hard_negative_record(
    model: torch.nn.Module,
    *,
    history: torch.Tensor,
    positive_action: torch.Tensor,
    positive_next_latent: torch.Tensor,
    negative_action: torch.Tensor,
    negative_next_latent: torch.Tensor,
    noise_level: float = 0.0,
    mining_margin: float = 0.1,
    require_violation: bool = True,
) -> dict | None:
    """Create one record from exact, already-normalized energy-head inputs."""
    if positive_action.shape != negative_action.shape:
        raise ValueError("Positive and negative action shapes differ")
    if positive_next_latent.shape != negative_next_latent.shape:
        raise ValueError("Positive and negative next-latent shapes differ")
    expected_action_dim = getattr(model, "action_dim", positive_action.shape[-1])
    if positive_action.shape[-1] != expected_action_dim:
        raise ValueError(
            f"Action dimension {positive_action.shape[-1]} != model.action_dim {expected_action_dim}"
        )
    positive_energy = model.energy(
        history,
        positive_action,
        positive_next_latent,
        noise_level=noise_level,
        reduction="mean",
    )
    negative_energy = model.energy(
        history,
        negative_action,
        negative_next_latent,
        noise_level=noise_level,
        reduction="mean",
    )
    hardness = positive_energy + mining_margin - negative_energy
    if require_violation and hardness.item() <= 0:
        return None
    return {
        "history": history.detach().cpu(),
        "positive_action": positive_action.detach().cpu(),
        "positive_next_latent": positive_next_latent.detach().cpu(),
        "negative_action": negative_action.detach().cpu(),
        "negative_next_latent": negative_next_latent.detach().cpu(),
        "positive_energy": float(positive_energy.item()),
        "negative_energy": float(negative_energy.item()),
        "hardness": float(hardness.item()),
    }
