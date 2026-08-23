from __future__ import annotations

import torch


def raw_to_actions(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map unconstrained optimizer variables smoothly into action bounds."""
    midpoint = (high + low) / 2.0
    radius = (high - low) / 2.0
    return midpoint + radius * torch.tanh(raw)


def actions_to_raw(actions: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Inverse of :func:`raw_to_actions`, with finite values at the bounds."""
    midpoint = (high + low) / 2.0
    radius = (high - low) / 2.0
    unit = ((actions - midpoint) / radius).clamp(-0.999999, 0.999999)
    return torch.atanh(unit)
