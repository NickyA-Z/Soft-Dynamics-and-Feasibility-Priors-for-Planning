from __future__ import annotations

import torch


def raw_to_actions(
    raw: torch.Tensor,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
    """Map unconstrained variables into scalar or per-coordinate bounds."""
    low = torch.as_tensor(low, device=raw.device, dtype=raw.dtype)
    high = torch.as_tensor(high, device=raw.device, dtype=raw.dtype)
    midpoint = (high + low) / 2.0
    radius = (high - low) / 2.0
    return midpoint + radius * torch.tanh(raw)


def actions_to_raw(
    actions: torch.Tensor,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`raw_to_actions`, with finite values at the bounds."""
    low = torch.as_tensor(low, device=actions.device, dtype=actions.dtype)
    high = torch.as_tensor(high, device=actions.device, dtype=actions.dtype)
    midpoint = (high + low) / 2.0
    radius = (high - low) / 2.0
    unit = ((actions - midpoint) / radius).clamp(-0.999999, 0.999999)
    return torch.atanh(unit)
