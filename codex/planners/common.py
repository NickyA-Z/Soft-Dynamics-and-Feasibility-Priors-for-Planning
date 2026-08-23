from __future__ import annotations

import torch


def interpolate_latents(start: torch.Tensor, goal: torch.Tensor, horizon: int) -> torch.Tensor:
    """Create a straight latent-space initialization including both endpoints."""
    weights = torch.linspace(0, 1, horizon + 1, device=start.device, dtype=start.dtype)
    weights = weights.reshape(horizon + 1, *([1] * start.ndim))
    return start.unsqueeze(0) * (1 - weights) + goal.unsqueeze(0) * weights


def zero_actions(horizon: int, action_dim: int, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros(horizon, action_dim, device=like.device, dtype=like.dtype)
