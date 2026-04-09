from __future__ import annotations

import torch


def flatten_event_dims(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def safe_l2_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = flatten_event_dims(tensor)
    return flat / flat.norm(p=2).clamp_min(eps)


def scale_invariant_alignment(
    latent: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    lhs = safe_l2_normalize(latent, eps=eps)
    rhs = safe_l2_normalize(reference, eps=eps)
    return (lhs - rhs).pow(2).sum()


def goal_mse(latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    lhs = flatten_event_dims(latent)
    rhs = flatten_event_dims(goal)
    return (lhs - rhs).pow(2).mean()


def squared_norm(tensor: torch.Tensor) -> torch.Tensor:
    return flatten_event_dims(tensor).pow(2).sum()
