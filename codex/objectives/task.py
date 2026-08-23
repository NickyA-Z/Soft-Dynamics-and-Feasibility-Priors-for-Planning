from __future__ import annotations

import torch


def action_l2(actions: torch.Tensor) -> torch.Tensor:
    return actions.pow(2).mean()


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[0] < 2:
        return actions.new_zeros(())
    return (actions[1:] - actions[:-1]).pow(2).mean()


def latent_acceleration(latents: torch.Tensor) -> torch.Tensor:
    if latents.shape[0] < 3:
        return latents.new_zeros(())
    velocity = latents[1:] - latents[:-1]
    return (velocity[1:] - velocity[:-1]).pow(2).mean()


def reference_alignment(
    latents: torch.Tensor,
    reference: torch.Tensor | None,
) -> torch.Tensor:
    if reference is None:
        return latents.new_zeros(())
    if latents.shape != reference.shape:
        raise ValueError("Reference and candidate latent shapes must match")
    return (latents[1:-1] - reference[1:-1]).pow(2).mean() if latents.shape[0] > 2 else latents.new_zeros(())
