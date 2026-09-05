from __future__ import annotations

import torch

from .model import TransformerFeasibilityModel


@torch.no_grad()
def sample_actions(
    model: TransformerFeasibilityModel,
    history: torch.Tensor,
    desired_next_latent: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    num_samples: int = 1,
    num_steps: int = 20,
    sigma_min: float = 0.05,
    sigma_max: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample normalized actions with a deterministic VE/DDIM update."""
    if num_samples < 1 or num_steps < 2:
        raise ValueError("num_samples must be >= 1 and num_steps must be >= 2")
    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max")

    single = history.ndim == 1 + len(model.latent_shape)
    if single:
        history = history.unsqueeze(0)
        desired_next_latent = desired_next_latent.unsqueeze(0)
    batch = history.shape[0]
    history = history.repeat_interleave(num_samples, dim=0)
    desired_next_latent = desired_next_latent.repeat_interleave(num_samples, dim=0)

    low = action_low.to(history).reshape(1, -1)
    high = action_high.to(history).reshape(1, -1)
    midpoint = 0.5 * (low + high)
    actions = midpoint + sigma_max * torch.randn(
        batch * num_samples,
        model.action_dim,
        device=history.device,
        dtype=history.dtype,
        generator=generator,
    )
    actions = actions.maximum(low).minimum(high)
    sigmas = torch.logspace(
        torch.log10(torch.as_tensor(sigma_max)).item(),
        torch.log10(torch.as_tensor(sigma_min)).item(),
        num_steps,
        device=history.device,
        dtype=history.dtype,
    )
    for index in range(num_steps - 1):
        sigma = sigmas[index].expand(actions.shape[0], 1)
        eps = model.predict_action_noise(
            history, actions, desired_next_latent, sigma
        )
        actions = actions + (sigmas[index + 1] - sigmas[index]) * eps
        actions = actions.maximum(low).minimum(high)

    sigma = sigmas[-1].expand(actions.shape[0], 1)
    actions = actions - sigmas[-1] * model.predict_action_noise(
        history, actions, desired_next_latent, sigma
    )
    actions = actions.maximum(low).minimum(high)
    result = actions.reshape(batch, num_samples, model.action_dim)
    return result.squeeze(0) if single else result
