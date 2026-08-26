from __future__ import annotations

import torch

from codex.optimization.actions import actions_to_raw, raw_to_actions


def local_action_negatives(
    positive: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    scales: tuple[float, ...],
) -> torch.Tensor:
    """Generate fixed-transition local negatives at several action scales."""
    span = (high - low).to(positive)
    candidates = []
    for scale in scales:
        noise = torch.randn_like(positive) * span * float(scale)
        candidates.append((positive + noise).maximum(low).minimum(high))
    return torch.stack(candidates, dim=1)


def global_action_negatives(
    positive: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    num_shuffles: int = 2,
) -> torch.Tensor:
    """Generate broad action negatives while keeping the transition fixed.

    Zero, negated, shuffled, and random negatives are adapted from
    ``extension.feasibility2.losses.contrastive_energy_loss``. The random and
    negated actions are additionally clipped to the dataset-derived
    per-coordinate bounds, which the original function does not enforce.
    """
    low = low.to(positive)
    high = high.to(positive)
    candidates = [
        torch.zeros_like(positive).maximum(low).minimum(high),
        (-positive).maximum(low).minimum(high),
        low + torch.rand_like(positive) * (high - low),
    ]
    max_shuffles = min(num_shuffles, positive.shape[0] - 1)
    candidates.extend(
        positive.roll(shifts=shift, dims=0)
        for shift in range(1, max_shuffles + 1)
    )
    return torch.stack(candidates, dim=1)


def mine_adversarial_actions(
    model: torch.nn.Module,
    history: torch.Tensor,
    positive_action: torch.Tensor,
    next_latent: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    noise_level: float,
    starts: int,
    steps: int,
    learning_rate: float,
    minimum_rms_distance: float,
    distance_penalty: float,
) -> torch.Tensor:
    """Mine bounded low-energy actions with the next latent held fixed.

    The bounded tanh parameterization is reused from
    ``codex.optimization.actions``. Unlike free-latent planner mining, this
    function deliberately keeps both history and next latent identical to the
    positive transition, so the energy must learn action discrimination.
    """
    batch, action_dim = positive_action.shape
    low = low.to(positive_action)
    high = high.to(positive_action)
    uniform = torch.rand(batch, starts, action_dim, device=positive_action.device)
    initial = low + uniform * (high - low)
    raw = torch.nn.Parameter(actions_to_raw(initial, low, high))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    expanded_history = history[:, None].expand(
        batch, starts, *history.shape[1:]
    ).reshape(batch * starts, *history.shape[1:])
    expanded_next = next_latent[:, None].expand(
        batch, starts, *next_latent.shape[1:]
    ).reshape(batch * starts, *next_latent.shape[1:])
    positive = positive_action[:, None]

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidates = raw_to_actions(raw, low, high)
        energies = model.energy(
            expanded_history,
            candidates.reshape(batch * starts, action_dim),
            expanded_next,
            noise_level=noise_level,
            reduction="none",
        ).reshape(batch, starts)
        rms_distance = (candidates - positive).pow(2).mean(dim=-1).sqrt()
        stay_wrong = torch.relu(minimum_rms_distance - rms_distance).pow(2)
        objective = (energies + distance_penalty * stay_wrong).mean()
        gradient = torch.autograd.grad(objective, raw, only_inputs=True)[0]
        raw.grad = gradient
        optimizer.step()

    with torch.no_grad():
        candidates = raw_to_actions(raw, low, high)
        energies = model.energy(
            expanded_history,
            candidates.reshape(batch * starts, action_dim),
            expanded_next,
            noise_level=noise_level,
            reduction="none",
        ).reshape(batch, starts)
        distance = (candidates - positive).pow(2).mean(dim=-1).sqrt()
        valid = distance >= minimum_rms_distance
        selection_energy = torch.where(valid, energies, torch.full_like(energies, torch.inf))
        choice = selection_energy.argmin(dim=1)
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            choice[no_valid] = distance[no_valid].argmax(dim=1)
        return candidates[torch.arange(batch, device=candidates.device), choice].detach()
