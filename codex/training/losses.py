from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_action_ranking_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    positive_action: torch.Tensor,
    next_latent: torch.Tensor,
    negative_actions: torch.Tensor,
    *,
    noise_level: float,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank one positive below many action-only negatives for a fixed transition."""
    batch, negatives, action_dim = negative_actions.shape
    positive_energy = model.energy(
        history, positive_action, next_latent,
        noise_level=noise_level, reduction="none",
    )
    expanded_history = history[:, None].expand(
        batch, negatives, *history.shape[1:]
    ).reshape(batch * negatives, *history.shape[1:])
    expanded_next = next_latent[:, None].expand(
        batch, negatives, *next_latent.shape[1:]
    ).reshape(batch * negatives, *next_latent.shape[1:])
    negative_energy = model.energy(
        expanded_history,
        negative_actions.reshape(batch * negatives, action_dim),
        expanded_next,
        noise_level=noise_level,
        reduction="none",
    ).reshape(batch, negatives)
    hardest = negative_energy.min(dim=1).values
    loss = temperature * F.softplus(
        (margin + positive_energy - hardest) / temperature
    ).mean()
    return loss, {
        "positive_energy": positive_energy.detach().mean(),
        "negative_energy": negative_energy.detach().mean(),
        "hard_gap": (hardest - positive_energy).detach().mean(),
        "ranking_accuracy": (positive_energy < hardest).float().detach().mean(),
        "margin_accuracy": (positive_energy + margin < hardest).float().detach().mean(),
    }


def calibrated_energy_loss(positive: torch.Tensor, negative: torch.Tensor,
                           negative_margin: float = 1.0) -> torch.Tensor:
    """Anchor raw scalar energies without claiming that anchoring fixes ranking."""
    return F.softplus(positive).mean() + F.softplus(negative_margin - negative).mean()
