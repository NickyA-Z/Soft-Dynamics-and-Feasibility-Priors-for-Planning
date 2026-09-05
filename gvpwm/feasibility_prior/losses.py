"""Energy-ranking losses for feasibility training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def paired_energy_ranking_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    positive_action: torch.Tensor,
    positive_z_next: torch.Tensor,
    negative_action: torch.Tensor,
    negative_z_next: torch.Tensor,
    *,
    noise_level: float,
    temperature: float,
    ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank an explicit planner-mined negative above its paired expert transition."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    positive_energy = model.energy(
        history,
        positive_action,
        positive_z_next,
        noise_level=noise_level,
        reduction="none",
    )
    negative_energy = model.energy(
        history,
        negative_action,
        negative_z_next,
        noise_level=noise_level,
        reduction="none",
    )
    violations = ranking_margin + positive_energy - negative_energy
    loss = temperature * F.softplus(violations / temperature).mean()
    metrics = {
        "positive_energy": positive_energy.detach().mean(),
        "negative_energy": negative_energy.detach().mean(),
        "energy_gap": (negative_energy - positive_energy).detach().mean(),
        "ranking_accuracy": (positive_energy < negative_energy).float().detach().mean(),
        "margin_accuracy": (positive_energy + ranking_margin < negative_energy)
        .float()
        .detach()
        .mean(),
    }
    return loss, metrics


def contrastive_energy_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    action: torch.Tensor,
    z_next: torch.Tensor,
    noise_level: float,
    temperature: float,
    ranking_margin: float,
    hard_negative_weight: float,
    calibration_margin: float,
    calibration_weight: float,
    num_action_shuffles: int,
    latent_noise_scales: tuple[float, ...],
    debug: bool = False,
) -> torch.Tensor:
    """Existing mixed synthetic-negative objective, retained for compatibility."""
    batch_size = history.shape[0]
    if batch_size < 2:
        return history.new_zeros(())

    sigma = z_next.new_full((batch_size,), float(noise_level))
    positive_energy = model.energy(
        history, action, z_next, noise_level=sigma, reduction="none"
    )
    negative_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_shuffles = min(num_action_shuffles, batch_size - 1)
    for shift in range(1, max_shuffles + 1):
        negative_pairs.append((action.roll(shifts=shift, dims=0), z_next))

    action_std = action.std(dim=0, keepdim=True).clamp_min(1e-3)
    negative_pairs.extend(
        [
            (torch.zeros_like(action), z_next),
            (-action, z_next),
            (torch.randn_like(action) * action_std, z_next),
            (action, z_next.roll(shifts=1, dims=0)),
        ]
    )
    for scale in latent_noise_scales:
        negative_pairs.append(
            (action, z_next + float(scale) * torch.randn_like(z_next))
        )

    negative_energy = torch.stack(
        [
            model.energy(
                history, neg_action, neg_next, noise_level=sigma, reduction="none"
            )
            for neg_action, neg_next in negative_pairs
        ],
        dim=1,
    )
    all_energies = torch.cat((positive_energy.unsqueeze(1), negative_energy), dim=1)
    logits = -all_energies / temperature
    targets = torch.zeros(batch_size, device=history.device, dtype=torch.long)
    info_nce = F.cross_entropy(logits, targets)
    hardest_negative = negative_energy.min(dim=1).values
    hard_ranking = (
        temperature
        * F.softplus(
            (ranking_margin + positive_energy - hardest_negative) / temperature
        ).mean()
    )
    calibration = (
        F.softplus(positive_energy).mean()
        + F.softplus(calibration_margin - negative_energy).mean()
    )
    loss = (
        info_nce
        + hard_negative_weight * hard_ranking
        + calibration_weight * calibration
    )

    if debug:
        print(
            f"E_pos={positive_energy.mean().item():.6f} "
            f"E_neg={negative_energy.mean().item():.6f} "
            f"gap={(negative_energy - positive_energy.unsqueeze(1)).mean().item():.6f} "
            f"E_hard={hardest_negative.mean().item():.6f} "
            f"nce={info_nce.item():.6f} hard={hard_ranking.item():.6f} "
            f"cal={calibration.item():.6f} K={negative_energy.shape[1]}"
        )
    return loss
