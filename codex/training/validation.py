from __future__ import annotations

import torch
import torch.nn.functional as F

from codex.training.mining import mine_adversarial_actions


def action_recovery_batch(
    model, history, expert, next_latent, low, high, *, noise_level,
    starts, steps, learning_rate, minimum_rms_distance, distance_penalty,
) -> dict[str, float]:
    """Run the same fixed-transition action minimization required by planning."""
    recovered = mine_adversarial_actions(
        model, history, expert, next_latent, low, high,
        noise_level=noise_level, starts=starts, steps=steps,
        learning_rate=learning_rate,
        # Validation reproduces unconstrained planner minimization; unlike
        # training-time hard-negative mining it must be allowed to recover the
        # expert action.
        minimum_rms_distance=0.0,
        distance_penalty=0.0,
    )
    zero = torch.zeros_like(expert).maximum(low).minimum(high)
    with torch.no_grad():
        expert_energy = model.energy(history, expert, next_latent, noise_level, "none")
        zero_energy = model.energy(history, zero, next_latent, noise_level, "none")
        recovered_energy = model.energy(history, recovered, next_latent, noise_level, "none")
        mse_recovered = F.mse_loss(recovered, expert)
        mse_zero = F.mse_loss(zero, expert)
        cosine = F.cosine_similarity(recovered, expert, dim=-1).mean()
    return {
        "recovery_mse": float(mse_recovered),
        "zero_mse": float(mse_zero),
        "recovery_ratio": float(mse_recovered / mse_zero.clamp_min(1e-8)),
        "cosine": float(cosine),
        "expert_beats_zero": float((expert_energy < zero_energy).float().mean()),
        "expert_beats_optimized": float((expert_energy < recovered_energy).float().mean()),
        "energy_gap_optimized_minus_expert": float((recovered_energy - expert_energy).mean()),
    }
