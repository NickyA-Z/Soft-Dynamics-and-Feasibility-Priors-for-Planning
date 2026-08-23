from __future__ import annotations

import torch
from torch import nn

from codex.training.losses import multi_action_ranking_loss
from codex.training.mining import local_action_negatives, mine_adversarial_actions


class QuadraticEnergy(nn.Module):
    def energy(self, history, action, z_next, noise_level, reduction="none"):
        target = z_next[..., : action.shape[-1]]
        values = (action - target).pow(2).mean(dim=-1)
        return values.mean() if reduction == "mean" else values


def test_local_negatives_respect_coordinate_bounds() -> None:
    action = torch.zeros(4, 2)
    low, high = torch.tensor([-0.1, -2.0]), torch.tensor([0.1, 3.0])
    negatives = local_action_negatives(action, low, high, (0.5, 2.0))
    assert negatives.shape == (4, 2, 2)
    assert torch.all(negatives >= low) and torch.all(negatives <= high)


def test_adversarial_mining_fixes_transition_and_lowers_energy() -> None:
    torch.manual_seed(1)
    model = QuadraticEnergy()
    history = torch.zeros(3, 2, 3)
    next_latent = torch.tensor([[0.7, -0.4, 9.0]]).expand(3, -1)
    expert = next_latent[:, :2].clone()
    low, high = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    mined = mine_adversarial_actions(
        model, history, expert, next_latent, low, high,
        noise_level=0.2, starts=3, steps=40, learning_rate=0.1,
        minimum_rms_distance=0.0, distance_penalty=0.0,
    )
    assert torch.mean((mined - expert).pow(2)) < 0.02


def test_multi_action_loss_has_model_gradient() -> None:
    class ScaledEnergy(QuadraticEnergy):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def energy(self, *args, **kwargs):
            return self.scale * super().energy(*args, **kwargs)

    model = ScaledEnergy()
    history = torch.zeros(2, 1, 3)
    positive = torch.zeros(2, 2)
    next_latent = torch.zeros(2, 3)
    negatives = torch.ones(2, 2, 2)
    loss, _ = multi_action_ranking_loss(
        model, history, positive, next_latent, negatives,
        noise_level=0.2, margin=0.2, temperature=0.1,
    )
    loss.backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
