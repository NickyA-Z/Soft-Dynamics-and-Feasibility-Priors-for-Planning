from __future__ import annotations

import torch
from torch import nn

from codex.models.feasibility import FeasibilityScorer
from codex.optimization.actions import actions_to_raw, raw_to_actions
from codex.objectives.combined import conditioned_history_window


class FakeFeasibility(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("latent_mean", torch.tensor([2.0, 4.0]))
        self.register_buffer("latent_std", torch.tensor([2.0, 4.0]))

    def forward(self, history, action, z_next, noise_level):
        return z_next + action.mean()

    def energy(self, history, action, z_next, noise_level, reduction):
        return (z_next.mean() - action.mean()).pow(2)


def test_latents_are_normalized_exactly_once() -> None:
    scorer = FeasibilityScorer(FakeFeasibility(), "fake.pt")
    raw = torch.tensor([4.0, 8.0])
    assert torch.allclose(scorer.normalize_latents(raw), torch.ones(2))
    expected = (torch.ones(2) + torch.tensor([0.5]).mean()).pow(2).mean()
    actual = scorer.dsm_energy(
        raw.repeat(3, 1), torch.tensor([0.5]), raw,
        noise_level=0.2, mode="paper_clean",
    )
    assert torch.allclose(actual, expected)


def test_action_reparameterization_round_trip() -> None:
    actions = torch.tensor([-1.5, 0.0, 2.0])
    assert torch.allclose(raw_to_actions(actions_to_raw(actions, -3, 3), -3, 3), actions)


def test_mpc_history_uses_real_prefix() -> None:
    prefix = torch.tensor([[1.0], [2.0], [3.0]])
    candidates = torch.tensor([[3.0], [4.0], [5.0]])
    assert torch.equal(conditioned_history_window(candidates, 1, 3, prefix), prefix)
    assert torch.equal(
        conditioned_history_window(candidates, 2, 3, prefix),
        torch.tensor([[2.0], [3.0], [4.0]]),
    )
