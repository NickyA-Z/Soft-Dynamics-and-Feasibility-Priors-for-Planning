"""
Tests for DinoWorldModelAdapter.rollout action history fix.

Before the fix, rollout used a_hist[-(history_length-1):], giving T=H-1
actions when predict_next_latent expected T=H (matching state history).
This caused a tensor size mismatch in _add_action_conditioning (torch.cat dim=3).

After the fix, a_hist[-history_length:] gives T=H for both state and action,
preventing the crash and correctly matching the model's expected input shape.
"""
from __future__ import annotations

import torch
import pytest

from gvpwm.adapters.dino_wm import DinoWorldModelAdapter


# ---------------------------------------------------------------------------
# Minimal fake DINO-style model that records action history T per call
# ---------------------------------------------------------------------------

class _FakePatchEmbed:
    def __init__(self, in_channels):
        self.in_channels = in_channels


class _FakeActionEncoder:
    def __init__(self, action_dim):
        self.patch_embed = _FakePatchEmbed(action_dim)


class RecordingFakeDinoModel(torch.nn.Module):
    """
    Simulates just enough of the DINO world model interface for
    DinoWorldModelAdapter to construct and run rollout.

    Records the action-history T dimension passed to encode_act,
    so tests can assert it equals history_length after the fix.
    """

    def __init__(
        self,
        num_patches: int = 3,
        emb_dim: int = 4,
        action_dim: int = 4,
        history_length: int = 3,
    ):
        super().__init__()
        self.num_hist = history_length
        self.concat_dim = 3           # concat along feature dim (not token dim)
        self.num_action_repeat = 1
        self.num_proprio_repeat = 1
        self._num_patches = num_patches
        self._emb_dim = emb_dim
        self.action_dim = action_dim
        # attribute needed by demo to compute wm_action_dim
        self.action_encoder = _FakeActionEncoder(action_dim)
        # record T dim of every encode_act call
        self.action_T_recorded: list[int] = []

    def encode_act(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, action_dim)
        self.action_T_recorded.append(x.shape[1])
        return x  # identity

    def encode_obs(self, obs_dict: dict) -> dict:
        visual = obs_dict["visual"]   # (B, T, C, H, W)
        B, T = visual.shape[:2]
        return {
            "visual": torch.zeros(B, T, self._num_patches, self._emb_dim),
            "proprio": torch.zeros(B, T, 1, 0),  # zero-width proprio
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, num_patches, emb_dim + action_dim)  → identity
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter(history_length: int = 3, action_dim: int = 4) -> DinoWorldModelAdapter:
    fake_model = RecordingFakeDinoModel(
        num_patches=3,
        emb_dim=4,
        action_dim=action_dim,
        history_length=history_length,
    )
    return DinoWorldModelAdapter(
        world_model=fake_model,
        action_dim=action_dim,
        action_low=-1.0,
        action_high=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rollout_does_not_crash_with_history_length_3():
    """
    With history_length=3, rollout must pass T=3 actions to encode_act.
    Before the fix this would crash with a torch.cat size mismatch.
    """
    history_length = 3
    action_dim = 4
    num_patches = 3
    emb_dim = 4
    horizon = 4

    adapter = make_adapter(history_length=history_length, action_dim=action_dim)

    latent_context = torch.zeros(history_length, num_patches, emb_dim)
    past_action_context = torch.zeros(history_length - 1, action_dim)
    planned_actions = torch.randn(horizon, action_dim) * 0.1

    # Should NOT raise "Sizes of tensors must match except in dimension 3"
    result = adapter.rollout(
        latent_context=latent_context,
        past_action_context=past_action_context,
        planned_actions=planned_actions,
    )
    # DinoWorldModelAdapter.rollout returns [initial] + [predicted * horizon]
    assert result.shape[0] == horizon + 1


def test_rollout_action_T_equals_history_length():
    """
    After the fix, every call to encode_act must receive T == history_length.
    """
    history_length = 3
    action_dim = 4
    horizon = 5

    adapter = make_adapter(history_length=history_length, action_dim=action_dim)
    fake_model = adapter.world_model

    latent_context = torch.zeros(history_length, 3, 4)
    past_action_context = torch.zeros(history_length - 1, action_dim)
    planned_actions = torch.randn(horizon, action_dim) * 0.1

    adapter.rollout(
        latent_context=latent_context,
        past_action_context=past_action_context,
        planned_actions=planned_actions,
    )

    assert len(fake_model.action_T_recorded) == horizon, (
        f"encode_act should be called once per horizon step, "
        f"got {len(fake_model.action_T_recorded)}"
    )
    for step, T in enumerate(fake_model.action_T_recorded):
        assert T == history_length, (
            f"Step {step}: encode_act received T={T}, expected T={history_length}. "
            "Action history length should equal state history length after the bug fix."
        )


def test_rollout_output_shape_matches_horizon():
    """rollout returns initial frame + horizon predicted latents = horizon+1 total."""
    adapter = make_adapter(history_length=2, action_dim=4)
    horizon = 6

    result = adapter.rollout(
        latent_context=torch.zeros(2, 3, 4),
        past_action_context=torch.zeros(1, 4),
        planned_actions=torch.zeros(horizon, 4),
    )
    assert result.shape[0] == horizon + 1


def test_rollout_with_history_length_1_does_not_crash():
    """Edge case: history_length=1 should also work without action history growing unbounded."""
    adapter = make_adapter(history_length=1, action_dim=4)
    horizon = 3

    result = adapter.rollout(
        latent_context=torch.zeros(1, 3, 4),
        past_action_context=torch.zeros(0, 4),
        planned_actions=torch.zeros(horizon, 4),
    )
    assert result.shape[0] == horizon + 1
