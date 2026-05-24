"""
Tests for the paper-aligned oracle planner entry points in dino_oracle_demo.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import torch

from gvpwm.examples.dino_oracle_demo import build_planner, candidate_episodes
from gvpwm.interfaces import WorldModelAdapter


class FakeWorldModel(WorldModelAdapter):
    LATENT_SHAPE = (3, 4)

    def __init__(self, action_dim: int = 4, history_length: int = 2):
        super().__init__(
            history_length=history_length,
            action_dim=action_dim,
            action_low=-1.0,
            action_high=1.0,
        )

    def encode_observation(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.float()
        return torch.zeros(*self.LATENT_SHAPE)

    def predict_next_latent(
        self,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        return latent_history[-1] + action_history[-1].mean()


def test_build_planner_matches_paper_h25_defaults():
    planner = build_planner(FakeWorldModel(), horizon=25)
    assert planner.config.alm.inner_steps == 25
    assert planner.config.alm.outer_steps == 25
    assert abs(planner.config.alm.learning_rate - 0.05) < 1e-9
    assert abs(planner.config.alm.rho_growth - 1.9) < 1e-9
    assert abs(planner.config.alm.lambda_action - 0.05) < 1e-9
    assert planner.config.mpc.execution_stride == 1
    assert planner.config.refinement.num_samples == 500
    assert abs(planner.config.refinement.noise_variance - 0.3) < 1e-9


def test_build_planner_uses_higher_action_regularization_for_long_horizon():
    planner = build_planner(FakeWorldModel(), horizon=50)
    assert abs(planner.config.alm.lambda_action - 0.1) < 1e-9


def test_candidate_episodes_filters_by_minimum_required_length(tmp_path: Path):
    seq_lengths = [126, 100, 251, 80]
    with open(tmp_path / "seq_lengths.pkl", "wb") as handle:
        pickle.dump(seq_lengths, handle)

    eligible = candidate_episodes(tmp_path, horizon=25, frame_skip=5)
    assert eligible == [0, 2]
