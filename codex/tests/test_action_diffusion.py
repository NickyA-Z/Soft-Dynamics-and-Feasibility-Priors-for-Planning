from __future__ import annotations

import torch

from extension.feasibility2.action_diffusion import sample_actions
from extension.feasibility2.model import TransformerFeasibilityModel
from extension.feasibility2.scheduler import LogUniformSigmaScheduler


def make_model() -> TransformerFeasibilityModel:
    return TransformerFeasibilityModel(
        action_dim=2,
        latent_shape=(3, 4),
        history_length=2,
        model_dim=16,
        num_layers=1,
        num_heads=4,
    )


def test_action_dsm_backpropagates_to_action_head() -> None:
    torch.manual_seed(0)
    model = make_model()
    history = torch.randn(5, 2, 3, 4)
    action = torch.randn(5, 2)
    desired_next = torch.randn(5, 3, 4)
    loss = model.action_dsm_loss(
        history,
        action,
        desired_next,
        scheduler=LogUniformSigmaScheduler(0.01, 0.5),
    )
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.action_out_proj.weight.grad is not None
    assert torch.isfinite(model.action_out_proj.weight.grad).all()


def test_action_sampling_has_expected_shape_and_bounds() -> None:
    torch.manual_seed(1)
    model = make_model().eval()
    history = torch.randn(2, 3, 4)
    desired_next = torch.randn(3, 4)
    low = torch.tensor([-0.5, -1.0])
    high = torch.tensor([0.75, 1.5])
    actions = sample_actions(
        model,
        history,
        desired_next,
        low,
        high,
        num_samples=4,
        num_steps=5,
        sigma_min=0.01,
        sigma_max=0.5,
    )
    assert actions.shape == (4, 2)
    assert torch.all(actions >= low)
    assert torch.all(actions <= high)


def test_old_checkpoint_can_omit_action_head(tmp_path) -> None:
    model = make_model()
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("action_out_proj.")
    }
    path = tmp_path / "old.pt"
    torch.save(
        {"model_state_dict": state, "config": model.config_dict()}, path
    )
    loaded = TransformerFeasibilityModel.from_checkpoint(path)
    assert hasattr(loaded, "action_out_proj")
