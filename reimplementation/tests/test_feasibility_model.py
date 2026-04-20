import torch

from gvpwm.feasibility_model.model import FeasibilityModel, ResidualFeasibilityModel


def test_feasibility_model_forward_returns_latent_sized_output():
    model = FeasibilityModel(
        latent_dim=4,
        action_dim=2,
        history_length=3,
        hidden_dim=16,
        num_layers=2,
    )
    history = torch.randn(5, 3, 4)
    action = torch.randn(5, 2)
    z_noisy = torch.randn(5, 4)
    noise_level = torch.full((5,), 0.1)

    output = model(history, action, z_noisy, noise_level)

    assert output.shape == (5, 4)


def test_penalty_matches_squared_noise_norm():
    model = FeasibilityModel(
        latent_dim=3,
        action_dim=2,
        history_length=2,
        hidden_dim=8,
        num_layers=1,
    )
    history = torch.randn(4, 2, 3)
    action = torch.randn(4, 2)
    z_noisy = torch.randn(4, 3)
    noise_level = torch.full((4,), 0.2)

    predicted_noise = model(history, action, z_noisy, noise_level)
    penalty = model.penalty(history, action, z_noisy, noise_level, reduction="none")

    assert torch.allclose(penalty, predicted_noise.pow(2).sum(dim=-1))


def test_dsm_loss_is_non_negative():
    model = FeasibilityModel(
        latent_dim=2,
        action_dim=1,
        history_length=2,
        hidden_dim=8,
        num_layers=2,
    )
    history = torch.randn(6, 2, 2)
    action = torch.randn(6, 1)
    target = torch.randn(6, 2)
    noise_level = torch.full((6,), 0.05)

    loss = model.dsm_loss(history, action, target, noise_level)

    assert loss.ndim == 0
    assert loss >= 0


def test_residual_feasibility_model_uses_world_model_residual_target():
    model = ResidualFeasibilityModel(
        latent_dim=2,
        action_dim=1,
        history_length=1,
        hidden_dim=8,
        num_layers=1,
    )
    predicted_next = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
    true_next = torch.tensor([[0.5, 0.0], [0.1, 0.9]])

    residual = model.residual_target(predicted_next, true_next)

    assert torch.allclose(residual, true_next - predicted_next)
