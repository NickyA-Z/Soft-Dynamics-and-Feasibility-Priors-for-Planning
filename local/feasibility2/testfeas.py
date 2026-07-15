import torch
import torch.nn as nn

from nicky_dl.feasibility2.model import load_feasibility_model_from_checkpoint
from reimplementation.src.gvpwm.config import PlannerConfig
from reimplementation.src.gvpwm.solver import LatentCollocationSolver
from nicky_dl.feasibility2.integrate import load_feasibility_scorer, score_transition_feasibility

torch.set_num_threads(1)

"""File to make feasibility model integrate with solver & planner
solver can initialize
feasibility config is passed
feasibility model is called
objective/backward works on CPU
result is returned

from cd DL2---Grounding-Generated-Videos-
run: 
CUDA_VISIBLE_DEVICES="" python -m nicky_dl.feasibility2.feasplan
"""


class DummyWorldModel:
    """Minimal CPU world-model adapter for solver smoke testing."""

    def __init__(self, latent_shape=(196, 394), action_dim=10, history_length=3):
        self.device = torch.device("cpu")
        self.latent_shape = latent_shape
        self.action_dim = action_dim
        self.history_length = history_length

        self.action_low = torch.full((action_dim,), -3.0, dtype=torch.float32)
        self.action_high = torch.full((action_dim,), 3.0, dtype=torch.float32)

    def initialize_latents_from_video(self, current_latent, video_latents):
        latents = video_latents.clone()
        latents[0] = current_latent
        return latents

    def predict_next_latent(self, state_window, action_window):
        # Cheap fake dynamics:
        # predict next latent as the last state plus a tiny action-dependent offset.
        last_state = state_window[-1]

        action_scalar = action_window[-1].mean()
        return last_state + 0.001 * action_scalar

    def video_alignment_loss(self, latent, reference):
        return (latent - reference).pow(2).mean()

    def goal_loss(self, latent, goal):
        return (latent - goal).pow(2).mean()


"""change later with

feasibility_model = load_feasibility_scorer(
    checkpoint_path="checkpoints/sweep_epochs150_old/transformer_noise0_2.pt",
    device=torch.device("cpu"),
)"""

class DummyFeasibilityModel(nn.Module):
    """Minimal feasibility model for checking solver integration."""

    def __init__(self, latent_shape=(196, 394)):
        super().__init__()
        self.latent_shape = latent_shape
        self.noise_level = 0.1

    def penalty(
        self,
        history,
        action,
        z_next,
        noise_level=None,
        reduction="mean",
    ):
        # Cheap fake feasibility energy:
        # penalize distance between proposed next latent and last history latent.
        if history.ndim == 3:
            # single sample: [H, 196, 394]
            last = history[-1]
            energy = (z_next - last).pow(2).mean().reshape(1)
        else:
            # batched: [B, H, 196, 394]
            last = history[:, -1]
            energy = (z_next - last).reshape(z_next.shape[0], -1).pow(2).mean(dim=-1)

        if reduction == "none":
            return energy
        if reduction == "mean":
            return energy.mean()
        if reduction == "sum":
            return energy.sum()

        raise ValueError(f"Unknown reduction: {reduction}")


def main():
    config = PlannerConfig()

    # Tiny CPU smoke-test settings.
    config.mpc.horizon = 2
    config.alm.inner_steps = 1
    config.alm.outer_steps = 1
    config.alm.learning_rate = 1e-3

    # Enable feasibility integration.
    config.feasibility.enabled = True
    config.feasibility.checkpoint_path = "checkpoints/sweep_epochs150_old/transformer_noise0_2.pt"
    config.feasibility.lambda_feasibility = 1e-6 
    config.feasibility.noise_level = 0.2 # best out of testing??
    config.feasibility.reduction = "mean"
    config.feasibility.detach_model = False

    latent_shape = (196, 394)
    action_dim = 10
    history_length = 3

    world_model = DummyWorldModel(
        latent_shape=latent_shape,
        action_dim=action_dim,
        history_length=history_length,
    )
    """
    feasibility_model = DummyFeasibilityModel(
        latent_shape=latent_shape,
    )
    for param in feasibility_model.parameters():
        param.requires_grad_(False)

    feasibility_model.eval()
    latent_context = torch.randn(history_length, *latent_shape)
    past_action_context = torch.zeros(history_length - 1, action_dim)
    goal_latent = torch.randn(*latent_shape)
    video_latents = torch.randn(config.mpc.horizon + 1, *latent_shape)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    world_model = DummyWorldModel(
        latent_shape=latent_shape,
        action_dim=action_dim,
        history_length=history_length,
    )
    world_model.device = device
    world_model.action_low = world_model.action_low.to(device)
    world_model.action_high = world_model.action_high.to(device)

    feasibility_model = load_feasibility_scorer(
        checkpoint_path=config.feasibility.checkpoint_path,
        device=device,
        freeze=True,
    )

    latent_context = torch.randn(history_length, *latent_shape, device=device)
    past_action_context = torch.zeros(history_length - 1, action_dim, device=device)
    goal_latent = torch.randn(*latent_shape, device=device)
    video_latents = torch.randn(config.mpc.horizon + 1, *latent_shape, device=device)

    collocation_solver = LatentCollocationSolver(
        world_model=world_model,
        config=config.alm,
        feasibility_config=config.feasibility,
        feasibility_model=feasibility_model,
    )

    result = collocation_solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_context,
        goal_latent=goal_latent,
        video_latents=video_latents,
        warm_start_latents=None,
        warm_start_actions=None,
    )

    print("CPU solver smoke test passed")
    print("actions:", result.actions.shape)
    print("latents:", result.latents.shape)
    print("objective:", result.objective)
    print("dynamics_residual_norm:", result.dynamics_residual_norm)

    if hasattr(result, "pieces"):
        print("pieces:", result.pieces)


if __name__ == "__main__":
    main()