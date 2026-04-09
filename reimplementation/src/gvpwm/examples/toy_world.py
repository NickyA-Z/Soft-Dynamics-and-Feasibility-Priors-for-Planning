from __future__ import annotations

import torch

from ..interfaces import WorldModelAdapter


class ToyLinearWorldModel(WorldModelAdapter):
    def __init__(
        self,
        action_limit: float = 0.15,
    ) -> None:
        super().__init__(
            history_length=1,
            action_dim=2,
            action_low=-action_limit,
            action_high=action_limit,
        )
        self.register_buffer("A", torch.eye(2))
        self.register_buffer("B", torch.eye(2))

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device)

    def encode_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(observations, dtype=torch.float32, device=self.device)

    def predict_next_latent(
        self,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        state = latent_history[-1]
        action = action_history[-1]
        return self.A @ state + self.B @ action


class ToyPointMassEnv:
    def __init__(self, start: torch.Tensor) -> None:
        self.state = torch.as_tensor(start, dtype=torch.float32).clone()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.state = self.state + action.cpu()
        return self.state.clone()


def make_infeasible_video_plan(
    start: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, horizon + 1).unsqueeze(-1)
    plan = start.unsqueeze(0) + alphas * (goal - start).unsqueeze(0)
    midpoint = horizon // 2
    plan[midpoint] = plan[midpoint] + torch.tensor([0.4, -0.3], dtype=plan.dtype)
    plan[midpoint + 1 :] = plan[midpoint + 1 :] * 1.15
    return plan
