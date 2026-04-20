from __future__ import annotations

import torch

from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from .toy_world import ToyLinearWorldModel, ToyPointMassEnv, make_infeasible_video_plan


def main() -> None:
    torch.manual_seed(0)
    start = torch.tensor([0.0, 0.0], dtype=torch.float32)
    goal = torch.tensor([1.0, 1.0], dtype=torch.float32)
    horizon = 10

    world_model = ToyLinearWorldModel(action_limit=0.15)
    env = ToyPointMassEnv(start=start)
    video_plan = make_infeasible_video_plan(start, goal, horizon)
    planner = GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            solver=ALMConfig(
                inner_steps=60,
                outer_steps=10,
                learning_rate=0.08,
                lambda_video=0.5,
                lambda_goal=25.0,
                lambda_action=0.1,
                rho_init=1.0,
                rho_growth=1.5,
                rho_max=250.0,
            ),
            mpc=MPCConfig(horizon=horizon, execution_stride=1, warm_start=True),
            refinement=RefinementConfig(enabled=True, num_samples=64, noise_std=0.03),
        ),
    )
    result = planner.run_mpc(
        observation_history=start.unsqueeze(0),
        goal_observation=goal,
        step_fn=env.step,
        video_source=PrecomputedVideoPlanSource(video_plan, encoded=False),
    )

    final_state = result.executed_latents[-1]
    goal_error = torch.norm(final_state - goal).item()
    print("Toy GVP-WM demo")
    print(f"Executed {result.executed_actions.shape[0]} actions")
    print(f"Final state: {final_state.tolist()}")
    print(f"Goal: {goal.tolist()}")
    print(f"Goal error: {goal_error:.4f}")


if __name__ == "__main__":
    main()
