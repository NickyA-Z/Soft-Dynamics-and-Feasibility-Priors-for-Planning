import torch

from gvpwm.config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from gvpwm.examples.toy_world import ToyLinearWorldModel, ToyPointMassEnv, make_infeasible_video_plan
from gvpwm.planner import GVPWMPlanner
from gvpwm.video import PrecomputedVideoPlanSource


def make_planner(horizon: int) -> GVPWMPlanner:
    return GVPWMPlanner(
        world_model=ToyLinearWorldModel(action_limit=0.15),
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=60,
                outer_steps=8,
                learning_rate=0.08,
                lambda_video=0.5,
                lambda_goal=25.0,
                lambda_action=0.1,
                rho_init=1.0,
                rho_growth=1.5,
                rho_max=250.0,
            ),
            mpc=MPCConfig(horizon=horizon, execution_stride=1, warm_start=True),
            refinement=RefinementConfig(enabled=True, num_samples=32, noise_variance=0.02 ** 2),
        ),
    )


def test_solve_once_hits_goal_on_toy_linear_system():
    torch.manual_seed(0)
    horizon = 8
    start = torch.tensor([0.0, 0.0], dtype=torch.float32)
    goal = torch.tensor([0.8, 0.8], dtype=torch.float32)
    planner = make_planner(horizon)
    video_plan = make_infeasible_video_plan(start, goal, horizon)
    result = planner.solve_once(
        observation_history=start.unsqueeze(0),
        goal_observation=goal,
        video_plan=PrecomputedVideoPlanSource(video_plan, encoded=False).generate(start, goal, horizon),
    )
    rollout = planner.world_model.rollout(
        latent_context=start.unsqueeze(0),
        past_action_context=torch.zeros(0, planner.world_model.action_dim),
        planned_actions=result.actions,
    )
    assert torch.norm(rollout[-1] - goal) < 0.15
    assert result.dynamics_residual_norm < 1e-2


def test_mpc_executes_full_horizon_on_toy_environment():
    torch.manual_seed(0)
    horizon = 10
    start = torch.tensor([0.0, 0.0], dtype=torch.float32)
    goal = torch.tensor([1.0, 1.0], dtype=torch.float32)
    env = ToyPointMassEnv(start=start)
    planner = make_planner(horizon)
    video_plan = make_infeasible_video_plan(start, goal, horizon)
    result = planner.run_mpc(
        observation_history=start.unsqueeze(0),
        goal_observation=goal,
        step_fn=env.step,
        video_source=PrecomputedVideoPlanSource(video_plan, encoded=False),
    )
    assert result.executed_actions.shape[0] == horizon
    assert torch.norm(result.executed_latents[-1] - goal) < 0.2
