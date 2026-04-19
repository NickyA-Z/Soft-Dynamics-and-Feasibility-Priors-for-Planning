from __future__ import annotations

import torch

from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from .toy_world import ToyLinearWorldModel, ToyPointMassEnv, make_infeasible_video_plan

from pathlib import Path
from omegaconf import OmegaConf

import gym

import sys
sys.path.append("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")

from plan import load_model

from ..adapters.dino_wm import DinoWorldModelAdapter


def main() -> None:
    torch.manual_seed(0)
    start = torch.tensor([0.0, 0.0], dtype=torch.float32)
    goal = torch.tensor([1.0, 1.0], dtype=torch.float32)
    horizon = 10

    #------------------load word model-------------------#
    ckpt_base_path = "/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/checkpoints"
    model_name = "pusht"
    model_path = f"{ckpt_base_path}/outputs/{model_name}"
    model_cfg = OmegaConf.load(f"{model_path}/hydra.yaml")
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device,)
    world_model = DinoWorldModelAdapter(world_model=model, action_dim=model_cfg.action_emb_dim,)
    #------------------load world model end--------------#
    #-------------------environment------------------------#
    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    obs, _ = env.reset()
    #print(type(obs))
    #print(obs.keys() if isinstance(obs, dict) else obs)
    #print(obs["visual"].shape)
    start_obs = {
        "visual": torch.tensor(obs["visual"]).permute(2, 0, 1).float() / 255.0,
        "proprio": torch.tensor(obs["proprio"]).float()
    }
    trajectory = [start_obs]
    for _ in range(horizon):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        obs = {
            "visual": torch.tensor(obs["visual"]).permute(2, 0, 1).float() / 255.0,
            "proprio": torch.tensor(obs["proprio"]).float()
        }
        trajectory.append(obs)
    goal_obs = trajectory[-1]
    obs, _ = env.reset()
    start_obs = {
        "visual": torch.tensor(obs["visual"]).permute(2, 0, 1).float() / 255.0,
        "proprio": torch.tensor(obs["proprio"]).float()
    }
    
    #--------------------environment end-------------------#
    #--------------------toy video plan-------------------------#
    video_plan = trajectory
    #--------------------toy video plan end---------------------#
    
    planner = GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            
            #alm=ALMConfig(
            #    inner_steps=60,
            #    outer_steps=10,
            #    learning_rate=0.08,
            #    lambda_video=0.5,
            #    lambda_goal=25.0,
            #    lambda_action=0.1,
            #    rho_init=1.0,
            #    rho_growth=1.5,
            #    rho_max=250.0,
            #),
            
            alm=ALMConfig(
                inner_steps=5,
                outer_steps=2,
                learning_rate=0.08,
                lambda_video=0.5,
                lambda_goal=10.0,
                lambda_action=0.1,
                rho_init=1.0,
                rho_growth=1.5,
                rho_max=50.0,
            ),
            mpc=MPCConfig(horizon=horizon, execution_stride=1, warm_start=True),
            refinement=RefinementConfig(enabled=True, num_samples=64, noise_variance=0.03 ** 2),
        ),
    )
    def step_fn(action):
        action_np = action.detach().cpu().numpy()
        obs, _, _, _ = env.step(action_np)
        return {
            "visual": torch.tensor(obs["visual"]).permute(2, 0, 1).float() / 255.0,
            "proprio": torch.tensor(obs["proprio"]).float()
        }
    result = planner.run_mpc(
        observation_history=[start_obs],
        goal_observation=goal_obs,
        step_fn=step_fn,
        video_source=PrecomputedVideoPlanSource(video_plan, encoded=False),
    )

    final_state = result.executed_latents[-1]
    #goal_error = torch.norm(final_state - goal).item()
    print("Toy GVP-WM demo")
    print(f"Executed {result.executed_actions.shape[0]} actions")
    print(f"Final state: {final_state.tolist()}")
    print(f"Goal: {goal.tolist()}")
    #print(f"Goal error: {goal_error:.4f}")


if __name__ == "__main__":
    main()
