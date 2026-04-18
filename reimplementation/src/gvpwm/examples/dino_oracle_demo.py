from __future__ import annotations

import sys
from pathlib import Path

import gym
import torch
import numpy as np
from omegaconf import OmegaConf

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from .dino_oracle_utils import infer_horizon, load_oracle_episode


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD


DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"
MAX_HORIZON = None


def to_runtime_observation(obs):
    return {
        "visual": torch.as_tensor(obs["visual"], dtype=torch.float32).permute(2, 0, 1) / 255.0,
        "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32)[..., :4],
    }


def step_env(
    env: gym.Env,
    action: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_np = action.detach().cpu().numpy().reshape(action_repeat, primitive_action_dim)

    mean_np = action_mean.detach().cpu().numpy()
    std_np = action_std.detach().cpu().numpy()

    # inverse of dataset normalization only
    primitive_actions = (action_np * std_np) + mean_np

    obs = None
    for primitive_action in primitive_actions:
        step_out = env.step(primitive_action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated, truncated = done, False

        if terminated or truncated:
            break

    return to_runtime_observation(obs)






def load_model_once(device):
    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    return model, model_cfg


def evaluate_episode(episode_idx: int, model=None, model_cfg=None, device=None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    torch.manual_seed(episode_idx)

    episode = load_oracle_episode(DATA_DIR, episode_idx)
    frame_skip = 5
    horizon = infer_horizon(episode["length"], frame_skip=frame_skip, max_horizon=MAX_HORIZON)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    velocities = torch.load("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data/pusht_noise/train/velocities.pth")

    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = velocities[episode_idx, 0].detach().cpu().numpy()

    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)
    


    primitive_action_dim = int(env.action_space.shape[0])   # 2
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)  # 10
    action_repeat = wm_action_dim // primitive_action_dim   # 5

    primitive_low_raw = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    primitive_high_raw = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)

    primitive_low = torch.full((primitive_action_dim,), -3.0, device=device, dtype=torch.float32)
    primitive_high = torch.full((primitive_action_dim,), 3.0, device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)


    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    
    # smaller config for quick demo runs
    planner = GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=5,
                outer_steps=2,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.5,
                rho_max=50.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=0.05,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
            ),
            mpc=MPCConfig(
                horizon=horizon,
                execution_stride=1,
                warm_start=True,
            ),
            refinement=RefinementConfig(
                enabled=False,
                num_samples=0,
                noise_std=0.0,
            ),
        ),
    )


    """ full demo run with planner and environment interaction
    planner = GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=10,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1000.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=0.05,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
            ),
            mpc=MPCConfig(
                horizon=horizon,
                execution_stride=1,
                warm_start=True,
            ),
            refinement=RefinementConfig(
                enabled=True,
                num_samples=100,
                noise_std=0.3,
            ),
        ),
    )
    """
    

    print("starting planner")
    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_env(env, action, action_repeat=action_repeat, primitive_action_dim=primitive_action_dim, 
                                        action_mean=action_mean, action_std=action_std,),
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
    )
    print("planner finished")
    
    
    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print("first 5 expert actions:", episode["actions"][:5])
    
    
    goal_state = np.concatenate(
        [
            episode["states"][-1].detach().cpu().numpy(),
            episode["velocities"][-1].detach().cpu().numpy(),
        ],
        axis=0,
    )

    cur_state = np.array(
        [
            env.unwrapped.agent.position[0],
            env.unwrapped.agent.position[1],
            env.unwrapped.block.position[0],
            env.unwrapped.block.position[1],
            env.unwrapped.block.angle,
            env.unwrapped.agent.velocity[0],
            env.unwrapped.agent.velocity[1],
        ],
        dtype=np.float32,
    )

    metrics = env.unwrapped.eval_state(goal_state, cur_state)

    print("eval metrics:", metrics)
    
    
    
    print("agent position:", env.unwrapped.agent.position)
    print("agent velocity:", env.unwrapped.agent.velocity)
    print("block position:", env.unwrapped.block.position)
    print("block angle:", env.unwrapped.block.angle)


    print(f"[ep {episode_idx}] oracle_len={episode['length']} horizon={horizon} "
          f"success={metrics['success']} state_dist={metrics['state_dist']:.2f} "
          f"dyn_residual={result.steps[-1].dynamics_residual_norm:.4f}")

    return {
        "episode_idx": episode_idx,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }
    

EVAL_EPISODES = list(range(10))  # evaluate episodes 0-9
EVAL_EPISODES = list(range(1))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    results = []
    for idx in EVAL_EPISODES:
        print(f"\n=== Episode {idx} ===")
        try:
            r = evaluate_episode(idx, model=model, model_cfg=model_cfg, device=device)
            results.append(r)
        except Exception as exc:
            print(f"[ep {idx}] ERROR: {exc}")

    print("\n=== Evaluation Summary ===")
    for r in results:
        print(r)
    if results:
        success_rate = sum(r["success"] for r in results) / len(results)
        mean_dist = sum(r["state_dist"] for r in results) / len(results)
        mean_res = sum(r["dynamics_residual"] for r in results) / len(results)
        print(f"Success rate: {success_rate:.3f}  mean_state_dist: {mean_dist:.2f}  mean_dyn_res: {mean_res:.4f}")



if __name__ == "__main__":
    main()
