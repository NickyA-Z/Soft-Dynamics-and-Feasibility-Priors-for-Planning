from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import gym
import numpy as np
import torch
from omegaconf import OmegaConf

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD


DATA_ROOT = DINO_WM_ROOT / "data" / "pusht_noise"
MODEL_NAME = "pusht"
FRAME_SKIP = 5

DEFAULT_HORIZON = 25
DEFAULT_SPLIT = "val"
DEFAULT_NUM_EPISODES = 50


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
    primitive_actions = (action_np * action_std.detach().cpu().numpy()) + action_mean.detach().cpu().numpy()

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


def build_planner(world_model: DinoWorldModelAdapter, horizon: int) -> GVPWMPlanner:
    lambda_action = 0.05 if horizon == 25 else 0.1
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=25,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1_000.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=lambda_action,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
                use_action_reparameterization=True,
            ),
            mpc=MPCConfig(
                horizon=horizon,
                execution_stride=1,
                warm_start=True,
            ),
            refinement=RefinementConfig(
                enabled=True,
                num_samples=500,
                noise_std=0.3,
            ),
        ),
    )


def candidate_episodes(split_dir: Path, horizon: int, frame_skip: int) -> list[int]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = pickle.load(handle)
    min_required_length = horizon * frame_skip + 1
    return [idx for idx, length in enumerate(seq_lengths) if int(length) >= min_required_length]


def evaluate_episode(
    episode_idx: int,
    split: str,
    horizon: int,
    model=None,
    model_cfg=None,
    device=None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    torch.manual_seed(episode_idx)

    split_dir = DATA_ROOT / split
    episode = load_oracle_episode(split_dir, episode_idx)
    episode = slice_oracle_episode(episode, horizon=horizon, frame_skip=FRAME_SKIP)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]

    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = episode["velocities"][0].detach().cpu().numpy()
    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)

    primitive_action_dim = int(env.action_space.shape[0])
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)
    rel_actions = torch.load(split_dir / "rel_actions.pth").float()
    rel_actions = rel_actions / 100.0
    rel_actions = (rel_actions - ACTION_MEAN) / ACTION_STD
    primitive_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32)
    primitive_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    planner = build_planner(world_model=world_model, horizon=horizon)

    print(
        f"starting ALM planner "
        f"(split={split}, horizon={horizon}, I=25, O=25, gamma=1.9, refinement=500x0.3)"
    )
    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        ),
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
    )
    print("ALM planner finished")

    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0
    expert_rel_norm = (episode["rel_actions"][:FRAME_SKIP].float() / 100.0 - action_mean.cpu()) / action_std.cpu()

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print("first 5 expert actions:", episode["actions"][:FRAME_SKIP])
    print("first 5 expert relative actions (normalized):", expert_rel_norm)

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
    print(
        f"[ep {episode_idx}] split={split} horizon={horizon} "
        f"success={metrics['success']} state_dist={metrics['state_dist']:.2f} "
        f"dyn_residual={result.steps[-1].dynamics_residual_norm:.4f}"
    )

    return {
        "episode_idx": episode_idx,
        "split": split,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GVP-WM Oracle evaluation with ALM latent collocation")
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--horizon", type=int, choices=(25, 50, 80), default=DEFAULT_HORIZON, help="Planning horizon")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the filtered eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of filtered episodes to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=args.horizon, frame_skip=FRAME_SKIP)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]

    print(f"Evaluating split={args.split} horizon={args.horizon}")
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode ids: {eval_episodes}")

    results = []
    for idx in eval_episodes:
        print(f"\n=== Episode {idx} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    split=args.split,
                    horizon=args.horizon,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback

            print(f"[ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Evaluation Summary ===")
    for result in results:
        print(result)
    if results:
        success_rate = sum(result["success"] for result in results) / len(results)
        mean_dist = sum(result["state_dist"] for result in results) / len(results)
        mean_dyn = sum(result["dynamics_residual"] for result in results) / len(results)
        print(
            f"Success rate: {success_rate:.3f}  "
            f"mean_state_dist: {mean_dist:.2f}  "
            f"mean_dyn_res: {mean_dyn:.4f}"
        )


if __name__ == "__main__":
    main()
