from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gym
import numpy as np
import torch

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..video import PrecomputedVideoPlanSource
from .dino_oracle_demo import (
    DATA_ROOT as PUSHT_DATA_ROOT,
    build_planner as build_pusht_planner,
    load_model_once as load_pusht_model_once,
    step_env as step_pusht_env,
)
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode
from .dino_wall_oracle_demo import (
    build_planner as build_wall_planner,
    load_model_once as load_wall_model_once,
    step_env as step_wall_env,
)
from .dino_wall_oracle_utils import (
    compute_wall_stats,
    load_wall_oracle_episode,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)
from .wan0s_utils import load_wan_video_plan


def _macro_horizon(raw_horizon: int, frame_skip: int) -> int:
    if raw_horizon % frame_skip != 0:
        raise ValueError(f"raw_horizon={raw_horizon} must be divisible by frame_skip={frame_skip}")
    return raw_horizon // frame_skip


def _parse_episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _prepare_wall_env(env: gym.Env, episode: dict[str, Any], episode_idx: int) -> None:
    env.unwrapped.update_env(episode["env_info"])
    init_state = episode["states"][0].detach().cpu().numpy()
    env.unwrapped.seed(episode_idx)
    env.unwrapped.set_init_state(init_state)
    env.reset()


def _current_wall_state(env: gym.Env) -> np.ndarray:
    return env.unwrapped.dot_position.detach().cpu().numpy().astype(np.float32)


def _wall_episode_video_path(video_root: Path, episode_idx: int) -> Path:
    return video_root / "wall" / f"episode_{episode_idx:03d}" / "wan0s.mp4"


def _pusht_episode_video_path(video_root: Path, episode_idx: int) -> Path:
    return video_root / "pusht" / f"episode_{episode_idx:03d}" / "wan0s.mp4"


def evaluate_wall(args: argparse.Namespace, episode_idx: int, model, model_cfg, device: torch.device) -> dict:
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    data_dir = resolve_wall_data_dir(args.data_root, args.split)
    stats = compute_wall_stats(data_dir)
    episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
    episode = slice_wall_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    _prepare_wall_env(env, episode, episode_idx)

    primitive_action_dim = int(getattr(env.unwrapped, "action_dim", 2))
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim
    if action_repeat != args.frame_skip:
        raise ValueError(f"Wall action_repeat={action_repeat} != frame_skip={args.frame_skip}")

    action_low = episode["action_low"].to(device=device, dtype=torch.float32).repeat(action_repeat)
    action_high = episode["action_high"].to(device=device, dtype=torch.float32).repeat(action_repeat)
    action_mean = episode["action_mean"].to(device=device, dtype=torch.float32)
    action_std = episode["action_std"].to(device=device, dtype=torch.float32)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    planner = build_wall_planner(
        world_model=world_model,
        horizon=horizon,
        paper_horizon=args.raw_horizon,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        lambda_action_override=args.lambda_action,
        lambda_goal=args.lambda_goal,
        lambda_video=args.lambda_video,
        residual_reduction=args.residual_reduction,
        use_action_reparameterization=not args.disable_action_reparameterization,
        refinement_samples=args.refinement_samples,
        refinement_variance=args.refinement_variance,
        disable_refinement=args.disable_refinement,
    )

    video_path = _wall_episode_video_path(Path(args.video_root), episode_idx)
    video_plan = load_wan_video_plan(video_path, task="wall", image_size=args.image_size)
    print(f"[wan0s wall ep {episode_idx}] using video={video_path} frames={len(video_plan)}")

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_wall_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=episode["proprio_mean"],
            proprio_std=episode["proprio_std"],
            env_action_scale=args.wall_env_action_scale,
        ),
        video_source=PrecomputedVideoPlanSource(video_plan, encoded=False),
    )
    cur_state = _current_wall_state(env)
    goal_state = episode["states"][-1].detach().cpu().numpy()
    metrics = env.unwrapped.eval_state(goal_state, cur_state)
    return {
        "task": "wall",
        "episode_idx": episode_idx,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "raw_horizon": args.raw_horizon,
        "frame_skip": args.frame_skip,
        "generated_video": str(video_path),
        "generated_frames": len(video_plan),
        "executed_actions": int(result.executed_actions.shape[0]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
    }


def evaluate_pusht(args: argparse.Namespace, episode_idx: int, model, model_cfg, device: torch.device) -> dict:
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    split_dir = Path(args.data_root) / args.split
    episode = load_oracle_episode(split_dir, episode_idx)
    episode = slice_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]
    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = episode["velocities"][0].detach().cpu().numpy()
    env.unwrapped._set_state(np.concatenate([initial_state, initial_velocity], axis=0))

    primitive_action_dim = int(env.action_space.shape[0])
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim
    if action_repeat != args.frame_skip:
        raise ValueError(f"PushT action_repeat={action_repeat} != frame_skip={args.frame_skip}")

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)
    rel_actions = torch.load(split_dir / "rel_actions.pth").float()
    rel_actions = (rel_actions / 100.0 - ACTION_MEAN) / ACTION_STD
    action_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32).repeat(action_repeat)
    action_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32).repeat(action_repeat)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    planner = build_pusht_planner(
        world_model=world_model,
        horizon=horizon,
        paper_horizon=args.raw_horizon,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        lambda_action_override=args.lambda_action,
        lambda_goal=args.lambda_goal,
        lambda_video=args.lambda_video,
        residual_reduction=args.residual_reduction,
        use_action_reparameterization=not args.disable_action_reparameterization,
        refinement_samples=args.refinement_samples,
        refinement_variance=args.refinement_variance,
        disable_refinement=args.disable_refinement,
    )

    video_path = _pusht_episode_video_path(Path(args.video_root), episode_idx)
    video_plan = load_wan_video_plan(video_path, task="pusht", image_size=args.image_size)
    print(f"[wan0s pusht ep {episode_idx}] using video={video_path} frames={len(video_plan)}")

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_pusht_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        ),
        video_source=PrecomputedVideoPlanSource(video_plan, encoded=False),
    )

    goal_state = np.concatenate(
        [episode["states"][-1].detach().cpu().numpy(), episode["velocities"][-1].detach().cpu().numpy()],
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
    block_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
    angle_diff = abs(
        (float(goal_state[4]) - float(cur_state[4]) + np.pi) % (2 * np.pi) - np.pi
    )
    success = block_diff < 20 and angle_diff < np.pi / 9
    return {
        "task": "pusht",
        "episode_idx": episode_idx,
        "success": bool(success),
        "block_diff": float(block_diff),
        "angle_diff": float(angle_diff),
        "raw_horizon": args.raw_horizon,
        "frame_skip": args.frame_skip,
        "generated_video": str(video_path),
        "generated_frames": len(video_plan),
        "executed_actions": int(result.executed_actions.shape[0]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GVP-WM with WAN-0S generated video plans.")
    parser.add_argument("--task", choices=("wall", "pusht"), required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--raw-horizon", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--video-root", default=os.environ.get("WAN0S_VIDEO_ROOT", "wan0s_videos"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--outer-steps", type=int, default=25)
    parser.add_argument("--lambda-action", type=float, default=None)
    parser.add_argument("--lambda-goal", type=float, default=10.0)
    parser.add_argument("--lambda-video", type=float, default=1.0)
    parser.add_argument("--residual-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--disable-action-reparameterization", action="store_true")
    parser.add_argument(
        "--wall-env-action-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to denormalized Wall actions before env.step.",
    )
    parser.add_argument("--refinement-samples", type=int, default=500)
    parser.add_argument("--refinement-variance", type=float, default=0.3)
    parser.add_argument("--disable-refinement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dino_root = Path(os.environ.get("DINO_WM_ROOT", Path.home() / "DL2---Grounding-Generated-Videos-" / "dino_wm"))
    if args.data_root is None:
        if args.task == "wall":
            args.data_root = str(dino_root / "data" / "wall_single")
        else:
            args.data_root = str(PUSHT_DATA_ROOT)
            if args.split == "all":
                args.split = "val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "wall":
        model, model_cfg = load_wall_model_once(device, model_name="wall_single")
        evaluator = evaluate_wall
    else:
        model, model_cfg = load_pusht_model_once(device)
        evaluator = evaluate_pusht

    results = []
    for episode_idx in _parse_episode_ids(args.episode_ids):
        try:
            result = evaluator(args, episode_idx, model=model, model_cfg=model_cfg, device=device)
            print(f"[wan0s result] {result}")
            results.append(result)
        except Exception as exc:
            import traceback

            print(f"[wan0s {args.task} ep {episode_idx}] ERROR: {exc}")
            traceback.print_exc()
            results.append({"task": args.task, "episode_idx": episode_idx, "error": str(exc)})

    successes = [item["success"] for item in results if "success" in item]
    summary = {
        "task": args.task,
        "num_results": len(results),
        "num_success": int(sum(successes)),
        "success_rate": float(sum(successes) / len(successes)) if successes else None,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
