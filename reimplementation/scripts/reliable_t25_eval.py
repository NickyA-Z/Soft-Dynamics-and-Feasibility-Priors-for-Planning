from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import gym
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf

from gvpwm.examples.dino_oracle_utils import load_oracle_episode, slice_oracle_episode
from gvpwm.examples.dino_wall_oracle_utils import (
    compute_wall_stats,
    load_wall_oracle_episode,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)


DINO_WM_ROOT = Path(
    os.environ.get("DINO_WM_ROOT", Path.home() / "DL2---Grounding-Generated-Videos-" / "dino_wm")
)
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

import env as _dino_env  # noqa: F401  # registers PushT and Wall gym envs


PUSHT_DATA_ROOT = DINO_WM_ROOT / "data" / "pusht_noise"
WALL_DATA_ROOT = DINO_WM_ROOT / "data" / "wall_single"


def _parse_episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _macro_horizon(raw_horizon: int, frame_skip: int) -> int:
    if raw_horizon % frame_skip != 0:
        raise ValueError(f"raw_horizon={raw_horizon} must be divisible by frame_skip={frame_skip}")
    return raw_horizon // frame_skip


def _load_env_cfg(model_name: str):
    return OmegaConf.load(DINO_WM_ROOT / "checkpoints" / "outputs" / model_name / "hydra.yaml")


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + np.pi) % (2 * np.pi) - np.pi)


def _pusht_state(env: gym.Env) -> np.ndarray:
    return np.array(
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


def _pusht_metrics(env: gym.Env, episode: dict[str, Any]) -> dict[str, Any]:
    goal_state = np.concatenate(
        [episode["states"][-1].detach().cpu().numpy(), episode["velocities"][-1].detach().cpu().numpy()],
        axis=0,
    )
    cur_state = _pusht_state(env)
    block_diff = float(np.linalg.norm(goal_state[2:4] - cur_state[2:4]))
    angle_diff = float(_angle_diff(goal_state[4], cur_state[4]))
    agent_diff = float(np.linalg.norm(goal_state[:2] - cur_state[:2]))
    return {
        "success": bool(block_diff < 20 and angle_diff < np.pi / 9),
        "block_diff": block_diff,
        "angle_diff": angle_diff,
        "agent_diff": agent_diff,
    }


def _prepare_pusht_env(episode: dict[str, Any]) -> gym.Env:
    cfg = _load_env_cfg("pusht")
    env = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]
    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = episode["velocities"][0].detach().cpu().numpy()
    env.unwrapped._set_state(np.concatenate([initial_state, initial_velocity], axis=0))
    return env


def _evaluate_pusht_expert(episode: dict[str, Any]) -> dict[str, Any]:
    env = _prepare_pusht_env(episode)
    for action in (episode["rel_actions"].float() / 100.0).detach().cpu().numpy():
        step_out = env.step(action)
        done = step_out[2] if len(step_out) == 4 else step_out[2] or step_out[3]
        if done:
            break
    result = _pusht_metrics(env, episode)
    result["policy"] = "oracle_expert_replay"
    return result


def _blue_center(frame: np.ndarray) -> np.ndarray | None:
    square = _center_square(frame).astype(np.float32)
    r, g, b = square[..., 0], square[..., 1], square[..., 2]
    mask = (b > 80) & (b > r * 1.15) & (b > g * 1.05)
    return _weighted_center(mask, b - np.maximum(r, g))


def _evaluate_pusht_video_track(episode: dict[str, Any], video_path: Path, gain: float) -> dict[str, Any]:
    frames = iio.imread(video_path)
    centers = _track_centers(frames, _blue_center)
    targets = _calibrate_targets(
        centers=centers,
        start_state=episode["states"][0, :2].detach().cpu().numpy(),
        goal_state=episode["states"][-1, :2].detach().cpu().numpy(),
        fallback_scale=512.0 / 720.0,
    )

    env = _prepare_pusht_env(episode)
    frame_indices = np.linspace(0, len(frames) - 1, episode["frame_skip"] * episode["planning_horizon"] + 1, dtype=int)[1:]
    for frame_idx in frame_indices:
        target = targets[frame_idx]
        current = np.array(env.unwrapped.agent.position, dtype=np.float32)
        action = np.clip((target - current) / 100.0 * gain, -1.0, 1.0)
        env.step(action)
    result = _pusht_metrics(env, episode)
    result["policy"] = "wan0s_blue_agent_track"
    result["generated_video"] = str(video_path)
    return result


def _wall_state(env: gym.Env) -> np.ndarray:
    return env.unwrapped.dot_position.detach().cpu().numpy().astype(np.float32)


def _prepare_wall_env(episode: dict[str, Any], episode_idx: int) -> gym.Env:
    cfg = _load_env_cfg("wall_single")
    env = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
    env.unwrapped.update_env(episode["env_info"])
    init_state = episode["states"][0].detach().cpu().numpy()
    env.unwrapped.seed(episode_idx)
    env.unwrapped.set_init_state(init_state)
    env.reset()
    return env


def _wall_metrics(env: gym.Env, episode: dict[str, Any]) -> dict[str, Any]:
    goal_state = episode["states"][-1].detach().cpu().numpy()
    cur_state = _wall_state(env)
    metrics = env.unwrapped.eval_state(goal_state, cur_state)
    return {
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
    }


def _evaluate_wall_expert(episode: dict[str, Any], episode_idx: int) -> dict[str, Any]:
    env = _prepare_wall_env(episode, episode_idx)
    for action in episode["actions"].float().detach().cpu():
        env.step(action / 2.0)
    result = _wall_metrics(env, episode)
    result["policy"] = "oracle_expert_replay"
    return result


def _evaluate_wall_state_track(
    episode: dict[str, Any],
    episode_idx: int,
    gain: float,
    max_action: float,
) -> dict[str, Any]:
    env = _prepare_wall_env(episode, episode_idx)
    for target in episode["states"][1:].detach().cpu().numpy():
        current = _wall_state(env)
        action = np.clip((target - current) / 2.0 * gain, -max_action, max_action)
        env.step(torch.as_tensor(action, dtype=torch.float32))
    result = _wall_metrics(env, episode)
    result["policy"] = "oracle_state_track"
    return result


def _red_center(frame: np.ndarray) -> np.ndarray | None:
    square = _center_square(frame).astype(np.float32)
    r, g, b = square[..., 0], square[..., 1], square[..., 2]
    mask = (r > 120) & (r > g * 1.2) & (r > b * 1.2)
    return _weighted_center(mask, r - np.maximum(g, b))


def _evaluate_wall_video_track(
    episode: dict[str, Any],
    episode_idx: int,
    video_path: Path,
    gain: float,
    max_action: float,
) -> dict[str, Any]:
    frames = iio.imread(video_path)
    centers = _track_centers(frames, _red_center)
    targets = _calibrate_targets(
        centers=centers,
        start_state=episode["states"][0].detach().cpu().numpy(),
        goal_state=episode["states"][-1].detach().cpu().numpy(),
        fallback_scale=65.0 / 720.0,
    )

    env = _prepare_wall_env(episode, episode_idx)
    raw_horizon = episode["frame_skip"] * episode["planning_horizon"]
    frame_indices = np.linspace(0, len(frames) - 1, raw_horizon + 1, dtype=int)[1:]
    for frame_idx in frame_indices:
        target = targets[frame_idx]
        current = _wall_state(env)
        action = np.clip((target - current) / 2.0 * gain, -max_action, max_action)
        env.step(torch.as_tensor(action, dtype=torch.float32))
    result = _wall_metrics(env, episode)
    result["policy"] = "wan0s_red_dot_track"
    result["generated_video"] = str(video_path)
    return result


def _center_square(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape={frame.shape}")
    height, width = frame.shape[:2]
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    return frame[top : top + crop, left : left + crop]


def _weighted_center(mask: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    selected_weights = weights[mask].clip(1.0)
    total = selected_weights.sum()
    return np.array(
        [
            float((xx * selected_weights).sum() / total),
            float((yy * selected_weights).sum() / total),
        ],
        dtype=np.float32,
    )


def _track_centers(frames: np.ndarray, detector) -> np.ndarray:
    if frames.ndim == 3:
        frames = frames[None]
    centers: list[np.ndarray] = []
    last = None
    for frame in frames:
        center = detector(frame)
        if center is None:
            if last is None:
                raise ValueError("Could not detect the tracked object in the first video frame.")
            center = last
        centers.append(center)
        last = center
    return np.stack(centers, axis=0)


def _calibrate_targets(
    centers: np.ndarray,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    fallback_scale: float,
) -> np.ndarray:
    scale = np.full(2, fallback_scale, dtype=np.float32)
    for axis in range(2):
        pixel_delta = centers[-1, axis] - centers[0, axis]
        if abs(float(pixel_delta)) > 1e-3:
            scale[axis] = (goal_state[axis] - start_state[axis]) / pixel_delta
    offset = start_state.astype(np.float32) - scale * centers[0]
    return centers * scale + offset


def _video_path(video_root: Path, task: str, episode_idx: int) -> Path:
    return video_root / task / f"episode_{episode_idx:03d}" / "wan0s.mp4"


def evaluate(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    if args.task == "pusht":
        split = args.split if args.split != "all" else "val"
        episode = load_oracle_episode(Path(args.data_root) / split, episode_idx)
        episode = slice_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)
        if args.policy == "expert":
            result = _evaluate_pusht_expert(episode)
        else:
            result = _evaluate_pusht_video_track(
                episode,
                video_path=_video_path(Path(args.video_root), "pusht", episode_idx),
                gain=args.video_track_gain,
            )
    else:
        data_dir = resolve_wall_data_dir(args.data_root, args.split)
        stats = compute_wall_stats(data_dir)
        episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
        episode = slice_wall_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)
        if args.policy == "expert":
            result = _evaluate_wall_expert(episode, episode_idx)
        elif args.policy == "state-track":
            result = _evaluate_wall_state_track(
                episode,
                episode_idx=episode_idx,
                gain=args.video_track_gain,
                max_action=args.wall_max_action,
            )
        else:
            result = _evaluate_wall_video_track(
                episode,
                episode_idx=episode_idx,
                video_path=_video_path(Path(args.video_root), "wall", episode_idx),
                gain=args.video_track_gain,
                max_action=args.wall_max_action,
            )
    result.update(
        {
            "task": args.task,
            "episode_idx": episode_idx,
            "raw_horizon": args.raw_horizon,
            "frame_skip": args.frame_skip,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliable T=25 sanity policies for PushT/Wall.")
    parser.add_argument("--task", choices=("pusht", "wall"), required=True)
    parser.add_argument("--policy", choices=("expert", "state-track", "video-track"), required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--raw-horizon", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--video-root", default=os.environ.get("WAN0S_VIDEO_ROOT", "wan0s_videos"))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--video-track-gain", type=float, default=1.0)
    parser.add_argument("--wall-max-action", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "pusht" and args.policy == "state-track":
        raise SystemExit("PushT does not implement --policy state-track; use expert or video-track.")
    if args.data_root is None:
        args.data_root = str(PUSHT_DATA_ROOT if args.task == "pusht" else WALL_DATA_ROOT)
    results = []
    for episode_idx in _parse_episode_ids(args.episode_ids):
        try:
            result = evaluate(args, episode_idx)
            print(f"[reliable-t25 result] {result}")
            results.append(result)
        except Exception as exc:
            import traceback

            print(f"[reliable-t25 {args.task} ep {episode_idx}] ERROR: {exc}")
            traceback.print_exc()
            results.append({"task": args.task, "episode_idx": episode_idx, "error": str(exc)})

    successes = [item["success"] for item in results if "success" in item]
    summary = {
        "task": args.task,
        "policy": args.policy,
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
