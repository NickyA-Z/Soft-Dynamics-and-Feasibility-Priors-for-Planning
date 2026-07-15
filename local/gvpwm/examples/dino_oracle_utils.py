from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch

import sys

DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from datasets.pusht_dset import PROPRIO_MEAN, PROPRIO_STD

def make_observation(frame, proprio):
    visual = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    visual = (visual - 0.5) / 0.5
    proprio = torch.as_tensor(proprio, dtype=torch.float32)[..., :4]
    return {
        "visual": visual,
        "proprio": proprio,
    }


def make_visual_only_observation(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "visual": observation["visual"],
    }


def load_oracle_episode(base_dir: str | Path, episode_idx: int) -> dict[str, Any]:
    base = Path(base_dir)

    abs_actions = torch.load(base / "abs_actions.pth")
    rel_actions = torch.load(base / "rel_actions.pth")
    states = torch.load(base / "states.pth")
    velocities = torch.load(base / "velocities.pth")
    
    with open(base / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = pickle.load(handle)

    length = int(seq_lengths[episode_idx])
    if length <= 0:
        raise ValueError(f"Episode {episode_idx} has non-positive length: {length}")

    video_path = base / "obses" / f"episode_{episode_idx:03d}.mp4"
    frames = iio.imread(video_path)
    frames = frames[:length]
    if frames.shape[0] < length:
        raise ValueError(
            f"Episode {episode_idx} video has {frames.shape[0]} frames, expected at least {length}."
        )

    episode_states = states[episode_idx, :length]
    episode_velocities = velocities[episode_idx, :length]
    episode_proprio_raw = torch.cat([episode_states[..., :2], episode_velocities], dim=-1)
    episode_proprio = (episode_proprio_raw.float() - PROPRIO_MEAN[:4]) / PROPRIO_STD[:4]
    episode_actions = abs_actions[episode_idx, : max(length - 1, 0)]
    episode_rel_actions = rel_actions[episode_idx, : max(length - 1, 0)]
    episode_velocities = velocities[episode_idx, :length]

    video_plan = [
        make_observation(frames[t], episode_proprio[t])
        for t in range(length)
    ]

    return {
        "video_plan": video_plan,
        "start_obs": video_plan[0],
        "goal_obs": video_plan[-1],
        "actions": episode_actions,
        "rel_actions": episode_rel_actions,
        "states": episode_states,
        "proprio": episode_proprio,
        "velocities": episode_velocities,
        "length": length,
    }


def slice_oracle_episode(
    episode: dict[str, Any],
    horizon: int,
    frame_skip: int,
    start_offset: int = 0,
) -> dict[str, Any]:
    required_frames = horizon * frame_skip + 1
    if start_offset < 0:
        raise ValueError(f"start_offset must be non-negative, got {start_offset}.")
    if episode["length"] < start_offset + required_frames:
        raise ValueError(
            f"Episode length {episode['length']} is too short for horizon={horizon} "
            f"with frame_skip={frame_skip} and start_offset={start_offset} "
            f"(need {start_offset + required_frames} frames)."
        )

    start = int(start_offset)
    stop = start + required_frames
    macro_indices = list(range(0, required_frames, frame_skip))

    truncated = dict(episode)
    full_video_plan = episode["video_plan"][start:stop]
    macro_video_plan = [full_video_plan[i] for i in macro_indices]
    
    truncated["video_plan"] = macro_video_plan
    truncated["start_obs"] = full_video_plan[0]
    truncated["goal_obs"] = full_video_plan[-1]
    truncated["actions"] = episode["actions"][start : start + horizon * frame_skip]
    truncated["rel_actions"] = episode["rel_actions"][start : start + horizon * frame_skip]
    truncated["states"] = episode["states"][start:stop]
    truncated["proprio"] = episode["proprio"][start:stop]
    truncated["velocities"] = episode["velocities"][start:stop]
    truncated["length"] = required_frames
    truncated["planning_horizon"] = horizon
    truncated["frame_skip"] = frame_skip
    truncated["start_offset"] = start
    return truncated


def infer_horizon(
    length: int,
    frame_skip: int = 1,
    max_horizon: int | None = None,
) -> int:
    horizon = max((length - 1) // frame_skip, 1)
    if max_horizon is not None:
        horizon = min(horizon, max_horizon)
    return horizon
