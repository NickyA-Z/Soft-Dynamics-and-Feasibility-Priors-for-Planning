from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch


def make_observation(frame, proprio):
    visual = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    proprio = torch.as_tensor(proprio, dtype=torch.float32)[..., :4]
    return {
        "visual": visual,
        "proprio": proprio,
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
    episode_proprio = episode_states[..., :4]
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


def infer_horizon(
    length: int,
    frame_skip: int = 1,
    max_horizon: int | None = None,
) -> int:
    horizon = max((length - 1) // frame_skip, 1)
    if max_horizon is not None:
        horizon = min(horizon, max_horizon)
    return horizon