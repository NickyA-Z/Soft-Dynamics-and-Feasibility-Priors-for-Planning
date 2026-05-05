from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import gym
import torch
import torch.nn.functional as F


DINO_WM_ROOT = Path(
    os.environ.get(
        "DINO_WM_ROOT",
        Path.home() / "DL2---Grounding-Generated-Videos-" / "dino_wm",
    )
)
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

import env as _dino_env  # noqa: F401  # registers "wall" with gym


def _safe_std(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().std(dim=(0, 1)).clamp_min(1e-8)


def compute_wall_stats(base_dir: str | Path) -> dict[str, torch.Tensor]:
    base = Path(base_dir)
    states = torch.load(base / "states.pth").float()
    actions = torch.load(base / "actions.pth").float()
    normalized_actions = (actions - actions.mean(dim=(0, 1))) / _safe_std(actions)

    return {
        "action_mean": actions.mean(dim=(0, 1)),
        "action_std": _safe_std(actions),
        "proprio_mean": states.mean(dim=(0, 1)),
        "proprio_std": _safe_std(states),
        "action_low": normalized_actions.amin(dim=(0, 1)),
        "action_high": normalized_actions.amax(dim=(0, 1)),
    }


def _to_chw_float(frame: Any, image_size: int = 224) -> torch.Tensor:
    visual = torch.as_tensor(frame, dtype=torch.float32)
    if visual.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape={tuple(visual.shape)}")

    if visual.shape[0] in (1, 3):
        visual = visual
    elif visual.shape[-1] in (1, 3):
        visual = visual.permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer image channel dimension from shape={tuple(visual.shape)}")

    if visual.max() > 1.5:
        visual = visual / 255.0

    if visual.shape[-2:] != (image_size, image_size):
        visual = F.interpolate(
            visual.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    # Match datasets.img_transforms.default_transform: Normalize([0.5], [0.5]).
    return (visual - 0.5) / 0.5


def make_wall_observation(frame: Any, proprio: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "visual": _to_chw_float(frame),
        "proprio": torch.as_tensor(proprio, dtype=torch.float32),
    }


def make_wall_runtime_observation(
    obs: dict,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw_proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)
    mean = proprio_mean.to(device=raw_proprio.device, dtype=raw_proprio.dtype)
    std = proprio_std.to(device=raw_proprio.device, dtype=raw_proprio.dtype)
    proprio = (raw_proprio - mean) / std
    return make_wall_observation(obs["visual"], proprio)


def make_visual_only_observation(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "visual": observation["visual"],
    }


def load_wall_oracle_episode(
    base_dir: str | Path,
    episode_idx: int,
    stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    if stats is None:
        stats = compute_wall_stats(base)

    states = torch.load(base / "states.pth").float()
    actions = torch.load(base / "actions.pth").float()
    door_locations = torch.load(base / "door_locations.pth")
    wall_locations = torch.load(base / "wall_locations.pth")

    if episode_idx < 0 or episode_idx >= states.shape[0]:
        raise IndexError(f"Episode {episode_idx} is outside dataset size {states.shape[0]}.")

    frames_path = base / "obses" / f"episode_{episode_idx:03d}.pth"
    frames = torch.load(frames_path)

    length = int(min(states.shape[1], frames.shape[0]))
    if length <= 1:
        raise ValueError(f"Episode {episode_idx} has non-positive usable length: {length}")

    episode_states = states[episode_idx, :length]
    episode_actions = actions[episode_idx, : max(length - 1, 0)]
    episode_actions_norm = (episode_actions - stats["action_mean"]) / stats["action_std"]
    episode_proprio = (episode_states - stats["proprio_mean"]) / stats["proprio_std"]

    video_plan = [
        make_wall_observation(frames[t], episode_proprio[t])
        for t in range(length)
    ]

    env_info = {
        "fix_door_location": door_locations[episode_idx, 0],
        "fix_wall_location": wall_locations[episode_idx, 0],
    }

    return {
        "video_plan": video_plan,
        "start_obs": video_plan[0],
        "goal_obs": video_plan[-1],
        "actions": episode_actions,
        "actions_normalized": episode_actions_norm,
        "states": episode_states,
        "proprio": episode_proprio,
        "door_locations": door_locations[episode_idx, :length],
        "wall_locations": wall_locations[episode_idx, :length],
        "env_info": env_info,
        "length": length,
        **stats,
    }


def slice_wall_oracle_episode(
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
    if episode["actions"].shape[0] < start_offset + horizon * frame_skip:
        raise ValueError(
            f"Episode has only {episode['actions'].shape[0]} actions, "
            f"need {start_offset + horizon * frame_skip}."
        )

    start = int(start_offset)
    stop = start + required_frames
    macro_indices = list(range(0, required_frames, frame_skip))
    full_video_plan = episode["video_plan"][start:stop]
    macro_video_plan = [full_video_plan[i] for i in macro_indices]

    truncated = dict(episode)
    truncated["video_plan"] = macro_video_plan
    truncated["start_obs"] = full_video_plan[0]
    # Keep terminal proprio attached; the adapter's default goal loss is visual-only,
    # but this avoids fake zero-proprio if diagnostics or future losses use it.
    truncated["goal_obs"] = full_video_plan[-1]
    truncated["actions"] = episode["actions"][start : start + horizon * frame_skip]
    truncated["actions_normalized"] = episode["actions_normalized"][
        start : start + horizon * frame_skip
    ]
    truncated["states"] = episode["states"][start:stop]
    truncated["proprio"] = episode["proprio"][start:stop]
    truncated["door_locations"] = episode["door_locations"][start:stop]
    truncated["wall_locations"] = episode["wall_locations"][start:stop]
    truncated["length"] = required_frames
    truncated["planning_horizon"] = horizon
    truncated["frame_skip"] = frame_skip
    truncated["start_offset"] = start
    return truncated


def replay_wall_episode_in_env(
    episode: dict[str, Any],
    episode_idx: int,
    model_cfg: Any,
    horizon: int,
    frame_skip: int,
    start_offset: int = 0,
    env_action_scale: float = 1.0,
) -> dict[str, Any]:
    """Build a DINO-WM-style Wall target by replaying dataset actions in env.

    DINO-WM's planning target path for dataset goals replays demonstration
    actions through the gym environment and uses the replayed observations and
    terminal state as the target. This differs from the raw wall_single tensors:
    those tensors store displacement actions, while DotWall.step applies its
    native transition. This helper makes the oracle-video protocol match the
    original evaluator instead of mixing dataset states with env rollouts.
    """

    required_frames = horizon * frame_skip + 1
    required_actions = horizon * frame_skip
    if start_offset < 0:
        raise ValueError(f"start_offset must be non-negative, got {start_offset}.")
    if episode["actions"].shape[0] < start_offset + required_actions:
        raise ValueError(
            f"Episode has only {episode['actions'].shape[0]} actions, "
            f"need {start_offset + required_actions}."
        )
    if episode["states"].shape[0] <= start_offset:
        raise ValueError(
            f"Episode has only {episode['states'].shape[0]} states, "
            f"cannot start at offset {start_offset}."
        )

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    env.unwrapped.update_env(episode["env_info"])
    init_state = episode["states"][start_offset].detach().cpu().numpy()
    env.unwrapped.seed(episode_idx)
    env.unwrapped.set_init_state(init_state)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    def normalize_env_obs(raw_obs: dict) -> dict[str, torch.Tensor]:
        if hasattr(env.unwrapped, "transform"):
            raw_obs = dict(raw_obs)
            raw_obs["visual"] = env.unwrapped.transform(raw_obs["visual"]).permute(1, 2, 0)
        return make_wall_runtime_observation(
            raw_obs,
            proprio_mean=episode["proprio_mean"],
            proprio_std=episode["proprio_std"],
        )

    full_video_plan = [normalize_env_obs(obs)]
    replay_states = [env.unwrapped.dot_position.detach().cpu().float()]
    action_slice = episode["actions"][start_offset : start_offset + required_actions].float()
    env_device = torch.device(getattr(env.unwrapped, "device", "cpu"))
    for action in action_slice:
        primitive_action = action.to(device=env_device, dtype=torch.float32) * env_action_scale
        step_out = env.step(primitive_action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, _, done, _ = step_out
        full_video_plan.append(normalize_env_obs(obs))
        replay_states.append(env.unwrapped.dot_position.detach().cpu().float())
        if done:
            break

    if len(full_video_plan) < required_frames:
        raise RuntimeError(
            f"Wall env replay ended after {len(full_video_plan)} frames; "
            f"need {required_frames}."
        )

    macro_indices = list(range(0, required_frames, frame_skip))
    replay_states_tensor = torch.stack(replay_states[:required_frames], dim=0)
    replay_proprio = (replay_states_tensor - episode["proprio_mean"]) / episode["proprio_std"]

    truncated = dict(episode)
    truncated["video_plan"] = [full_video_plan[i] for i in macro_indices]
    truncated["start_obs"] = full_video_plan[0]
    truncated["goal_obs"] = full_video_plan[required_frames - 1]
    truncated["actions"] = action_slice
    truncated["actions_normalized"] = (
        action_slice - episode["action_mean"]
    ) / episode["action_std"]
    truncated["states"] = replay_states_tensor
    truncated["proprio"] = replay_proprio
    truncated["door_locations"] = episode["door_locations"][
        start_offset : start_offset + required_frames
    ]
    truncated["wall_locations"] = episode["wall_locations"][
        start_offset : start_offset + required_frames
    ]
    truncated["length"] = required_frames
    truncated["planning_horizon"] = horizon
    truncated["frame_skip"] = frame_skip
    truncated["start_offset"] = int(start_offset)
    truncated["wall_target_source"] = "env-replay"
    return truncated


def candidate_wall_episodes(base_dir: str | Path, horizon: int, frame_skip: int) -> list[int]:
    base = Path(base_dir)
    states = torch.load(base / "states.pth")
    actions = torch.load(base / "actions.pth")
    required_frames = horizon * frame_skip + 1
    required_actions = horizon * frame_skip
    if states.shape[1] < required_frames or actions.shape[1] < required_actions:
        return []
    return list(range(int(states.shape[0])))


def resolve_wall_data_dir(data_root: str | Path, split: str) -> Path:
    root = Path(data_root)
    split_dir = root / split
    if split != "all" and split_dir.exists():
        return split_dir
    return root
