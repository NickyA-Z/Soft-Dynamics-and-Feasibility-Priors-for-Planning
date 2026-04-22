from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))


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
) -> dict[str, Any]:
    required_frames = horizon * frame_skip + 1
    if episode["length"] < required_frames:
        raise ValueError(
            f"Episode length {episode['length']} is too short for horizon={horizon} "
            f"with frame_skip={frame_skip} (need {required_frames} frames)."
        )
    if episode["actions"].shape[0] < horizon * frame_skip:
        raise ValueError(
            f"Episode has only {episode['actions'].shape[0]} actions, need {horizon * frame_skip}."
        )

    macro_indices = list(range(0, required_frames, frame_skip))
    full_video_plan = episode["video_plan"][:required_frames]
    macro_video_plan = [full_video_plan[i] for i in macro_indices]

    truncated = dict(episode)
    truncated["video_plan"] = macro_video_plan
    truncated["start_obs"] = full_video_plan[0]
    # Keep terminal proprio attached; the adapter's default goal loss is visual-only,
    # but this avoids fake zero-proprio if diagnostics or future losses use it.
    truncated["goal_obs"] = full_video_plan[-1]
    truncated["actions"] = episode["actions"][: horizon * frame_skip]
    truncated["actions_normalized"] = episode["actions_normalized"][: horizon * frame_skip]
    truncated["states"] = episode["states"][:required_frames]
    truncated["proprio"] = episode["proprio"][:required_frames]
    truncated["door_locations"] = episode["door_locations"][:required_frames]
    truncated["wall_locations"] = episode["wall_locations"][:required_frames]
    truncated["length"] = required_frames
    truncated["planning_horizon"] = horizon
    truncated["frame_skip"] = frame_skip
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
