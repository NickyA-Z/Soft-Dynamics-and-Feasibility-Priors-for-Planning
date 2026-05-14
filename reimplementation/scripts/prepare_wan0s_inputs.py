from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from gvpwm.examples.dino_oracle_utils import load_oracle_episode, slice_oracle_episode
from gvpwm.examples.dino_wall_oracle_utils import (
    DINO_WM_ROOT,
    compute_wall_stats,
    load_wall_oracle_episode,
    replay_wall_episode_in_env,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)
from gvpwm.examples.wan0s_utils import prompt_for_task, write_wan0s_case


def _parse_episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_episode_specs(value: str) -> list[tuple[int, int]]:
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            episode_text, offset_text = item.split(":", 1)
            specs.append((int(episode_text), int(offset_text)))
        else:
            specs.append((int(item), 0))
    return specs


def _case_dir_name(episode_idx: int, start_offset: int) -> str:
    if start_offset == 0:
        return f"episode_{episode_idx:03d}"
    return f"episode_{episode_idx:03d}_offset_{start_offset:03d}"


def _macro_horizon(raw_horizon: int, frame_skip: int) -> int:
    if raw_horizon % frame_skip != 0:
        raise ValueError(f"raw_horizon={raw_horizon} must be divisible by frame_skip={frame_skip}")
    return raw_horizon // frame_skip


def prepare_wall(args: argparse.Namespace) -> list[dict]:
    data_dir = resolve_wall_data_dir(args.data_root, args.split)
    stats = compute_wall_stats(data_dir)
    model_cfg = OmegaConf.load(DINO_WM_ROOT / "checkpoints" / "outputs" / "wall_single" / "hydra.yaml")
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    records = []
    for episode_idx, start_offset in _parse_episode_specs(args.episode_specs):
        episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
        if args.wall_target_source == "env-replay":
            env_action_scale = 1.0 if args.wall_env_action_scale is None else args.wall_env_action_scale
            episode = replay_wall_episode_in_env(
                episode,
                episode_idx=episode_idx,
                model_cfg=model_cfg,
                horizon=horizon,
                frame_skip=args.frame_skip,
                start_offset=start_offset,
                env_action_scale=env_action_scale,
            )
        else:
            episode = slice_wall_oracle_episode(
                episode,
                horizon=horizon,
                frame_skip=args.frame_skip,
                start_offset=start_offset,
            )
        out_dir = Path(args.output_root) / "wall" / _case_dir_name(episode_idx, start_offset)
        records.append(
            write_wan0s_case(
                output_dir=out_dir,
                task="wall",
                episode_idx=episode_idx,
                start_offset=start_offset,
                start_visual=episode["start_obs"]["visual"],
                goal_visual=episode["goal_obs"]["visual"],
                raw_horizon=args.raw_horizon,
                frame_skip=args.frame_skip,
                prompt=args.prompt or prompt_for_task("wall"),
                width=args.width,
                height=args.height,
            )
        )
    return records


def prepare_pusht(args: argparse.Namespace) -> list[dict]:
    data_dir = Path(args.data_root) / args.split
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    records = []
    for episode_idx, start_offset in _parse_episode_specs(args.episode_specs):
        episode = load_oracle_episode(data_dir, episode_idx)
        episode = slice_oracle_episode(
            episode,
            horizon=horizon,
            frame_skip=args.frame_skip,
            start_offset=start_offset,
        )
        out_dir = Path(args.output_root) / "pusht" / _case_dir_name(episode_idx, start_offset)
        records.append(
            write_wan0s_case(
                output_dir=out_dir,
                task="pusht",
                episode_idx=episode_idx,
                start_offset=start_offset,
                start_visual=episode["start_obs"]["visual"],
                goal_visual=episode["goal_obs"]["visual"],
                raw_horizon=args.raw_horizon,
                frame_skip=args.frame_skip,
                prompt=args.prompt or prompt_for_task("pusht"),
                width=args.width,
                height=args.height,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare start/goal frames for WAN-0S FLF2V generation.")
    parser.add_argument("--task", choices=("wall", "pusht"), required=True)
    parser.add_argument("--episode-ids", default="", help="Comma-separated episode ids")
    parser.add_argument(
        "--episode-specs",
        default=None,
        help="Comma-separated episode[:offset] specs. Overrides --episode-ids.",
    )
    parser.add_argument("--split", default="all", help="Wall uses all by default; PushT usually uses val")
    parser.add_argument("--raw-horizon", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280, help="WAN conditioning image width")
    parser.add_argument("--height", type=int, default=720, help="WAN conditioning image height")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=os.environ.get("WAN0S_VIDEO_ROOT", "wan0s_videos"))
    parser.add_argument("--prompt", default=None, help="Override the default Chinese FLF2V prompt")
    parser.add_argument(
        "--wall-target-source",
        choices=("env-replay", "dataset"),
        default="env-replay",
        help="For Wall, prepare DINO-WM env-replayed goals or raw dataset goals.",
    )
    parser.add_argument(
        "--wall-env-action-scale",
        type=float,
        default=None,
        help="For Wall env-replay preparation, scale dataset actions before env.step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_specs is None:
        if not args.episode_ids:
            raise SystemExit("Either --episode-ids or --episode-specs is required.")
        args.episode_specs = ",".join(f"{episode_idx}:0" for episode_idx in _parse_episode_ids(args.episode_ids))
    if args.data_root is None:
        dino_root = Path(os.environ.get("DINO_WM_ROOT", Path.home() / "DL2---Grounding-Generated-Videos-" / "dino_wm"))
        if args.task == "wall":
            args.data_root = str(dino_root / "data" / "wall_single")
        else:
            args.data_root = str(dino_root / "data" / "pusht_noise")
            if args.split == "all":
                args.split = "val"

    records = prepare_wall(args) if args.task == "wall" else prepare_pusht(args)
    for record in records:
        print(
            record["episode_idx"],
            record["start_offset"],
            record["first_frame"],
            record["last_frame"],
            record["expected_video"],
        )


if __name__ == "__main__":
    main()
