from __future__ import annotations

import argparse
import os
from pathlib import Path

from gvpwm.examples.dino_oracle_utils import load_oracle_episode, slice_oracle_episode
from gvpwm.examples.dino_wall_oracle_utils import (
    compute_wall_stats,
    load_wall_oracle_episode,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)
from gvpwm.examples.wan0s_utils import prompt_for_task, write_wan0s_case


def _parse_episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _macro_horizon(raw_horizon: int, frame_skip: int) -> int:
    if raw_horizon % frame_skip != 0:
        raise ValueError(f"raw_horizon={raw_horizon} must be divisible by frame_skip={frame_skip}")
    return raw_horizon // frame_skip


def prepare_wall(args: argparse.Namespace) -> list[dict]:
    data_dir = resolve_wall_data_dir(args.data_root, args.split)
    stats = compute_wall_stats(data_dir)
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    records = []
    for episode_idx in _parse_episode_ids(args.episode_ids):
        episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
        episode = slice_wall_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)
        out_dir = Path(args.output_root) / "wall" / f"episode_{episode_idx:03d}"
        records.append(
            write_wan0s_case(
                output_dir=out_dir,
                task="wall",
                episode_idx=episode_idx,
                start_visual=episode["start_obs"]["visual"],
                goal_visual=episode["goal_obs"]["visual"],
                raw_horizon=args.raw_horizon,
                frame_skip=args.frame_skip,
                prompt=args.prompt or prompt_for_task("wall"),
            )
        )
    return records


def prepare_pusht(args: argparse.Namespace) -> list[dict]:
    data_dir = Path(args.data_root) / args.split
    horizon = _macro_horizon(args.raw_horizon, args.frame_skip)
    records = []
    for episode_idx in _parse_episode_ids(args.episode_ids):
        episode = load_oracle_episode(data_dir, episode_idx)
        episode = slice_oracle_episode(episode, horizon=horizon, frame_skip=args.frame_skip)
        out_dir = Path(args.output_root) / "pusht" / f"episode_{episode_idx:03d}"
        records.append(
            write_wan0s_case(
                output_dir=out_dir,
                task="pusht",
                episode_idx=episode_idx,
                start_visual=episode["start_obs"]["visual"],
                goal_visual=episode["goal_obs"]["visual"],
                raw_horizon=args.raw_horizon,
                frame_skip=args.frame_skip,
                prompt=args.prompt or prompt_for_task("pusht"),
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare start/goal frames for WAN-0S FLF2V generation.")
    parser.add_argument("--task", choices=("wall", "pusht"), required=True)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated episode ids")
    parser.add_argument("--split", default="all", help="Wall uses all by default; PushT usually uses val")
    parser.add_argument("--raw-horizon", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=os.environ.get("WAN0S_VIDEO_ROOT", "wan0s_videos"))
    parser.add_argument("--prompt", default=None, help="Override the default Chinese FLF2V prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        print(record["episode_idx"], record["first_frame"], record["last_frame"], record["expected_video"])


if __name__ == "__main__":
    main()
