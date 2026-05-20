#!/usr/bin/env python
"""Generate fixed sampled-50 episode:offset specs for Table 2 runs."""

from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

import torch


def _sample_pusht_specs(
    split_dir: Path,
    raw_horizon: int,
    frame_skip: int,
    num_specs: int,
    seed: int,
) -> list[tuple[int, int]]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = [int(length) for length in pickle.load(handle)]

    macro_horizon = raw_horizon // frame_skip
    required_frames = macro_horizon * frame_skip + 1
    rng = random.Random(seed)
    specs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(specs) < num_specs:
        episode_idx = rng.randint(0, len(seq_lengths) - 1)
        max_offset = seq_lengths[episode_idx] - required_frames
        if max_offset < 0:
            continue
        spec = (episode_idx, rng.randint(0, max_offset))
        if spec in seen:
            continue
        seen.add(spec)
        specs.append(spec)
    return specs


def _sample_wall_specs(
    data_dir: Path,
    raw_horizon: int,
    frame_skip: int,
    num_specs: int,
    seed: int,
) -> list[tuple[int, int]]:
    actions = torch.load(data_dir / "actions.pth", map_location="cpu")
    first_frames = torch.load(data_dir / "obses" / "episode_000.pth", map_location="cpu")
    macro_horizon = raw_horizon // frame_skip
    required_frames = macro_horizon * frame_skip + 1
    required_actions = macro_horizon * frame_skip
    max_offset = min(
        int(first_frames.shape[0]) - required_frames,
        int(actions.shape[1]) - required_actions,
    )
    if max_offset < 0:
        raise ValueError(
            f"Wall data is too short for T={raw_horizon}: "
            f"frames={int(first_frames.shape[0])}, actions={int(actions.shape[1])}, "
            f"need frames={required_frames}, actions={required_actions}."
        )
    rng = random.Random(seed)
    specs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(specs) < num_specs:
        episode_idx = rng.randint(0, int(actions.shape[0]) - 1)
        spec = (episode_idx, rng.randint(0, max_offset))
        if spec in seen:
            continue
        seen.add(spec)
        specs.append(spec)
    return specs


def _format_specs(specs: list[tuple[int, int]]) -> str:
    return ",".join(f"{episode}:{offset}" for episode, offset in specs)


def _write_specs(specs: list[tuple[int, int]], specs_path: Path, chunks_path: Path, chunk_size: int) -> None:
    specs_path.write_text(_format_specs(specs) + "\n")
    chunks = [specs[index : index + chunk_size] for index in range(0, len(specs), chunk_size)]
    chunks_path.write_text("\n".join(_format_specs(chunk) for chunk in chunks) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    runtime_default = os.environ.get("DL2RUNTIME_ROOT", str(Path.cwd()))
    dino_default = os.environ.get(
        "DINO_DATA_ROOT",
        str(Path.home() / "DL2---Grounding-Generated-Videos-" / "dino_wm" / "data"),
    )
    parser.add_argument("--runtime-dir", default=runtime_default)
    parser.add_argument("--dino-data-root", default=dino_default)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--num-specs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--pusht-horizons", type=int, nargs="+", default=[25, 50, 80])
    parser.add_argument("--wall-horizons", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--force", action="store_true", help="Overwrite existing spec files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(horizon % args.frame_skip != 0 for horizon in args.pusht_horizons + args.wall_horizons):
        raise SystemExit(f"All raw horizons must be divisible by frame_skip={args.frame_skip}.")

    runtime_dir = Path(args.runtime_dir)
    specs_dir = runtime_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    dino_data_root = Path(args.dino_data_root)
    pusht_split_dir = dino_data_root / "pusht_noise" / "val"
    wall_data_dir = dino_data_root / "wall_single"

    jobs: list[tuple[str, int, list[tuple[int, int]]]] = []
    for raw_horizon in args.pusht_horizons:
        jobs.append(
            (
                "pusht",
                raw_horizon,
                _sample_pusht_specs(
                    pusht_split_dir,
                    raw_horizon=raw_horizon,
                    frame_skip=args.frame_skip,
                    num_specs=args.num_specs,
                    seed=args.seed,
                ),
            )
        )
    for raw_horizon in args.wall_horizons:
        jobs.append(
            (
                "wall",
                raw_horizon,
                _sample_wall_specs(
                    wall_data_dir,
                    raw_horizon=raw_horizon,
                    frame_skip=args.frame_skip,
                    num_specs=args.num_specs,
                    seed=args.seed,
                ),
            )
        )

    for task, raw_horizon, specs in jobs:
        specs_path = specs_dir / f"t{raw_horizon}_{task}_seed{args.seed}_{args.num_specs}.specs"
        chunks_path = specs_dir / f"t{raw_horizon}_{task}_seed{args.seed}_{args.num_specs}_chunks{args.chunk_size}.txt"
        if not args.force and (specs_path.exists() or chunks_path.exists()):
            print(f"[skip] {task} T={raw_horizon}: existing {specs_path.name}/{chunks_path.name}")
            continue
        _write_specs(specs, specs_path=specs_path, chunks_path=chunks_path, chunk_size=args.chunk_size)
        duplicate_count = len(specs) - len(set(specs))
        print(
            f"[write] {task} T={raw_horizon}: {len(specs)} specs, "
            f"duplicates={duplicate_count}, specs={specs_path}, chunks={chunks_path}"
        )


if __name__ == "__main__":
    main()
