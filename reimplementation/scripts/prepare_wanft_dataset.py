from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
import torch

from gvpwm.examples.wan0s_utils import letterbox_for_wan, prompt_for_task, visual_to_uint8_hwc


def _parse_episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _resample_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot resample an empty video.")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if length == num_frames:
        return np.arange(length)
    return np.rint(np.linspace(0, length - 1, num_frames)).astype(np.int64)


def _resample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    mode: str,
) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Cannot resample an empty video.")
    if mode == "nearest":
        return [frames[i] for i in _resample_indices(len(frames), num_frames)]
    if mode != "linear":
        raise ValueError(f"Unsupported resample mode: {mode!r}")
    if len(frames) == num_frames:
        return [np.asarray(frame) for frame in frames]
    if len(frames) == 1:
        return [np.asarray(frames[0]) for _ in range(num_frames)]

    positions = np.linspace(0.0, len(frames) - 1, num_frames)
    frame_array = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames], axis=0)
    sampled = []
    for pos in positions:
        left = int(np.floor(pos))
        right = min(left + 1, len(frames) - 1)
        alpha = float(pos - left)
        blended = (1.0 - alpha) * frame_array[left] + alpha * frame_array[right]
        sampled.append(np.clip(np.rint(blended), 0, 255).astype(np.uint8))
    return sampled


def _write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(list(frames), axis=0), fps=fps, codec="libx264")


def _letterbox_sequence(frames: Iterable[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    return [np.asarray(letterbox_for_wan(frame, size=(width, height))) for frame in frames]


def _load_pusht_lengths(split_dir: Path) -> list[int]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        return pickle.load(handle)


def _candidate_pusht_ids(split_dir: Path, min_frames: int, limit: int) -> list[int]:
    lengths = _load_pusht_lengths(split_dir)
    ids = [
        idx
        for idx, length in enumerate(lengths)
        if length >= min_frames and (split_dir / "obses" / f"episode_{idx:03d}.mp4").exists()
    ]
    return ids[:limit]


def _candidate_wall_ids(data_root: Path, min_frames: int, limit: int) -> list[int]:
    states = torch.load(data_root / "states.pth", map_location="cpu")
    first_episode_frames = torch.load(data_root / "obses" / "episode_000.pth", map_location="cpu").shape[0]
    if first_episode_frames < min_frames:
        return []
    ids = [
        idx
        for idx in range(int(states.shape[0]))
        if (data_root / "obses" / f"episode_{idx:03d}.pth").exists()
    ]
    return ids[:limit]


def _load_pusht_frames(split_dir: Path, episode_idx: int) -> list[np.ndarray]:
    frames = iio.imread(split_dir / "obses" / f"episode_{episode_idx:03d}.mp4")
    return [frame for frame in frames]


def _load_wall_frames(data_root: Path, episode_idx: int) -> list[np.ndarray]:
    frames = torch.load(
        data_root / "obses" / f"episode_{episode_idx:03d}.pth",
        map_location="cpu",
    )
    return [visual_to_uint8_hwc(frame) for frame in frames]


def _clip_specs(
    num_source_frames: int,
    raw_horizon: int,
    clip_mode: str,
    segments_per_demo: int,
    episode_position: int = 0,
    num_episodes: int = 1,
) -> list[tuple[int, int | None]]:
    required_frames = raw_horizon + 1
    if clip_mode == "prefix":
        if num_source_frames < required_frames:
            return []
        return [(0, required_frames)]
    if clip_mode == "full":
        if num_source_frames < 2:
            return []
        return [(0, None)]
    if clip_mode == "uniform":
        if num_source_frames < required_frames:
            return []
        max_offset = num_source_frames - required_frames
        count = max(int(segments_per_demo), 1)
        if count == 1:
            offsets = [0]
        else:
            offsets = np.rint(np.linspace(0, max_offset, count)).astype(np.int64).tolist()
        return [(int(offset), required_frames) for offset in offsets]
    if clip_mode == "staggered":
        if num_source_frames < required_frames:
            return []
        max_offset = num_source_frames - required_frames
        if max_offset <= 0:
            offset = 0
        elif num_episodes <= 1:
            offset = max_offset // 2
        else:
            # Keep one horizon-length clip per demo while spreading temporal
            # coverage across the 100-demo WAN-FT set.
            fraction = float(episode_position) / float(max(num_episodes - 1, 1))
            offset = int(round(fraction * max_offset))
        return [(offset, required_frames)]
    raise ValueError(f"Unsupported clip_mode={clip_mode!r}")


def _prepare_task(args: argparse.Namespace) -> list[dict]:
    output_root = Path(args.output_root)
    video_dir = output_root / "videos"
    min_frames = 2 if args.clip_mode == "full" else args.raw_horizon + 1
    if args.task == "pusht":
        split_dir = Path(args.data_root) / args.split
        episode_ids = (
            _parse_episode_ids(args.episode_ids)
            if args.episode_ids
            else _candidate_pusht_ids(split_dir, min_frames, args.num_demos)
        )
        lengths = _load_pusht_lengths(split_dir)

        def loader(idx: int) -> list[np.ndarray]:
            frames = _load_pusht_frames(split_dir, idx)
            return frames[: int(lengths[idx])]
    else:
        data_root = Path(args.data_root)
        episode_ids = (
            _parse_episode_ids(args.episode_ids)
            if args.episode_ids
            else _candidate_wall_ids(data_root, min_frames, args.num_demos)
        )
        loader = lambda idx: _load_wall_frames(data_root, idx)

    if len(episode_ids) < args.num_demos and not args.episode_ids:
        raise RuntimeError(
            f"Only found {len(episode_ids)} usable {args.task} demos; "
            f"requested {args.num_demos}."
        )
    episode_ids = episode_ids[: args.num_demos]

    prompt = args.prompt or prompt_for_task(args.task)
    rows = []
    out_idx = 0
    for episode_position, episode_idx in enumerate(episode_ids):
        raw_frames = loader(episode_idx)
        specs = _clip_specs(
            num_source_frames=len(raw_frames),
            raw_horizon=args.raw_horizon,
            clip_mode=args.clip_mode,
            segments_per_demo=args.segments_per_demo,
            episode_position=episode_position,
            num_episodes=len(episode_ids),
        )
        if not specs:
            print(
                f"[wanft dataset] skipping episode={episode_idx}: "
                f"source_frames={len(raw_frames)} clip_mode={args.clip_mode}"
            )
            continue
        for segment_idx, (start_offset, clip_length) in enumerate(specs):
            if clip_length is None:
                clip = raw_frames[start_offset:]
            else:
                clip = raw_frames[start_offset : start_offset + clip_length]
            sampled = _resample_frames(clip, args.num_frames, mode=args.resample_mode)
            frames = _letterbox_sequence(sampled, width=args.width, height=args.height)
            video_name = (
                f"{args.task}_{out_idx:04d}_episode_{episode_idx:05d}"
                f"_offset_{start_offset:04d}.mp4"
            )
            _write_video(frames, video_dir / video_name, fps=args.fps)
            rows.append(
                {
                    "video": f"videos/{video_name}",
                    "prompt": prompt,
                    "task": args.task,
                    "episode_idx": episode_idx,
                    "segment_idx": segment_idx,
                    "start_offset": start_offset,
                    "raw_horizon": args.raw_horizon,
                    "clip_mode": args.clip_mode,
                    "clip_frames": len(clip),
                    "source_frames": len(raw_frames),
                    "num_frames": args.num_frames,
                    "resample_mode": args.resample_mode,
                }
            )
            print(
                f"[wanft dataset] wrote {video_name} from episode={episode_idx} "
                f"offset={start_offset} clip_frames={len(clip)}"
            )
            out_idx += 1

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"video": row["video"], "prompt": row["prompt"]})

    manifest = {
        "task": args.task,
        "data_root": str(args.data_root),
        "split": args.split,
        "num_demos": len(rows),
        "raw_horizon": args.raw_horizon,
        "clip_mode": args.clip_mode,
        "segments_per_demo": args.segments_per_demo,
        "num_frames": args.num_frames,
        "resample_mode": args.resample_mode,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "metadata_csv": str(metadata_path),
        "rows": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare task demos for WAN-FT LoRA training.")
    parser.add_argument("--task", choices=("pusht", "wall"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--raw-horizon", type=int, default=25)
    parser.add_argument(
        "--clip-mode",
        choices=("prefix", "full", "uniform", "staggered"),
        default="prefix",
        help=(
            "prefix matches the original local implementation by using the first "
            "raw_horizon+1 frames; full uses the whole demonstration; uniform "
            "samples fixed-horizon windows across each demo; staggered keeps one "
            "window per demo while spreading offsets across the 100-demo set."
        ),
    )
    parser.add_argument("--segments-per-demo", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--resample-mode",
        choices=("nearest", "linear"),
        default="nearest",
        help=(
            "Temporal resampling used when demo clips do not already have "
            "num_frames. nearest preserves the previous behavior; linear "
            "blends adjacent rendered frames to avoid repeat-frame jump artifacts."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "pusht" and args.split == "all":
        args.split = "train"
    rows = _prepare_task(args)
    print(f"[wanft dataset] completed task={args.task} rows={len(rows)}")


if __name__ == "__main__":
    main()
