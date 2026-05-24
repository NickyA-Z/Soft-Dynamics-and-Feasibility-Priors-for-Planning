from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


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


def _model_paths(model_dir: Path) -> list[str | list[str]]:
    diffusion = sorted(glob.glob(str(model_dir / "diffusion_pytorch_model-*.safetensors")))
    if not diffusion:
        single = model_dir / "diffusion_pytorch_model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"Missing Wan diffusion weights under {model_dir}")
        diffusion = [str(single)]
    required = [
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    return [diffusion, *(str(path) for path in required)]


def _load_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    model_dir = Path(args.model_dir)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else model_dir / "google" / "umt5-xxl"
    model_configs = [
        ModelConfig(path=path)
        for path in _model_paths(model_dir)
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
        redirect_common_files=False,
        vram_limit=args.vram_limit_gb,
    )
    pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=args.lora_alpha)
    return pipe


def _read_prompt(case_dir: Path, override: str | None) -> str:
    if override is not None:
        return override
    return (case_dir / "prompt.txt").read_text(encoding="utf-8").strip()


def _generate_one(
    pipe: WanVideoPipeline,
    args: argparse.Namespace,
    case_dir: Path,
    seed: int,
) -> dict:
    output_path = case_dir / args.output_name
    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        print(f"[wanft generate] existing video found, skipping: {output_path}")
        return {"case_dir": str(case_dir), "output": str(output_path), "seed": seed, "skipped": True}

    first_frame = Image.open(case_dir / "first_frame.png").convert("RGB")
    last_frame = Image.open(case_dir / "last_frame.png").convert("RGB")
    prompt = _read_prompt(case_dir, args.prompt)
    print(f"[wanft generate] generating {output_path} seed={seed}")
    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        input_image=first_frame,
        end_image=last_frame,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        cfg_scale=args.cfg_scale,
        seed=seed,
        tiled=not args.disable_tiling,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(f".tmp{output_path.suffix}")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    save_video(video, str(tmp_output_path), fps=args.fps, quality=args.quality)
    tmp_output_path.replace(output_path)
    return {"case_dir": str(case_dir), "output": str(output_path), "seed": seed, "skipped": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WAN-FT FLF2V videos from prepared case folders.")
    parser.add_argument("--task", choices=("pusht", "wall"), required=True)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--episode-specs", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--output-name", default="wanft.mp4")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--sigma-shift", type=float, default=16.0)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Offset added to seed-base before per-case indexing. Use this when "
            "splitting one evaluation list across multiple Slurm jobs so each "
            "case receives the same seed it would have received in a single job."
        ),
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--disable-tiling", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_specs:
        specs = _parse_episode_specs(args.episode_specs)
    else:
        specs = [(episode_idx, 0) for episode_idx in _parse_episode_ids(args.episode_ids)]
    if not specs:
        raise SystemExit("No episodes requested.")

    pipe = _load_pipeline(args)
    results = []
    for case_idx, (episode_idx, start_offset) in enumerate(specs):
        case_dir = Path(args.case_root) / args.task / _case_dir_name(episode_idx, start_offset)
        results.append(_generate_one(pipe, args, case_dir, seed=args.seed_base + args.seed_offset + case_idx))

    summary = {
        "task": args.task,
        "model_dir": args.model_dir,
        "lora_checkpoint": args.lora_checkpoint,
        "num_cases": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
