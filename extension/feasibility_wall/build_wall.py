from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model

from nicky_dl.adapters.dino_wm import DinoWorldModelAdapter
from extension.feasibility2.dataset import build_feasibility_tensors_from_oracle
from reimplementation.src.gvpwm.examples.dino_wall_oracle_utils import compute_wall_stats

PRIMITIVE_ACTION_DIM = 2


def load_world_model(device: torch.device):
    model_dir = DINO_WM_ROOT / "checkpoints" / "outputs" / "wall_single"
    model_cfg = OmegaConf.load(model_dir / "hydra.yaml")
    model_ckpt = model_dir / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    model.eval()

    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    if wm_action_dim % PRIMITIVE_ACTION_DIM != 0:
        raise ValueError(
            f"World model action dim {wm_action_dim} is not divisible by "
            f"primitive action dim {PRIMITIVE_ACTION_DIM}"
        )

    action_repeat = wm_action_dim // PRIMITIVE_ACTION_DIM
    action_low = torch.full((wm_action_dim,), -3.0, device=device, dtype=torch.float32)
    action_high = torch.full((wm_action_dim,), 3.0, device=device, dtype=torch.float32)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    world_model.world_model.eval()

    print("Loaded Wall DINO-WM")
    print("history_length:", int(world_model.history_length))
    print("action_repeat:", int(action_repeat))
    print("wm_action_dim:", int(wm_action_dim))
    print("frameskip:", getattr(model_cfg, "frameskip", None))

    return world_model, action_repeat, model_cfg, model_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Wall feasibility dataset.")
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional .pt dataset file to reuse latent_mean/latent_std from.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DINO_WM_ROOT / "data" / "wall_single"),
        help="Wall dataset root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_root)

    world_model, action_repeat, model_cfg, model_ckpt = load_world_model(device)
    wall_stats = compute_wall_stats(data_dir)
    action_mean = wall_stats["action_mean"].to(device=device, dtype=torch.float32)
    action_std = wall_stats["action_std"].to(device=device, dtype=torch.float32)

    histories, actions, next_latents, past_action_histories = build_feasibility_tensors_from_oracle(
        world_model=world_model,
        data_dir=data_dir,
        episode_indices=range(args.episode_start, args.episode_end),
        action_repeat=action_repeat,
        primitive_action_dim=PRIMITIVE_ACTION_DIM,
        action_mean=action_mean,
        action_std=action_std,
        divide_actions_by=1.0,
        preserve_latent_shape=True,
        device=device,
        return_past_action_histories=True,
        domain="wall_single",
    )

    history_length = int(world_model.history_length)
    expected_action_dim = action_repeat * PRIMITIVE_ACTION_DIM
    latent_shape = tuple(next_latents.shape[1:])

    if histories.shape[1] != history_length:
        raise RuntimeError(f"Expected history length {history_length}, got {histories.shape[1]}")
    if actions.shape[-1] != expected_action_dim:
        raise RuntimeError(f"Expected action dim {expected_action_dim}, got {actions.shape[-1]}")
    if past_action_histories.shape[-2:] != (history_length, expected_action_dim):
        raise RuntimeError(
            f"Expected past action history shape [N,{history_length},{expected_action_dim}], "
            f"got {tuple(past_action_histories.shape)}"
        )
    if tuple(histories.shape[2:]) != latent_shape:
        raise RuntimeError(
            f"History latent shape {tuple(histories.shape[2:])} does not match next_latents {latent_shape}"
        )

    if args.norm_stats is not None:
        stats = torch.load(args.norm_stats, map_location="cpu")
        latent_mean = stats["latent_mean"]
        latent_std = stats["latent_std"]
        print(f"Loaded norm stats from {args.norm_stats}")
    else:
        all_latents = torch.cat([histories.reshape(-1, *latent_shape), next_latents], dim=0)
        latent_mean = all_latents.mean(dim=0)
        latent_std = all_latents.std(dim=0).clamp_min(1e-6)
        print("Computed norm stats from this split")

    histories = histories.cpu()
    next_latents = next_latents.cpu()
    past_action_histories = past_action_histories.cpu()
    latent_mean = latent_mean.cpu()
    latent_std = latent_std.cpu()

    histories = (histories - latent_mean.view(1, 1, *latent_shape)) / latent_std.view(1, 1, *latent_shape)
    next_latents = (next_latents - latent_mean.view(1, *latent_shape)) / latent_std.view(1, *latent_shape)

    print("histories:", tuple(histories.shape), histories.dtype)
    print("actions:", tuple(actions.shape), actions.dtype)
    print("next_latents:", tuple(next_latents.shape), next_latents.dtype)
    print("past_action_histories:", tuple(past_action_histories.shape), past_action_histories.dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "histories": histories.cpu(),
            "actions": actions.cpu(),
            "next_latents": next_latents.cpu(),
            "past_action_histories": past_action_histories.cpu(),
            "domain": "wall_single",
            "history_length": history_length,
            "latent_shape": latent_shape,
            "preserve_latent_shape": True,
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            "action_repeat": action_repeat,
            "primitive_action_dim": PRIMITIVE_ACTION_DIM,
            "action_scale": 1.0,
            "action_mean": action_mean.detach().cpu(),
            "action_std": action_std.detach().cpu(),
            "data_dir": str(data_dir),
            "model_cfg": str(model_cfg),
            "model_ckpt": str(model_ckpt),
            "episode_start": int(args.episode_start),
            "episode_end": int(args.episode_end),
        },
        out_path,
    )
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
