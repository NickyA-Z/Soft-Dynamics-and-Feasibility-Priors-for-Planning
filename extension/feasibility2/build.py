"""
Builds the feasibility dataset from DINO-WM oracle episodes.
Loads the pretrained DINO world model,
encodes expert videos into latent states,
applies the same relative-action preprocessing as DINO-WM, and saves tuples:

(h_t, a_t, z_{t+1})
"""
from pathlib import Path
import argparse
import sys
from typing import Optional
import torch
from omegaconf import OmegaConf

DINO_WM_ROOT = Path(__file__).resolve().parents[2] / "dino_wm"
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from nicky_dl.adapters.dino_wm import DinoWorldModelAdapter

from extension.feasibility2.dataset import build_feasibility_tensors_from_oracle
from trash.dino_wall_oracle_utils import compute_wall_stats
"""
# PUSHT Variables
DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"
MODEL_DIR = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
MODEL_CFG = MODEL_DIR / "hydra.yaml"
MODEL_CKPT = MODEL_DIR / "checkpoints" / "model_latest.pth"
# WALL Variables 
DATA_DIR = DINO_WM_ROOT / "data" / "wall_single"
MODEL_NAME = "wall_single"
MODEL_DIR = DINO_WM_ROOT / "checkpoints" / "outputs" / "wall_single"
MODEL_CFG = MODEL_DIR / "hydra.yaml" 
"""

PRIMITIVE_ACTION_DIM = 2
EXPECTED_ACTION_REPEAT = 5
ACTION_SCALE = 100.0



def resolve_domain_paths(domain: str, split: str = "train"):
    if domain == "pusht":
        model_name = "pusht"
        data_dir = DINO_WM_ROOT / "data" / "pusht_noise" / split
    elif domain == "wall_single":
        model_name = "wall_single"
        # Wall may not have train/val folders; adjust if your data layout differs.
        data_dir = DINO_WM_ROOT / "data" / "wall_single" 
        split_dir = DINO_WM_ROOT / "data" / "wall_single" / split
        data_dir = split_dir if split_dir.exists() else data_dir

    else:
        raise ValueError(f"Unsupported domain: {domain}")

    model_dir = DINO_WM_ROOT / "checkpoints" / "outputs" / model_name
    model_cfg = model_dir / "hydra.yaml"
    model_ckpt = model_dir / "checkpoints" / "model_latest.pth"

    return model_name, data_dir, model_cfg, model_ckpt

def load_world_model(
        device: torch.device,
        model_cfg_path: Path,
        model_ckpt_path: Path,
        primitive_action_dim: int,
        expected_action_repeat: Optional[int] = None,
        ):
    #model_cfg = OmegaConf.load(MODEL_CFG)
    model_cfg = OmegaConf.load(model_cfg_path)

    model = load_model(
        #MODEL_CKPT,
        model_ckpt_path,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )
    model.eval()

    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)

    if wm_action_dim % primitive_action_dim != 0:
        raise ValueError(
            f"World model action dim {wm_action_dim} is not divisible by "
            f"primitive action dim {primitive_action_dim}"
        )

    #action_repeat = wm_action_dim // PRIMITIVE_ACTION_DIM
    action_repeat = wm_action_dim // primitive_action_dim

    """"
    if action_repeat != EXPECTED_ACTION_REPEAT:
        raise ValueError(f"Expected action_repeat={EXPECTED_ACTION_REPEAT}, got {action_repeat}")
    """

    if expected_action_repeat is not None and action_repeat != expected_action_repeat:
        raise ValueError(
            f"Expected action_repeat={expected_action_repeat}, got {action_repeat}"
        )

    cfg_frameskip = getattr(model_cfg, "frameskip", None)
    """
    if cfg_frameskip is not None and int(cfg_frameskip) != EXPECTED_ACTION_REPEAT:
        raise ValueError(f"Expected frameskip={EXPECTED_ACTION_REPEAT}, got {cfg_frameskip}")
    """
    if (
        expected_action_repeat is not None
        and cfg_frameskip is not None
        and int(cfg_frameskip) != expected_action_repeat
    ):
        raise ValueError(
            f"Expected frameskip={expected_action_repeat}, got {cfg_frameskip}"
        )

    action_low = torch.full((wm_action_dim,), -3.0, device=device, dtype=torch.float32)
    action_high = torch.full((wm_action_dim,), 3.0, device=device, dtype=torch.float32)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    world_model.world_model.eval()

    print("Loaded DINO-WM")
    print("history_length:", int(world_model.history_length))
    print("world_model action_dim:", wm_action_dim)
    print("primitive_action_dim:", primitive_action_dim)
    print("action_repeat:", action_repeat)
    print("frameskip:", cfg_frameskip)

    return world_model, action_repeat

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episode-start", type=int, required=True)
    p.add_argument("--episode-end", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--norm-stats",
        type=str,
        default=None,
        help="Path to a .pt file with latent_mean/latent_std. "
        "If provided, use those stats instead of computing from this split.",
    )

    p.add_argument(
        "--domain",
        choices=["pusht", "wall_single"],
        default="pusht",
    )
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #world_model, action_repeat = load_world_model(device)

    model_name, data_dir, model_cfg, model_ckpt = resolve_domain_paths(args.domain, split="train")
    expected_repeat = 5 if args.domain == "pusht" else None

    
    world_model, action_repeat = load_world_model(
        device=device,
        model_cfg_path=model_cfg,
        model_ckpt_path=model_ckpt,
        primitive_action_dim=PRIMITIVE_ACTION_DIM,
        expected_action_repeat=expected_repeat,
    )

    if args.domain == "pusht":
        action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
        action_std = ACTION_STD.to(device=device, dtype=torch.float32)
        divide_actions_by = 100 #ACTION_SCALE
    elif args.domain == "wall_single":
        wall_stats = compute_wall_stats(data_dir)
        action_mean = wall_stats["action_mean"].to(device=device, dtype=torch.float32)
        action_std = wall_stats["action_std"].to(device=device, dtype=torch.float32)
        divide_actions_by = 1.0
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    histories, actions, next_latents, past_action_histories = build_feasibility_tensors_from_oracle(
        world_model=world_model,
        data_dir=data_dir,
        episode_indices=range(args.episode_start, args.episode_end),
        action_repeat=action_repeat,
        primitive_action_dim=PRIMITIVE_ACTION_DIM,
        action_mean=action_mean,
        action_std=action_std,
        divide_actions_by=divide_actions_by,
        preserve_latent_shape=True,
        device=device,
        return_past_action_histories=True,
        domain=args.domain,
    )

    expected_action_dim = action_repeat * PRIMITIVE_ACTION_DIM

    if actions.shape[-1] != expected_action_dim:
        raise RuntimeError(f"Expected action dim {expected_action_dim}, got {actions.shape[-1]}")

    if histories.shape[1] != int(world_model.history_length):
        raise RuntimeError(f"Expected history length {int(world_model.history_length)}, " f"got {histories.shape[1]}")

    if not (histories.shape[0] == actions.shape[0] == next_latents.shape[0]):
        raise RuntimeError(
            f"Sample count mismatch: histories={histories.shape}, "
            f"actions={actions.shape}, next_latents={next_latents.shape}"
        )
    if past_action_histories.shape[-2:] != (
        int(world_model.history_length),
        expected_action_dim,
    ):
        raise RuntimeError(
            f"Expected past_action_histories shape [N, {int(world_model.history_length)}, "
            f"{expected_action_dim}], got {tuple(past_action_histories.shape)}"
        )
    
    latent_shape = tuple(next_latents.shape[1:])

    if args.domain == "pusht" and latent_shape != (196, 394):
        raise RuntimeError(f"Expected DINO latent_shape=(196, 394), got {latent_shape}")

    if tuple(histories.shape[2:]) != latent_shape:
        raise RuntimeError(
            f"History latent shape {tuple(histories.shape[2:])} does not match " f"next_latents shape {latent_shape}"
        )

    # use external stats (val set) or compute from this split (train set)
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

    # ensure same device, normalize on CPU
    histories = histories.cpu()
    next_latents = next_latents.cpu()
    latent_mean = latent_mean.cpu()
    latent_std = latent_std.cpu()
    past_action_histories = past_action_histories.cpu()

    histories = (histories - latent_mean.view(1, 1, *latent_shape)) / latent_std.view(1, 1, *latent_shape)
    next_latents = (next_latents - latent_mean.view(1, *latent_shape)) / latent_std.view(1, *latent_shape)

    print("histories:", histories.shape, histories.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("next_latents:", next_latents.shape, next_latents.dtype)
    print("past_action_histories:", past_action_histories.shape, past_action_histories.dtype)

    print("history min/max:", histories.min().item(), histories.max().item())
    print("action min/max:", actions.min().item(), actions.max().item())
    print("next_latents min/max:", next_latents.min().item(), next_latents.max().item())
    print("past_action_histories min/max:",past_action_histories.min().item(),past_action_histories.max().item())

    # change this correspondinly
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    action_mean_save = action_mean.detach().cpu()
    action_std_save = action_std.detach().cpu()
    action_scale_save = float(divide_actions_by)

    torch.save(
        {
            "histories": histories.cpu(),
            "actions": actions.cpu(),
            "next_latents": next_latents.cpu(),
            "past_action_histories": past_action_histories.cpu(),

            "domain": args.domain,
            "latent_shape": latent_shape,
            "preserve_latent_shape": True,
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            
            "action_repeat": action_repeat,
            "primitive_action_dim": PRIMITIVE_ACTION_DIM,
            #"action_scale": ACTION_SCALE,
            "action_scale": action_scale_save,
            #"action_mean": ACTION_MEAN.cpu(),
            "action_mean": action_mean_save,
            #"action_std": ACTION_STD.cpu(),
            "action_std": action_std_save,

            "model_name": model_name,
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
