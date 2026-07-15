"""
Builds the feasibility dataset from DINO-WM oracle episodes. 
Loads the pretrained DINO world model, 
encodes expert videos into latent states, 
applies the same relative-action preprocessing as DINO-WM, and saves tuples:

(h_t, a_t, z_{t+1})
"""

from pathlib import Path
import sys
import torch

DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from nicky_dl.feasibility2.dataset import (
    build_feasibility_tensors_from_oracle,
    save_tensor_dataset,
)
from nicky_dl.feasibility2.tests import load_world_model, DATA_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model, action_repeat = load_world_model(device)

    histories, actions, next_latents = build_feasibility_tensors_from_oracle(
        world_model=world_model,
        data_dir=DATA_DIR,
        episode_indices=range(0, 10),  # smoke test first
        action_repeat=action_repeat,
        primitive_action_dim=2,
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        divide_actions_by=100.0,   # rel_actions / 100, matching DINO-WM training
        device=device,
    )
    latent_mean = next_latents.mean(dim=0)
    latent_std = next_latents.std(dim=0).clamp_min(1e-6)

    histories = (histories - latent_mean.view(1, 1, -1)) / latent_std.view(1, 1, -1)
    next_latents = (next_latents - latent_mean.view(1, -1)) / latent_std.view(1, -1)

    print("histories:", histories.shape, histories.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("next_latents:", next_latents.shape, next_latents.dtype)

    print("history min/max:", histories.min().item(), histories.max().item())
    print("action min/max:", actions.min().item(), actions.max().item())
    print("next_latents min/max:", next_latents.min().item(), next_latents.max().item())

    out_path = Path("data/feasibility_tensors_smoke_norm.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "histories": histories.cpu(),
            "actions": actions.cpu(),
            "next_latents": next_latents.cpu(),
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
        },
        out_path,
    )
    """
    out_path = Path("data/feasibility_tensors_smoke.pt")
    save_tensor_dataset(out_path, histories, actions, next_latents)
    """
    print(f"saved to: {out_path}")
    


if __name__ == "__main__":
    main()