"""
dino into training samples 
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from gvpwm.task.pusht.pusht_utils import load_oracle_episode
from gvpwm.task.wall.wall_utils import (
    compute_wall_stats, load_wall_oracle_episode, slice_wall_oracle_episode,
)

PRIMITIVE_ACTION_DIM = 2
ACTION_REPEAT = 5
ACTION_SCALE = 100.0

class FeasibilityDataset(Dataset):
    """PyTorch dataset of expert feasibility transition tuples.

    Each sample contains:
        - history: [history_length, *latent_shape]
        - action: [action_repeat * primitive_action_dim]
        - next_latent: [*latent_shape]

    For the MLP pipeline, latent_shape is usually [latent_dim].
    For the transformer pipeline, latent_shape can preserve the original DINO token
    or spatial latent shape.
    """

    def __init__(self, histories: torch.Tensor, actions: torch.Tensor, next_latents: torch.Tensor, past_action_histories: torch.Tensor | None = None,) -> None:
        if not (histories.shape[0] == actions.shape[0] == next_latents.shape[0]):
            raise ValueError("histories, actions, and next_latents must have the same first dimension")
        self.histories = histories.float()
        self.actions = actions.float()
        self.next_latents = next_latents.float()
        self.past_action_histories = (past_action_histories.float() if past_action_histories is not None else None)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, idx: int):
        if self.past_action_histories is not None:
            return self.histories[idx], self.actions[idx], self.next_latents[idx], self.past_action_histories[idx]
        
        return self.histories[idx], self.actions[idx], self.next_latents[idx]


def make_history_windows(latents: torch.Tensor, history_length: int) -> torch.Tensor:
    """Build padded history windows while preserving DINO latent token shape.

    Input:
        latents: [T + 1, *latent_shape]

    Output:
        histories: [T, history_length, *latent_shape]

    Example:
        If latents are [T + 1, N, D], histories become [T, H, N, D].
        If latents are [T + 1, C, H, W], histories become [T, H_hist, C, H, W].
    """
    if latents.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 latents to build history windows, got {latents.shape[0]}"
        )

    windows = []
    # ToDO need to look into 
    for i in range(latents.shape[0] - 1):
        start = max(0, i - history_length + 1)
        hist = latents[start : i + 1]
        if hist.shape[0] < history_length:
            pad_shape = [history_length - hist.shape[0]] + list(hist.shape[1:])
            pad = hist[:1].expand(*pad_shape)
            hist = torch.cat([pad, hist], dim=0)
        windows.append(hist)
    return torch.stack(windows, dim=0)

def make_action_history_windows(
    macro_actions: torch.Tensor,
    history_length: int,
    pad_mode: str = "zeros",
) -> torch.Tensor:
    """Build padded action-history windows.

    Input:
        macro_actions: [T, action_dim]

    Output:
        action_histories: [T, history_length, action_dim]

    For sample t, the returned window includes actions up to and including a_t.
    This matches the action window shape expected by DINO-WM predict_next_latent.
    """
    if macro_actions.ndim != 2:
        raise ValueError(
            f"Expected macro_actions shape [T, action_dim], got {tuple(macro_actions.shape)}"
        )

    if history_length <= 0:
        raise ValueError(f"history_length must be positive, got {history_length}")

    windows = []
    action_dim = macro_actions.shape[-1]

    for t in range(macro_actions.shape[0]):
        start = max(0, t - history_length + 1)
        hist = macro_actions[start : t + 1]

        missing = history_length - hist.shape[0]
        if missing > 0:
            if pad_mode == "zeros":
                pad = macro_actions.new_zeros(missing, action_dim)
            elif pad_mode == "repeat_first":
                pad = hist[:1].expand(missing, -1)
            else:
                raise ValueError(f"Unknown pad_mode={pad_mode!r}")
            hist = torch.cat([pad, hist], dim=0)

        windows.append(hist)

    return torch.stack(windows, dim=0)

def make_macro_actions(
    primitive_actions: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int = 2,
    action_mean: torch.Tensor | float | None = ACTION_MEAN,
    action_std: torch.Tensor | float | None = ACTION_STD,
    divide_by: float | None = ACTION_SCALE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert primitive Push-T actions to DINO-WM macro actions.

    This keeps action scaling explicit. Use the same divide_by/action_mean/action_std
    values here as your ALM planner/world model uses.
    """
    device = torch.device(device) if device is not None else primitive_actions.device
    actions = primitive_actions.to(device=device, dtype=torch.float32)

    if action_repeat <= 0:
        raise ValueError(f"action_repeat must be positive, got {action_repeat}")

    if actions.ndim != 2:
        raise ValueError(
            f"Expected primitive_actions shape [T, action_dim], got {tuple(actions.shape)}"
        )

    if actions.shape[-1] != primitive_action_dim:
        raise ValueError(
            f"Expected primitive action dim {primitive_action_dim}, got {actions.shape[-1]}"
        )

    usable = (actions.shape[0] // action_repeat) * action_repeat
    actions = actions[:usable]
    if actions.numel() == 0:
        return actions.new_zeros((0, action_repeat * primitive_action_dim))
    actions = actions.reshape(-1, action_repeat, primitive_action_dim)

    if divide_by is not None:
        actions = actions / float(divide_by)
    if action_mean is not None:
        mean = torch.as_tensor(action_mean, device=device, dtype=torch.float32)
        actions = actions - mean
    if action_std is not None:
        std = torch.as_tensor(action_std, device=device, dtype=torch.float32).clamp_min(1e-8)
        actions = actions / std

    return actions.reshape(-1, action_repeat * primitive_action_dim)

def load_domain_episode_and_actions(
    domain: str,
    data_dir: str | Path,
    episode_idx: int,
    action_repeat: int,
) -> tuple[dict, torch.Tensor]:
    data_dir = Path(data_dir)

    if domain == "pusht":
        if load_oracle_episode is None:
            raise ImportError("Could not import Push-T load_oracle_episode.")

        episode = load_oracle_episode(data_dir, int(episode_idx))

        rel_actions_all = torch.load(data_dir / "rel_actions.pth", map_location="cpu")
        primitive_actions = rel_actions_all[int(episode_idx)].float()

        return episode, primitive_actions

    if domain == "wall_single":
        if load_wall_oracle_episode is None or slice_wall_oracle_episode is None:
            raise ImportError("Could not import Wall oracle utilities.")

        stats = compute_wall_stats(data_dir)
        episode = load_wall_oracle_episode(data_dir, int(episode_idx), stats=stats)

        # Use the full episode. The builder will trim to valid macro transitions.
        primitive_actions = episode["actions"].float()

        return episode, primitive_actions

    raise ValueError(f"Unsupported domain: {domain}")

@torch.no_grad()
def build_feasibility_tensors_from_oracle(
    world_model,
    data_dir: str | Path,
    episode_indices: Iterable[int],
    action_repeat: int = ACTION_REPEAT,
    primitive_action_dim: int = PRIMITIVE_ACTION_DIM,
    action_mean: torch.Tensor | float | None = ACTION_MEAN,
    action_std: torch.Tensor | float | None = ACTION_STD,
    divide_actions_by: float | None = ACTION_SCALE,
    device: torch.device | str | None = None,
    preserve_latent_shape: bool = False, # true for transformer false for MLP
    
    return_past_action_histories: bool = False,
    action_history_pad_mode: str = "zeros", 
    domain: str = "pusht",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode oracle episodes and return (histories, actions, next_latents).

    Works with DinoWorldModelAdapter used by plannerALM/solverALM. It does not
    import or modify the ALM solver.
    """
    if domain == "pusht" and load_oracle_episode is None:
        raise ImportError("Could not import load_oracle_episode. Put dino_oracle_utils.py on PYTHONPATH.")

    device = torch.device(device) if device is not None else world_model.device
    history_length = int(getattr(world_model, "history_length", 1))

    all_histories: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_next: list[torch.Tensor] = []
    all_past_action_histories: list[torch.Tensor] = []

    """
    rel_actions_all = torch.load(Path(data_dir) / "rel_actions.pth", map_location="cpu")
    if rel_actions_all.shape[-1] != primitive_action_dim:
        raise ValueError(
            f"rel_actions.pth has primitive action dim {rel_actions_all.shape[-1]}, "
            f"expected {primitive_action_dim}"
        )
    """

    expected_action_dim = action_repeat * primitive_action_dim
    world_model_action_dim = int(getattr(world_model, "action_dim", expected_action_dim))

    if world_model_action_dim != expected_action_dim:
        raise ValueError(
            f"world_model.action_dim={world_model_action_dim}, but expected "
            f"{expected_action_dim} from action_repeat={action_repeat} and "
            f"primitive_action_dim={primitive_action_dim}"
        )

    for episode_idx in episode_indices:
        episode, primitive_actions = load_domain_episode_and_actions(
            domain=domain,
            data_dir=data_dir,
            episode_idx=int(episode_idx),
            action_repeat=action_repeat,
        )
        if primitive_actions.shape[-1] != primitive_action_dim:
            raise ValueError(
                f"Episode {episode_idx} has primitive action dim {primitive_actions.shape[-1]}, "
                f"expected {primitive_action_dim}"
            )
        #episode = load_oracle_episode(data_dir, int(episode_idx))
        if int(episode_idx) == 0:
            print("DEBUG episode actions first 5:")
            print(episode["actions"][:5])
            print("DEBUG episode actions shape:", episode["actions"].shape)
            print("DEBUG episode actions dtype:", episode["actions"].dtype)
            print("DEBUG episode actions min/max:",
                episode["actions"].min().item(),
                episode["actions"].max().item())
            
        latents = world_model.encode_sequence(episode["video_plan"]).detach().to(device=device, dtype=torch.float32)
        #latent_shape = tuple(latents.shape[1:])

        if not preserve_latent_shape:
            latents = latents.reshape(latents.shape[0], -1)

        #primitive_actions = rel_actions_all[int(episode_idx)]
        macro_actions = make_macro_actions(
            #episode["actions"],
            primitive_actions,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
            divide_by=divide_actions_by,
            device=device,
        )

        if int(episode_idx) == 0 and macro_actions.shape[0] > 0:
            print("DEBUG macro_actions first row:")
            print(macro_actions[0])
            print("DEBUG macro_actions shape:", macro_actions.shape)
            print("DEBUG macro_actions min/max:",
                macro_actions.min().item(),
                macro_actions.max().item())

        num_macros = min(macro_actions.shape[0], (latents.shape[0] - 1) // action_repeat)
        if num_macros <= 0:
            continue

        latent_idx = torch.arange(0, (num_macros + 1) * action_repeat, action_repeat, device=device)
        macro_latents = latents[latent_idx]
        histories = make_history_windows(macro_latents, history_length=history_length)
        
        past_action_histories = make_action_history_windows(
            macro_actions[:num_macros],
            history_length=history_length,
            pad_mode=action_history_pad_mode,
        )
        if past_action_histories.shape != (num_macros, history_length, expected_action_dim):
            raise RuntimeError(
                f"past_action_histories shape mismatch in episode {episode_idx}: "
                f"got {tuple(past_action_histories.shape)}, "
                f"expected {(num_macros, history_length, expected_action_dim)}"
            )

        if histories.shape[0] != num_macros:
            raise RuntimeError(
                f"Mismatch in episode {episode_idx}: "
                f"histories={histories.shape[0]}, actions={num_macros}"
            )

        if all_next and macro_latents.shape[1:] != all_next[0].shape[1:]:
            raise RuntimeError(
                f"Latent shape mismatch in episode {episode_idx}: "
                f"got {tuple(macro_latents.shape[1:])}, "
                f"expected {tuple(all_next[0].shape[1:])}"
            )

        all_histories.append(histories.cpu())
        all_actions.append(macro_actions[:num_macros].cpu())
        all_next.append(macro_latents[1 : num_macros + 1].cpu())
        all_past_action_histories.append(past_action_histories.cpu())

    if not all_histories:
        raise RuntimeError("No valid feasibility tuples were built.")

    if return_past_action_histories:
        past_action_histories_out = torch.cat(all_past_action_histories, dim=0)
        return torch.cat(all_histories, dim=0), torch.cat(all_actions, dim=0), torch.cat(all_next, dim=0), past_action_histories_out
    
    return torch.cat(all_histories, dim=0), torch.cat(all_actions, dim=0), torch.cat(all_next, dim=0)


def save_tensor_dataset(path: str | Path, histories: torch.Tensor, actions: torch.Tensor, next_latents: torch.Tensor,past_action_histories: torch.Tensor | None = None,) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if past_action_histories is not None:
        torch.save({
            "histories": histories.cpu(),
            "actions": actions.cpu(),
            "next_latents": next_latents.cpu(),
            "past_action_histories": past_action_histories.cpu(),
        }, path)
    else:
        torch.save({"histories": histories.cpu(), "actions": actions.cpu(), "next_latents": next_latents.cpu()}, path)

"""
def load_tensor_dataset(path: str | Path) -> FeasibilityDataset:
    data = torch.load(path, map_location="cpu")
    return FeasibilityDataset(data["histories"], data["actions"], data["next_latents"])
"""

def load_tensor_dataset(
    path: str | Path,
) -> tuple[FeasibilityDataset, dict]:
    data = torch.load(path, map_location="cpu")

    dataset = FeasibilityDataset(
        data["histories"],
        data["actions"],
        data["next_latents"],
        past_action_histories=data.get("past_action_histories", None),
    )

    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"histories", "actions", "next_latents","past_action_histories"}
    }

    return dataset, metadata
