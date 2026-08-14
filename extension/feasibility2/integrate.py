from __future__ import annotations
from pathlib import Path

import sys
import torch
import torch.nn as nn

DINO_WM_ROOT = Path("/home/nvzutphen/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

from extension.feasibility2 import model
from extension.feasibility2.model import load_feasibility_model_from_checkpoint

"File to make feasibility model integrate with solver & planner"


def _load_checkpoint_stat(
    checkpoint: dict,
    key: str,
) -> torch.Tensor | float | None:
    dataset_metadata = checkpoint.get("dataset_metadata", {})
    metadata = checkpoint.get("metadata", {})

    if key in checkpoint:
        return checkpoint[key]
    if isinstance(dataset_metadata, dict) and key in dataset_metadata:
        return dataset_metadata[key]
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return None


def _normalize_action(action: torch.Tensor, feasibility_model: nn.Module) -> torch.Tensor:
    action_scale = getattr(feasibility_model, "action_scale", 100.0)
    action_mean = getattr(feasibility_model, "action_mean", ACTION_MEAN)
    action_std = getattr(feasibility_model, "action_std", ACTION_STD)

    normalized = action.to(dtype=torch.float32)
    original_shape = normalized.shape

    if action_scale is not None:
        normalized = normalized / float(action_scale)

    mean = None
    std = None
    if action_mean is not None:
        mean = torch.as_tensor(
            action_mean,
            device=normalized.device,
            dtype=normalized.dtype,
        )
    if action_std is not None:
        std = torch.as_tensor(
            action_std,
            device=normalized.device,
            dtype=normalized.dtype,
        ).clamp_min(1e-8)

    if mean is not None or std is not None:
        action_dim = original_shape[-1]
        primitive_dim = None
        if mean is not None and mean.ndim == 1:
            primitive_dim = int(mean.shape[0])
        elif std is not None and std.ndim == 1:
            primitive_dim = int(std.shape[0])

        if primitive_dim is not None and primitive_dim > 0 and action_dim % primitive_dim == 0:
            normalized = normalized.reshape(*original_shape[:-1], action_dim // primitive_dim, primitive_dim)
            if mean is not None:
                normalized = normalized - mean
            if std is not None:
                normalized = normalized / std
            normalized = normalized.reshape(*original_shape)
        else:
            if mean is not None:
                normalized = normalized - mean
            if std is not None:
                normalized = normalized / std

    return normalized

def load_feasibility_scorer(
    checkpoint_path: str | Path,
    device: str | torch.device,
    freeze: bool = True,
) -> nn.Module:
    """Load a trained feasibility model checkpoint.

    Args:
        checkpoint_path: Path to a saved feasibility checkpoint.
        device: Device on which to place the model.

    Returns:
        Loaded feasibility model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = load_feasibility_model_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    latent_mean = _load_checkpoint_stat(checkpoint, "latent_mean")
    latent_std = _load_checkpoint_stat(checkpoint, "latent_std")
    if latent_mean is not None and latent_std is not None:
        model.latent_mean = latent_mean.to(device=device, dtype=torch.float32)
        model.latent_std = latent_std.to(device=device, dtype=torch.float32).clamp_min(1e-6)
    else:
        print(
            "WARNING: checkpoint does not contain latent_mean/latent_std. "
            "Feasibility scoring will use raw planner latents."
        )

    action_scale = _load_checkpoint_stat(checkpoint, "action_scale")  # 100.0
    action_mean = _load_checkpoint_stat(checkpoint, "action_mean")  # PushT mean
    action_std = _load_checkpoint_stat(checkpoint, "action_std")  # PushT std

    if action_mean is not None:
        model.action_mean = torch.as_tensor(
            action_mean,
            device=device,
            dtype=torch.float32,
        )
    if action_std is not None:
        model.action_std = torch.as_tensor(
            action_std,
            device=device,
            dtype=torch.float32,
        ).clamp_min(1e-8)

    print("\nhas latent_mean:", hasattr(model, "latent_mean"))
    print("\nhas latent_std:", hasattr(model, "latent_std"))
    print("has action_scale:", hasattr(model, "action_scale"))
    print("has action_mean:", hasattr(model, "action_mean"))
    print("has action_std:", hasattr(model, "action_std"))
    if hasattr(model, "latent_mean"):
        print("latent_mean shape:", model.latent_mean.shape)
        print("latent_std shape:", model.latent_std.shape)
    if hasattr(model, "action_mean"):
        print("action_mean shape:", model.action_mean.shape)
        print("action_std shape:", model.action_std.shape)

    print("feasibility model class:", type(model))
    print("has predict_delta:", hasattr(model, "predict_delta"))
    print("has transition_penalty:", hasattr(model, "transition_penalty"))
    print("has delta_query:", hasattr(model, "delta_query"))
    print("has delta_out_proj:", hasattr(model, "delta_out_proj"))

    # when to freeze the model? For ALM we need gradflow i believe
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)

    return model


def score_transition_feasibility(
    feasibility_model: nn.Module,
    history: torch.Tensor,
    action: torch.Tensor,
    z_next: torch.Tensor,
    noise_level: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Score one candidate transition using the learned feasibility model.

    Args:
        feasibility_model: Trained feasibility model.
        history: Latent history with shape ``[H, *latent_shape]`` or
            ``[B, H, *latent_shape]``.
        action: Macro action with shape ``[action_dim]`` or ``[B, action_dim]``.
        z_next: Candidate next latent with shape ``[*latent_shape]`` or
            ``[B, *latent_shape]``.
        noise_level: Sigma value used by the feasibility model.
        reduction: Energy reduction mode, usually ``"none"`` or ``"mean"``.

    Returns:
        Feasibility energy. Lower means more feasible.
    """
    latent_mean = getattr(feasibility_model, "latent_mean", None)
    latent_std = getattr(feasibility_model, "latent_std", None)
    if latent_mean is not None and latent_std is not None:
        latent_mean = latent_mean.to(device=z_next.device, dtype=z_next.dtype)
        latent_std = latent_std.to(device=z_next.device, dtype=z_next.dtype)
        if history.ndim == latent_mean.ndim + 1:
            history = (history - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)
        elif history.ndim == latent_mean.ndim + 2:
            history = (history - latent_mean.unsqueeze(0).unsqueeze(0)) / latent_std.unsqueeze(0).unsqueeze(0)
        if z_next.ndim == latent_mean.ndim:
            z_next = (z_next - latent_mean) / latent_std
        elif z_next.ndim == latent_mean.ndim + 1:
            z_next = (z_next - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)

    action = _normalize_action(action, feasibility_model).to(
        device=z_next.device,
        dtype=z_next.dtype,
    )

    return feasibility_model.penalty(
        history,
        action,
        z_next,
        noise_level=noise_level,
        reduction=reduction,
    )

def score_trajectory_feasibility(
    feasibility_model: nn.Module,
    latent_context: torch.Tensor,
    candidate_latents: torch.Tensor,
    candidate_actions: torch.Tensor,
    history_length: int,
    noise_level: float = 0.2,

    reduction: str = "mean",
    lambda_dsm: float = 1.0,
    lambda_contrastive_plan: float = 0.0,
    lambda_transition: float = 10.0,
    lambda_action_consistency: float = 0.0,
    action_consistency_margin: float = 0.1,
    latent_reduction: str = "mean",
    dsm_noise: torch.Tensor | None = None,

    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if dsm_noise is not None:
        expected_shape = (candidate_actions.shape[0],) + tuple(
            candidate_latents.shape[1:]
        )
        if tuple(dsm_noise.shape) != expected_shape:
            raise ValueError(
                "dsm_noise must contain one latent-shaped sample per action: "
                f"got {tuple(dsm_noise.shape)}, expected {expected_shape}"
            )

    full_latents = torch.cat([latent_context, candidate_latents[1:]], dim=0)
    context_len = latent_context.shape[0]

    latent_mean = getattr(feasibility_model, "latent_mean", None)
    latent_std = getattr(feasibility_model, "latent_std", None)

    if latent_mean is not None and latent_std is not None:
        lm = latent_mean.to(
            device=candidate_latents.device,
            dtype=candidate_latents.dtype,
        )
        ls = latent_std.to(
            device=candidate_latents.device,
            dtype=candidate_latents.dtype,
        ).clamp_min(1e-6)
        full_latents_norm = (full_latents - lm) / ls
    else:
        full_latents_norm = full_latents

    # ``candidate_actions`` comes directly from the GVP-WM planner. The
    # planner and feasibility dataset both use the DINO-WM normalized action
    # convention, so applying ``_normalize_action`` here would divide by the
    # environment scale and standardize a second time.
    candidate_actions_norm = candidate_actions.to(
        device=candidate_actions.device,
        dtype=candidate_actions.dtype,
    )

    energies = []
    dsm_energies = []
    contrastive_energies = []
    weighted_contrastive_energies = []
    transition_energies = []
    weighted_transition_energies = []
    action_consistency_energies = []
    """
    lambda_transition = float(getattr(feasibility_model, "lambda_transition", 10.0))
    lambda_action_consistency = float(
        getattr(feasibility_model, "lambda_action_consistency", 0.0)
    )
    action_consistency_margin = float(
        getattr(feasibility_model, "action_consistency_margin", 0.1)
    ) """

    for t in range(candidate_actions.shape[0]):
        end = context_len + t

        history = full_latents_norm[max(0, end - history_length):end]

        if history.shape[0] < history_length:
            pad = history[:1].expand(
                history_length - history.shape[0],
                *history.shape[1:],
            )
            history = torch.cat([pad, history], dim=0)

        action = candidate_actions_norm[t]
        z_next = full_latents_norm[end]
        step_noise = None if dsm_noise is None else dsm_noise[t]

        dsm_energy = feasibility_model.penalty(
            history,
            action,
            z_next,
            noise_level=noise_level,
            reduction="mean",
            eps=step_noise,
        )

        contrastive_energy = history.new_tensor(0.0)
        if lambda_contrastive_plan > 0:
            if not hasattr(feasibility_model, "energy"):
                raise ValueError(
                    "lambda_contrastive_plan is enabled, but the feasibility "
                    "model does not define a scalar energy head."
                )
            contrastive_energy = feasibility_model.energy(
                history,
                action,
                z_next,
                noise_level=noise_level,
                reduction="mean",
            )

        action_consistency_penalty = history.new_tensor(0.0)

        if lambda_action_consistency > 0:
            zero_action = torch.zeros_like(action)

            zero_action_energy = feasibility_model.penalty(
                history,
                zero_action,
                z_next,
                noise_level=noise_level,
                reduction="mean",
                eps=step_noise,
            )

            action_consistency_penalty = torch.relu(
                action_consistency_margin + dsm_energy - zero_action_energy
            )

        transition_energy = history.new_tensor(0.0)

        if hasattr(feasibility_model, "transition_penalty"):
            transition_energy = feasibility_model.transition_penalty(
                history,
                action,
                z_next,
                reduction="mean",
            )

        # look into weighted transition scales
        weighted_transition = lambda_transition * transition_energy
        weighted_contrastive = (
            lambda_contrastive_plan * contrastive_energy
        )
        weighted_action_consistency = (
            lambda_action_consistency * action_consistency_penalty
        )

        energy = (
            lambda_dsm * dsm_energy
            + weighted_contrastive
            + weighted_action_consistency
            + weighted_transition
        )

        dsm_energies.append(dsm_energy)
        contrastive_energies.append(contrastive_energy)
        weighted_contrastive_energies.append(weighted_contrastive)
        transition_energies.append(transition_energy)
        weighted_transition_energies.append(weighted_transition)
        action_consistency_energies.append(weighted_action_consistency)
        energies.append(energy)

        """
        if t == 0:
            print("[feas action debug]")
            print("candidate action:", action.detach().cpu())
            print("candidate action mean/std:", float(action.mean()), float(action.std()))
            if hasattr(feasibility_model, "action_mean"):
                print("model action_mean:", feasibility_model.action_mean.detach().cpu())
            if hasattr(feasibility_model, "action_std"):
                print("model action_std:", feasibility_model.action_std.detach().cpu())
            print(
                "[feas diag] "
                f"dsm_energy={float(dsm_energy.detach().cpu()):.6f} "
                f"transition_energy={float(transition_energy.detach().cpu()):.6f} "
                f"w_transition={float(weighted_transition.detach().cpu()):.6f} "
                f"action_consistency={float(action_consistency_penalty.detach().cpu()):.6f} "
                f"w_action_consistency={float(weighted_action_consistency.detach().cpu()):.6f} "
                f"lambda_transition={lambda_transition:.6f} "
                f"lambda_action_consistency={lambda_action_consistency:.6f}"
            )
        """

    total_energy = torch.stack(energies).mean()

    if not return_diagnostics:
        return total_energy

    diagnostics = {
        "dsm_energy": torch.stack(dsm_energies).mean().detach(),
        "contrastive_energy": torch.stack(contrastive_energies).mean().detach(),
        "weighted_contrastive_energy": torch.stack(
            weighted_contrastive_energies
        ).mean().detach(),
        "transition_energy": torch.stack(transition_energies).mean().detach(),
        "weighted_transition_energy": torch.stack(weighted_transition_energies).mean().detach(),
        "action_consistency_energy": torch.stack(action_consistency_energies).mean().detach(),
        "lambda_transition": lambda_transition,
        "lambda_dsm": lambda_dsm,
        "lambda_contrastive_plan": lambda_contrastive_plan,
        "lambda_action_consistency": lambda_action_consistency,
    }

    return total_energy, diagnostics

    """
        # average over noise levels for more stable signal
        step_energy = torch.stack([
            feasibility_model.penalty(history, action, z_next, noise_level=s, reduction="mean")
            for s in noise_levels
        ]).mean()

        energies.append(step_energy)

    if not energies:
        return candidate_latents.new_tensor(0.0)

    # normalize by feature dims
    D = full_latents.shape[-2] * full_latents.shape[-1]
    return torch.stack(energies).mean() / D
    """
