from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

# should be changed to call from reimplementation folder??
from ..examples.dino_oracle_utils import load_oracle_episode
from ..adapters.dino_wm import DinoWorldModelAdapter


DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"

MODEL_DIR = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
MODEL_CFG = MODEL_DIR / "hydra.yaml"
MODEL_CKPT = MODEL_DIR / "checkpoints" / "model_latest.pth"

PRIMITIVE_ACTION_DIM = 2


def make_macro_actions(
    primitive_actions: torch.Tensor,
    action_repeat: int,
    divide_by: float | None,
    normalize: bool,
    device: torch.device,
) -> torch.Tensor:
    actions = primitive_actions.to(device=device, dtype=torch.float32)

    usable = (actions.shape[0] // action_repeat) * action_repeat
    actions = actions[:usable]

    if actions.numel() == 0:
        return actions.new_zeros((0, action_repeat * PRIMITIVE_ACTION_DIM))

    actions = actions.reshape(-1, action_repeat, PRIMITIVE_ACTION_DIM)

    if divide_by is not None:
        actions = actions / float(divide_by)

    if normalize:
        mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
        std = ACTION_STD.to(device=device, dtype=torch.float32).clamp_min(1e-8)
        actions = (actions - mean) / std

    return actions.reshape(-1, action_repeat * PRIMITIVE_ACTION_DIM)


def make_history(latents: torch.Tensor, t: int, history_length: int) -> torch.Tensor:
    """
    Build a padded latent history.

    Input latents should stay in the unflattened DINO-WM latent shape:
        [T, ...latent_dims]

    Output:
        [history_length, ...latent_dims]
    """
    start = max(0, t - history_length + 1)
    hist = latents[start : t + 1]

    if hist.shape[0] < history_length:
        pad = hist[:1].repeat(
            history_length - hist.shape[0],
            *([1] * (hist.ndim - 1)),
        )
        hist = torch.cat([pad, hist], dim=0)

    return hist


def make_past_action_context(
    macro_actions: torch.Tensor,
    t: int,
    history_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build padded past action context.

    Expected output:
        [history_length - 1, action_dim]
    """
    target_past_len = max(history_length - 1, 0)

    if target_past_len == 0:
        return macro_actions.new_zeros((0, macro_actions.shape[-1])).to(device)

    start_a = max(0, t - target_past_len)
    past_action_context = macro_actions[start_a:t].to(device)

    if past_action_context.shape[0] < target_past_len:
        missing = target_past_len - past_action_context.shape[0]

        if past_action_context.shape[0] == 0:
            pad = torch.zeros(
                missing,
                macro_actions.shape[-1],
                device=device,
                dtype=macro_actions.dtype,
            )
        else:
            pad = past_action_context[:1].repeat(missing, 1)

        past_action_context = torch.cat([pad, past_action_context], dim=0)

    return past_action_context


@torch.no_grad()
def compute_one_step_error(
    world_model,
    macro_latents: torch.Tensor,
    macro_actions: torch.Tensor,
    candidate_name: str,
    debug_shapes: bool = False,
) -> float:
    """
    Compare one-step DINO-WM predictions against expert macro latents.

    Computes:

        mean_t || f_psi(h_t, a_t) - z_{t+1}^{expert} ||^2

    Latents are kept unflattened for DINO-WM rollout and flattened only for MSE.
    """
    device = macro_latents.device
    history_length = int(world_model.history_length)

    num_steps = min(macro_actions.shape[0], macro_latents.shape[0] - 1)

    if num_steps <= 0:
        raise RuntimeError(
            f"No valid rollout-error steps for {candidate_name}: "
            f"macro_actions={tuple(macro_actions.shape)}, "
            f"macro_latents={tuple(macro_latents.shape)}"
        )

    errors: list[float] = []

    for t in range(num_steps):
        history = make_history(macro_latents, t, history_length).to(device)
        past_action_context = make_past_action_context(
            macro_actions=macro_actions,
            t=t,
            history_length=history_length,
            device=device,
        )
        planned_action = macro_actions[t : t + 1].to(device)
        true_next = macro_latents[t + 1 : t + 2].to(device)

        if debug_shapes and t == 0:
            print(f"\nShape debug for candidate: {candidate_name}")
            print("history shape:", tuple(history.shape))
            print("past_action_context shape:", tuple(past_action_context.shape))
            print("planned_action shape:", tuple(planned_action.shape))
            print("true_next shape:", tuple(true_next.shape))

        pred_rollout = world_model.rollout(
            latent_context=history,
            past_action_context=past_action_context,
            planned_actions=planned_action,
        )

        if debug_shapes and t == 0:
            print("pred_rollout shape:", tuple(pred_rollout.shape))

        if pred_rollout.shape[0] < 2:
            raise RuntimeError(
                f"world_model.rollout returned too few latents: "
                f"{tuple(pred_rollout.shape)}. Expected at least [2, ...]."
            )

        pred_next = pred_rollout[1:2]

        if pred_next.shape != true_next.shape:
            try:
                pred_next = pred_next.reshape(true_next.shape)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cannot reshape pred_next from {tuple(pred_next.shape)} "
                    f"to true_next shape {tuple(true_next.shape)}"
                ) from exc

        pred_next_flat = pred_next.reshape(pred_next.shape[0], -1)
        true_next_flat = true_next.reshape(true_next.shape[0], -1)

        mse = torch.mean((pred_next_flat - true_next_flat) ** 2)
        errors.append(float(mse.item()))

    return sum(errors) / len(errors)


@torch.no_grad()
def check_episode(
    world_model: DinoWorldModelAdapter,
    episode_idx: int,
    action_repeat: int,
    device: torch.device,
    debug_shapes: bool = False,
) -> dict[str, float]:
    episode = load_oracle_episode(DATA_DIR, episode_idx)

    video = episode["video_plan"]
    latents = world_model.encode_sequence(video).detach().to(device)

    abs_actions_all = torch.load(DATA_DIR / "abs_actions.pth", map_location="cpu")
    rel_actions_all = torch.load(DATA_DIR / "rel_actions.pth", map_location="cpu")

    abs_actions = abs_actions_all[episode_idx]
    rel_actions = rel_actions_all[episode_idx]

    wm_action_dim = int(world_model.action_dim)

    num_macro_latents = (latents.shape[0] - 1) // action_repeat + 1
    macro_latent_indices = torch.arange(
        0,
        num_macro_latents * action_repeat,
        action_repeat,
        device=device,
    )
    macro_latent_indices = macro_latent_indices[macro_latent_indices < latents.shape[0]]
    macro_latents = latents[macro_latent_indices]

    candidates = {
        "abs_div100_norm": make_macro_actions(
            abs_actions,
            action_repeat=action_repeat,
            divide_by=100.0,
            normalize=True,
            device=device,
        ),
        "abs_no_div_norm": make_macro_actions(
            abs_actions,
            action_repeat=action_repeat,
            divide_by=None,
            normalize=True,
            device=device,
        ),
        "rel_div100_norm": make_macro_actions(
            rel_actions,
            action_repeat=action_repeat,
            divide_by=100.0,
            normalize=True,
            device=device,
        ),
        "rel_no_div_norm": make_macro_actions(
            rel_actions,
            action_repeat=action_repeat,
            divide_by=None,
            normalize=True,
            device=device,
        ),
        "rel_no_div_no_norm": make_macro_actions(
            rel_actions,
            action_repeat=action_repeat,
            divide_by=None,
            normalize=False,
            device=device,
        ),
    }

    print(f"\nEpisode {episode_idx}")
    print(f"video_plan length: {len(episode['video_plan'])}")
    print(f"first video_plan item type: {type(episode['video_plan'][0])}")
    print(f"latents shape: {tuple(latents.shape)}")
    print(f"macro_latents shape: {tuple(macro_latents.shape)}")
    print(f"action_repeat: {action_repeat}")
    print(f"world_model action_dim: {wm_action_dim}")

    if len(episode["video_plan"]) != latents.shape[0]:
        print(
            "WARNING: video_plan length and encoded latent length differ: "
            f"{len(episode['video_plan'])} vs {latents.shape[0]}"
        )

    results: dict[str, float] = {}

    for name, actions in candidates.items():
        if actions.shape[-1] != wm_action_dim:
            print(f"{name}: skipped, action dim {actions.shape[-1]} != {wm_action_dim}")
            continue

        max_steps = min(actions.shape[0], macro_latents.shape[0] - 1)
        if max_steps <= 0:
            print(f"{name}: skipped, no valid macro steps")
            continue

        err = compute_one_step_error(
            world_model=world_model,
            macro_latents=macro_latents,
            macro_actions=actions,
            candidate_name=name,
            debug_shapes=debug_shapes,
        )

        results[name] = err
        print(f"{name:20s}: {err:.8f}")

    if not results:
        raise RuntimeError("No candidate action format produced valid results.")

    best_name, best_err = min(results.items(), key=lambda item: item[1])
    print(f"BEST: {best_name} | error={best_err:.8f}")

    return results


def load_world_model(device: torch.device) -> tuple[DinoWorldModelAdapter, int]:
    model_cfg = OmegaConf.load(MODEL_CFG)

    model = load_model(
        MODEL_CKPT,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )
    model.eval()

    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
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

    import inspect

    print("rollout signature:")
    print(inspect.signature(world_model.rollout))
    print("history_length:", int(world_model.history_length))
    print("action_dim:", int(world_model.action_dim))
    print("action_repeat:", int(action_repeat))

    return world_model, action_repeat


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)

    world_model, action_repeat = load_world_model(device)

    all_results: list[dict[str, float]] = []

    # Start with one episode. Add more after the first run works.
    #episode_indices = [0]
    episode_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for episode_idx in episode_indices:
        results = check_episode(
            world_model=world_model,
            episode_idx=episode_idx,
            action_repeat=action_repeat,
            device=device,
            debug_shapes=True,
        )
        all_results.append(results)

    print("\n=== Average errors ===")

    keys = sorted(set().union(*(r.keys() for r in all_results)))
    averages: dict[str, float] = {}

    for key in keys:
        vals = [r[key] for r in all_results if key in r]
        averages[key] = sum(vals) / len(vals)
        print(f"{key:20s}: {averages[key]:.8f}")

    best_name, best_err = min(averages.items(), key=lambda item: item[1])

    print("\n=== Final recommendation ===")
    print(f"Use: {best_name}")
    print(f"Average one-step latent error: {best_err:.8f}")


if __name__ == "__main__":
    main()