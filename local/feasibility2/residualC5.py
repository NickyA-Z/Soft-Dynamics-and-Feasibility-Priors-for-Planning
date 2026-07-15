"""
Builds a residual feasibility dataset from DINO-WM oracle episodes.

Instead of training on full next latents:
    (h_t, a_t, z_{t+1})

this builds residual targets:
    delta_t = z_{t+1}^{expert} - f_psi(h_t, a_t)
    this better? \delta_t = z_{t+1}^{expert} - f_\psi(h_t,a_{t-H+1:t-1},a_t)

and saves:
    (h_t, a_t, delta_t)

This is useful as a diagnostic model for world-model prediction drift.
PYTHONPATH=.:/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm \
/home/scur0196/.conda/envs/dino_wm/bin/python -u -m nicky_dl.feasibility2.build_residual
"""

from __future__ import annotations

from pathlib import Path
import sys

from reimplementation.src.gvpwm.examples.dino_oracle_utils import load_oracle_episode
import torch


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

from nicky_dl.feasibility2.dataset import build_feasibility_tensors_from_oracle
from nicky_dl.feasibility2.tests import load_world_model, DATA_DIR, make_macro_actions, make_history, make_past_action_context


def unflatten_latents(flat_latents: torch.Tensor, latent_shape: tuple[int, ...]) -> torch.Tensor:
    """
    Convert flat latents [N, latent_dim] or [N, H, latent_dim]
    back to structured DINO latent shape.
    """
    if flat_latents.ndim == 2:
        return flat_latents.reshape(flat_latents.shape[0], *latent_shape)

    if flat_latents.ndim == 3:
        return flat_latents.reshape(flat_latents.shape[0], flat_latents.shape[1], *latent_shape)

    raise ValueError(f"Unexpected flat_latents shape: {tuple(flat_latents.shape)}")


@torch.no_grad()
def predict_next_latents_from_tuples(
    world_model,
    histories_flat: torch.Tensor,
    actions: torch.Tensor,
    latent_shape: tuple[int, ...],
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Predict f_psi(h_t, a_t) for each saved feasibility tuple.

    Inputs:
        histories_flat: [N, history_length, latent_dim]
        actions:        [N, action_dim]

    Returns:
        pred_next_flat: [N, latent_dim]
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    history_length = int(world_model.history_length)
    if histories_flat.shape[1] != history_length:
        raise ValueError(
            f"history length mismatch: got {histories_flat.shape[1]}, "
            f"expected {history_length}"
        )

    preds = []

    for start in range(0, histories_flat.shape[0], batch_size):
        end = min(start + batch_size, histories_flat.shape[0])

        hist_flat = histories_flat[start:end].to(device)
        act = actions[start:end].to(device)

        # The adapter rollout expects one trajectory at a time:
        # latent_context: [H, ...latent_shape]
        # past_action_context: [H-1, action_dim]
        # planned_actions: [1, action_dim]
        batch_preds = []

        for i in range(hist_flat.shape[0]):
            latent_context = unflatten_latents(
                hist_flat[i],
                latent_shape=latent_shape,
            )

            target_past_len = max(history_length - 1, 0)
            if target_past_len > 0:
                # We only have current tuple action, not the true past action history
                # from this flattened dataset. Use zeros for diagnostic prediction.
                #
                # Better version later: modify dataset builder to also save past_action_context.
                past_action_context = torch.zeros(
                    target_past_len,
                    act.shape[-1],
                    device=device,
                    dtype=act.dtype,
                )
            else:
                past_action_context = torch.zeros(
                    0,
                    act.shape[-1],
                    device=device,
                    dtype=act.dtype,
                )

            planned_actions = act[i : i + 1]

            pred_rollout = world_model.rollout(
                latent_context=latent_context,
                past_action_context=past_action_context,
                planned_actions=planned_actions,
            )

            pred_next = pred_rollout[1]
            batch_preds.append(pred_next.reshape(-1))

        preds.append(torch.stack(batch_preds, dim=0).cpu())

    return torch.cat(preds, dim=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model, action_repeat = load_world_model(device)
    world_model.world_model.eval()
    history_length = int(world_model.history_length)
    latent_shape = (196, 394)
    all_histories = []
    all_actions = []
    all_residuals = []
    ''' -> did not use previous actions 
    histories, actions, next_latents = build_feasibility_tensors_from_oracle(
        world_model=world_model,
        data_dir=DATA_DIR,
        episode_indices=range(0, 10),
        action_repeat=action_repeat,
        primitive_action_dim=2,
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        divide_actions_by=100.0,  # rel_actions / 100, matching DINO-WM training
        device=device,
    )

    # Infer structured DINO latent shape from one encoded sequence.
    latent_dim = next_latents.shape[-1]
    latent_shape = (196, 394)

    if latent_dim != 196 * 394:
        raise ValueError(
            f"Expected latent_dim 196*394={196 * 394}, got {latent_dim}. "
            "Update latent_shape inference."
        )

    pred_next = predict_next_latents_from_tuples(
        world_model=world_model,
        histories_flat=histories,
        actions=actions,
        latent_shape=latent_shape,
        batch_size=16,
        device=device,
    )

    residuals = next_latents - pred_next
    '''

    for episode_idx in range(0, 10):
        episode = load_oracle_episode(DATA_DIR, int(episode_idx))
        latents = world_model.encode_sequence(episode["video_plan"]).detach().to(device)
        latents = latents.reshape(latents.shape[0], *latent_shape)
        macro_actions = make_macro_actions(
            episode["actions"],
            action_repeat=action_repeat,
            divide_by=100.0,
            normalize=True,
            device=device,
        )
        num_macros = min(
            macro_actions.shape[0],
            (latents.shape[0] - 1) // action_repeat,
        )
        if num_macros <= 0:
            continue
        latent_idx = torch.arange(
            0,
            (num_macros + 1) * action_repeat,
            action_repeat,
            device=device,
        )
        macro_latents = latents[latent_idx]
        for t in range(num_macros):
            latent_context = make_history(
                macro_latents,
                t=t,
                history_length=history_length,
            )
            past_action_context = make_past_action_context(
                macro_actions,
                t=t,
                history_length=history_length,
                device=device,
            )
            planned_actions = macro_actions[t : t + 1]
            with torch.no_grad():
                pred_rollout = world_model.rollout(
                    latent_context=latent_context,
                    past_action_context=past_action_context,
                    planned_actions=planned_actions,
                )
            pred_next = pred_rollout[1]
            true_next = macro_latents[t + 1]
            residual = true_next - pred_next
            all_histories.append(latent_context.reshape(history_length, -1).cpu())
            all_actions.append(planned_actions.reshape(-1).cpu())
            all_residuals.append(residual.reshape(-1).cpu())

            del pred_rollout, pred_next, true_next, residual
        # memory issues 
        del episode, latents, macro_latents, macro_actions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    histories = torch.stack(all_histories, dim=0)
    actions = torch.stack(all_actions, dim=0)
    residuals = torch.stack(all_residuals, dim=0)

    residual_mean = residuals.mean(dim=0)
    residual_std = residuals.std(dim=0).clamp_min(1e-6)

    residuals_norm = (residuals - residual_mean.view(1, -1)) / residual_std.view(1, -1)

    # Histories are still used as conditioning. Normalize them using their own stats.
    history_mean = histories.reshape(-1, histories.shape[-1]).mean(dim=0)
    history_std = histories.reshape(-1, histories.shape[-1]).std(dim=0).clamp_min(1e-6)
    histories_norm = (histories - history_mean.view(1, 1, -1)) / history_std.view(1, 1, -1)

    print("histories:", histories_norm.shape, histories_norm.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("residuals:", residuals_norm.shape, residuals_norm.dtype)

    print("history min/max:", histories_norm.min().item(), histories_norm.max().item())
    print("action min/max:", actions.min().item(), actions.max().item())
    print("residual raw min/max:", residuals.min().item(), residuals.max().item())
    print("residual norm min/max:", residuals_norm.min().item(), residuals_norm.max().item())
    print("residual raw mean/std:", residuals.mean().item(), residuals.std().item())

    # change this when needed
    out_path = Path("data/feasibility_residual_smoke_norm.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "histories": histories_norm.cpu(),
            "actions": actions.cpu(),
            # Reuse existing train.py naming: the target is residual, stored as next_latents.
            "next_latents": residuals_norm.cpu(),
            "target_type": "residual",
            "history_mean": history_mean.cpu(),
            "history_std": history_std.cpu(),
            "residual_mean": residual_mean.cpu(),
            "residual_std": residual_std.cpu(),
            "latent_shape": latent_shape,
        },
        out_path,
    )

    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()