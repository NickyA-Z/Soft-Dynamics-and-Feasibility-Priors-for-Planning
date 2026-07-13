"""Diagnostic experiments to disentangle ALM warm-start vs residual-scale issues.

Experiment A (warm-start): run a SINGLE solver.solve() call (no MPC, no env step)
on one episode, comparing two action initializations:
  (A0) zeros (paper default)
  (A1) expert macro actions

We log the inner-loop residual / goal / video losses at every iteration so we can
see whether ALM, when STARTED from a feasible point, keeps residual low and lets
goal_loss drop, or whether it drifts away regardless of initialization.

Experiment B (latent scale): measure DINO-WM latent norms (per-token, per-frame)
along the oracle video, plus the one-step dynamics residual obtained by rolling
out expert macro actions through the world model. This tells us the natural
magnitude of ||L_dyn|| in our setup and whether rho_max=1000 is in the right
ballpark.

Usage:
    python -m src.gvpwm.examples.dino_oracle_diagnose \
        --split val --horizon 25 --episode-idx 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..adapters.dino_wm import DinoWorldModelAdapter
from ..planner import GVPWMPlanner
from ..solver import LatentCollocationSolver
from ..video import temporal_resample_sequence
from ..utils import ensure_history_length
from .dino_oracle_demo import (
    DATA_ROOT,
    FRAME_SKIP,
    DEFAULT_HORIZON,
    DEFAULT_SPLIT,
    load_model_once,
)
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD


def build_alm_config(diagnostic: bool = True) -> ALMConfig:
    return ALMConfig(
        inner_steps=25,
        outer_steps=25,
        learning_rate=0.05,
        rho_init=1.0,
        rho_growth=1.9,
        rho_max=1_000.0,
        lambda_video=1.0,
        lambda_goal=10.0,
        lambda_action=0.05,
        use_video_init=True,
        use_video_loss=True,
        fix_states_to_video=False,
        use_action_reparameterization=True,
        diagnostic_inner_interval=5 if diagnostic else None,
        diagnostic_outer=diagnostic,
    )


def make_world_model(model, model_cfg, device):
    primitive_action_dim = model_cfg.action_emb_dim if hasattr(model_cfg, "action_emb_dim") else 2
    primitive_action_dim = 2  # pusht xy delta
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim

    rel_actions = torch.load(DATA_ROOT / DEFAULT_SPLIT / "rel_actions.pth").float() / 100.0
    rel_actions = (rel_actions - ACTION_MEAN) / ACTION_STD
    primitive_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32)
    primitive_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    return world_model, primitive_action_dim, action_repeat


def expert_macro_actions(
    episode: dict,
    action_repeat: int,
    primitive_action_dim: int,
    horizon: int,
) -> torch.Tensor:
    """Pack expert primitive rel_actions into world-model macro actions of
    shape [horizon, wm_action_dim] using the same normalization as training."""
    rel = episode["rel_actions"][: horizon * action_repeat].float() / 100.0
    rel = (rel - ACTION_MEAN) / ACTION_STD
    rel = rel.reshape(horizon, action_repeat * primitive_action_dim)
    return rel


def encode_video(world_model, episode, horizon, device):
    video = world_model.encode_sequence(episode["video_plan"]).to(device)
    video = temporal_resample_sequence(video, target_length=horizon + 1)
    return video


def latent_context_from_start(world_model, episode, device):
    latent_context = world_model.encode_sequence([episode["start_obs"]]).to(device)
    latent_context = ensure_history_length(
        latent_context,
        world_model.history_length,
        pad_mode="repeat_first",
    )
    return latent_context


def experiment_a_warmstart(world_model, solver, episode, horizon, device):
    print("\n================ EXPERIMENT A: warm-start vs zero-init ================")
    latent_context = latent_context_from_start(world_model, episode, device)
    past_action_history = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=device,
    )
    encoded_video = encode_video(world_model, episode, horizon, device)
    goal_latent = world_model.encode_observation(episode["goal_obs"]).to(device)

    primitive_action_dim = 2
    action_repeat = world_model.action_dim // primitive_action_dim
    expert = expert_macro_actions(episode, action_repeat, primitive_action_dim, horizon).to(device)

    print(f"\n--- A0: zero-init actions (paper default) ---")
    res_zero = solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_history,
        goal_latent=goal_latent,
        video_latents=encoded_video,
        warm_start_actions=None,
        warm_start_latents=None,
    )
    print(
        f"[A0 final] residual={res_zero.dynamics_residual_norm:.4f} "
        f"goal={res_zero.diagnostics.get('goal_loss', float('nan')):.6f} "
        f"video={res_zero.diagnostics.get('video_loss', float('nan')):.6f} "
        f"rho={res_zero.rho:.2f}"
    )

    print(f"\n--- A1: expert action warm-start ---")
    res_expert = solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_history,
        goal_latent=goal_latent,
        video_latents=encoded_video,
        warm_start_actions=expert,
        warm_start_latents=None,
    )
    print(
        f"[A1 final] residual={res_expert.dynamics_residual_norm:.4f} "
        f"goal={res_expert.diagnostics.get('goal_loss', float('nan')):.6f} "
        f"video={res_expert.diagnostics.get('video_loss', float('nan')):.6f} "
        f"rho={res_expert.rho:.2f}"
    )

    print("\n--- A summary ---")
    print(
        f"zero-init  : final_residual={res_zero.dynamics_residual_norm:.4f}  "
        f"goal_loss={res_zero.diagnostics['goal_loss']:.6f}"
    )
    print(
        f"expert-init: final_residual={res_expert.dynamics_residual_norm:.4f}  "
        f"goal_loss={res_expert.diagnostics['goal_loss']:.6f}"
    )
    if res_expert.dynamics_residual_norm < 0.1 * res_zero.dynamics_residual_norm:
        print(">> verdict: warm-start drastically helps -> bottleneck is INITIALIZATION (basin).")
    elif res_expert.dynamics_residual_norm > 0.5 * res_zero.dynamics_residual_norm:
        print(">> verdict: warm-start barely helps -> bottleneck is ALM SCALE (rho/penalty).")
    else:
        print(">> verdict: mixed signal — both factors contribute.")


def experiment_b_scale(world_model, episode, horizon, device):
    print("\n================ EXPERIMENT B: latent scale measurement ================")
    encoded_video = encode_video(world_model, episode, horizon, device)
    print(f"video latent shape per frame: {tuple(encoded_video.shape[1:])}")

    flat = encoded_video.reshape(encoded_video.shape[0], -1)
    per_frame_norm = flat.norm(dim=1)
    print(
        f"per-frame ||z||_2: mean={per_frame_norm.mean().item():.3f} "
        f"min={per_frame_norm.min().item():.3f} "
        f"max={per_frame_norm.max().item():.3f}"
    )

    if encoded_video.dim() >= 3:
        token_flat = encoded_video.reshape(encoded_video.shape[0], encoded_video.shape[1], -1)
        token_norm = token_flat.norm(dim=2)
        print(
            f"per-token ||z_i||_2: mean={token_norm.mean().item():.4f} "
            f"std={token_norm.std().item():.4f} "
            f"max={token_norm.max().item():.4f}"
        )
        print(f"num tokens per frame: {token_flat.shape[1]}, dim per token: {token_flat.shape[2]}")

    diffs = encoded_video[1:] - encoded_video[:-1]
    diff_norm = diffs.reshape(diffs.shape[0], -1).norm(dim=1)
    print(
        f"||z_{{t+1}} - z_t||_2 along oracle video: "
        f"mean={diff_norm.mean().item():.4f} "
        f"max={diff_norm.max().item():.4f}"
    )

    primitive_action_dim = 2
    action_repeat = world_model.action_dim // primitive_action_dim
    expert = expert_macro_actions(episode, action_repeat, primitive_action_dim, horizon).to(device)
    latent_context = latent_context_from_start(world_model, episode, device)
    past_action_history = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=device,
    )
    rollout = world_model.rollout(
        latent_context=latent_context,
        past_action_context=past_action_history,
        planned_actions=expert,
    )
    print(f"expert rollout latent shape: {tuple(rollout.shape)}")
    rollout_flat = rollout.reshape(rollout.shape[0], -1)
    rollout_norm = rollout_flat.norm(dim=1)
    print(
        f"per-frame ||z||_2 of EXPERT ROLLOUT: mean={rollout_norm.mean().item():.3f} "
        f"max={rollout_norm.max().item():.3f}"
    )

    pair_diff = (rollout - encoded_video[: rollout.shape[0]]).reshape(rollout.shape[0], -1)
    pair_norm = pair_diff.norm(dim=1)
    print(
        f"||rollout_t - video_t||_2 (expert vs oracle video latent): "
        f"mean={pair_norm.mean().item():.4f} "
        f"max={pair_norm.max().item():.4f}"
    )

    rho_scale_estimate = float((diff_norm.mean() ** 2).item())
    print(
        f"\nReference scales for ALM: "
        f"||L_dyn||^2 ~ {rho_scale_estimate:.2f}, "
        f"so 0.5*rho*||L_dyn||^2 with rho=1 ~ {0.5 * rho_scale_estimate:.2f}, "
        f"with rho=1000 ~ {500.0 * rho_scale_estimate:.2f}"
    )
    print(
        ">> Compare to lambda_g*goal_loss (~10*MSE, MSE is mean-elementwise so O(0.01-1)). "
        "If rho*||L_dyn||^2 >> lambda_g*goal_loss even at rho=1, the constraint dominates from step 1."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--episode-idx", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    episode = load_oracle_episode(split_dir, args.episode_idx)
    episode = slice_oracle_episode(episode, horizon=args.horizon, frame_skip=FRAME_SKIP)
    print(f"loaded episode {args.episode_idx} (split={args.split}, horizon={args.horizon})")

    world_model, _, _ = make_world_model(model, model_cfg, device)

    experiment_b_scale(world_model, episode, args.horizon, device)

    solver = LatentCollocationSolver(world_model=world_model, config=build_alm_config(diagnostic=True))
    experiment_a_warmstart(world_model, solver, episode, args.horizon, device)


if __name__ == "__main__":
    main()