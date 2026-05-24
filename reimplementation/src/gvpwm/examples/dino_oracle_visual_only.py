"""Plan B: visual-only dynamics residual.

Hypothesis: proprio mismatch dominates the residual; the visual constraint
is what we actually care about for goal/video alignment. So drop proprio
from the dynamics residual entirely by overriding predict_next_latent to
copy the input proprio to the output. Then residual_proprio = 0 always,
and only visual residual contributes to the augmented Lagrangian.

This is more invasive than Plan A (changes the constraint set), but if
ALM still fails, it tells us the failure isn't proprio per se but visual
WM mismatch.

Uses paper ALM schedule (rho_init=1, growth=1.9, rho_max=1000) with
mean residual reduction.

Usage:
    python -m src.gvpwm.examples.dino_oracle_visual_only \
        --split val --horizon 25 --episode-idx 2
"""
from __future__ import annotations

import argparse

import torch

from ..config import ALMConfig
from ..solver import LatentCollocationSolver
from .dino_oracle_demo import (
    DATA_ROOT,
    DEFAULT_HORIZON,
    DEFAULT_SPLIT,
    FRAME_SKIP,
    load_model_once,
)
from .dino_oracle_diagnose import (
    encode_video,
    latent_context_from_start,
    make_world_model,
)
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--episode-idx", type=int, default=2)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--outer-steps", type=int, default=25)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--rho-growth", type=float, default=1.9)
    parser.add_argument("--rho-max", type=float, default=1000.0)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="mean")
    return parser.parse_args()


def build_alm(args) -> ALMConfig:
    return ALMConfig(
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        learning_rate=0.05,
        rho_init=args.rho_init,
        rho_growth=args.rho_growth,
        rho_max=args.rho_max,
        lambda_video=1.0,
        lambda_goal=10.0,
        lambda_action=0.05,
        use_video_init=True,
        use_video_loss=True,
        fix_states_to_video=False,
        use_action_reparameterization=True,
        diagnostic_inner_interval=5,
        diagnostic_outer=True,
        residual_reduction=args.reduction,
    )


def patch_predict_visual_only(world_model):
    """Override predict_next_latent to copy the input proprio onto the
    output. With this, residual_proprio = candidate_proprio - candidate_proprio
    = 0, so only visual residual feeds the augmented Lagrangian."""
    visual_dim = int(world_model.world_model.encoder.emb_dim)
    orig_predict = world_model.predict_next_latent

    def predict_visual_only(latent_history, action_history):
        pred = orig_predict(latent_history, action_history)
        out = pred.clone()
        out[..., visual_dim:] = latent_history[-1, ..., visual_dim:]
        return out

    world_model.predict_next_latent = predict_visual_only
    return visual_dim


def main():
    args = parse_args()
    print(
        f"Plan B (visual-only residual): outer={args.outer_steps} inner={args.inner_steps} "
        f"rho_init={args.rho_init} growth={args.rho_growth} rho_max={args.rho_max} "
        f"reduction={args.reduction}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    episode = load_oracle_episode(split_dir, args.episode_idx)
    episode = slice_oracle_episode(episode, horizon=args.horizon, frame_skip=FRAME_SKIP)

    world_model, _, _ = make_world_model(model, model_cfg, device)
    visual_dim = patch_predict_visual_only(world_model)
    print(f"[Plan B] monkey-patched predict_next_latent; visual_dim={visual_dim} "
          f"(proprio slice [{visual_dim}:] is forced to copy input).")

    latent_context = latent_context_from_start(world_model, episode, device)
    past_action_history = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=device,
    )
    encoded_video = encode_video(world_model, episode, args.horizon, device)
    goal_latent = world_model.encode_observation(episode["goal_obs"]).to(device)

    solver = LatentCollocationSolver(world_model=world_model, config=build_alm(args))
    result = solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_history,
        goal_latent=goal_latent,
        video_latents=encoded_video,
        warm_start_actions=None,
        warm_start_latents=None,
    )
    d = result.diagnostics
    print(
        f"\n[FINAL Plan B] residual_norm={result.dynamics_residual_norm:.4f} "
        f"goal={d.get('goal_loss', float('nan')):.6f} "
        f"video={d.get('video_loss', float('nan')):.6f} "
        f"action={d.get('action_loss', float('nan')):.4f} "
        f"final_rho={result.rho:.4e}"
    )


if __name__ == "__main__":
    main()
