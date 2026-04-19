"""Grad-norm diagnostic.

Logs ||grad(action_param)|| and ||grad(latent_param)|| (plus per-element
mean abs) at every diagnostic inner step. Lets us check whether
latent_parameter actually moves under Adam, or whether its gradient is so
diluted across 196*394 elements that it's effectively pinned.

Hypothesis we want to test: residual barely budges (6972 -> 5670 over many
outer iters). Is that because:
  (a) latent_grad_norm is small relative to action_grad_norm, so Adam
      preferentially updates actions and latents stay near video init, or
  (b) both grads are large but the gradient direction is fighting the
      task term, or
  (c) latent_grad_norm is huge but spread thin across N=77k elements, so
      per-element step is tiny.

Run on a single episode for a few outer iters at one rho regime.

Usage:
    python -m src.gvpwm.examples.dino_oracle_grad_diag \
        --split val --horizon 25 --episode-idx 2 \
        --rho-init 1.0 --reduction mean
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
    parser.add_argument("--outer-steps", type=int, default=4)
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
        diagnostic_grad_norms=True,
    )


def main():
    args = parse_args()
    print(
        f"grad diag: outer={args.outer_steps} inner={args.inner_steps} "
        f"rho_init={args.rho_init} rho_growth={args.rho_growth} "
        f"reduction={args.reduction}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    episode = load_oracle_episode(split_dir, args.episode_idx)
    episode = slice_oracle_episode(episode, horizon=args.horizon, frame_skip=FRAME_SKIP)

    world_model, _, _ = make_world_model(model, model_cfg, device)

    latent_context = latent_context_from_start(world_model, episode, device)
    past_action_history = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=device,
    )
    encoded_video = encode_video(world_model, episode, args.horizon, device)
    goal_latent = world_model.encode_observation(episode["goal_obs"]).to(device)

    # Reference scales for interpreting grad norms.
    n_latent_per_step = encoded_video[0].numel()
    n_action_per_step = world_model.action_dim
    print(
        f"[ref] latent params: horizon={args.horizon}, per-step numel={n_latent_per_step}, "
        f"total={args.horizon * n_latent_per_step}"
    )
    print(
        f"[ref] action params: horizon={args.horizon}, per-step numel={n_action_per_step}, "
        f"total={args.horizon * n_action_per_step}"
    )

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
        f"\n[FINAL] residual_norm={result.dynamics_residual_norm:.4f} "
        f"goal={d.get('goal_loss', float('nan')):.6f} "
        f"video={d.get('video_loss', float('nan')):.6f} "
        f"action={d.get('action_loss', float('nan')):.4f} "
        f"final_rho={result.rho:.4e}"
    )


if __name__ == "__main__":
    main()
