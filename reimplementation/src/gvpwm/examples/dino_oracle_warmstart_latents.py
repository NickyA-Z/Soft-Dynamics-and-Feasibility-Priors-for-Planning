"""Plan C: bypass _copy_current_nonvisual_state via warm_start_latents.

Instead of patching the adapter, pass warm_start_latents=encoded_video
into solver.solve(). Solver code path:

    if warm_start_latents is not None and warm_start_latents.shape[0] == horizon + 1:
        latents = warm_start_latents.clone()
    elif self.config.use_video_init:
        latents = self.world_model.initialize_latents_from_video(...)  # the buggy path

So passing warm_start_latents short-circuits the proprio-freezing init.
After init, latents[0] is then forced to current_latent (line 64), which
is correct.

This is the least invasive option (no monkey-patching, no adapter change,
no constraint change). Useful as the cleanest "is the bug really the
cause" test.

Uses paper ALM schedule (rho_init=1, growth=1.9, rho_max=1000) with mean
residual reduction.

Usage:
    python -m src.gvpwm.examples.dino_oracle_warmstart_latents \
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


def main():
    args = parse_args()
    print(
        f"Plan C (warm_start_latents bypass): outer={args.outer_steps} inner={args.inner_steps} "
        f"rho_init={args.rho_init} growth={args.rho_growth} rho_max={args.rho_max} "
        f"reduction={args.reduction}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    print(f"[Plan C] passing warm_start_latents=encoded_video (shape={tuple(encoded_video.shape)}) "
          f"to short-circuit initialize_latents_from_video.")

    solver = LatentCollocationSolver(world_model=world_model, config=build_alm(args))
    result = solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_history,
        goal_latent=goal_latent,
        video_latents=encoded_video,
        warm_start_actions=None,
        warm_start_latents=encoded_video,
    )
    d = result.diagnostics
    print(
        f"\n[FINAL Plan C] residual_norm={result.dynamics_residual_norm:.4f} "
        f"goal={d.get('goal_loss', float('nan')):.6f} "
        f"video={d.get('video_loss', float('nan')):.6f} "
        f"action={d.get('action_loss', float('nan')):.4f} "
        f"final_rho={result.rho:.4e}"
    )


if __name__ == "__main__":
    main()
