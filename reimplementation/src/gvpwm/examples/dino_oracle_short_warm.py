"""Single-episode ALM solve with two combined fixes:

  (1) Early-stop schedule: outer_steps=2, rho_growth=1.0, rho_max=1.0,
      mean residual reduction. The mean-residual run showed goal_loss
      reaches ~0.008 by outer 0 and degrades only because the rho ramp
      keeps pushing the penalty up. So we just don't ramp.

  (2) Warm-start actions = expert. Sanity T1 showed expert actions give
      per-step ||L_dyn|| ~ 170 (a feasible point), but the planner starts
      at midpoint actions where residual ~ 6900 -- ALM never finds the
      feasible basin. Starting from expert lets dual ascent converge from
      a near-feasible point.

This is the cleanest test of "is ALM viable here at all".

Usage:
    python -m src.gvpwm.examples.dino_oracle_short_warm \
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
    expert_macro_actions,
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
    parser.add_argument("--outer-steps", type=int, default=2,
                        help="Keep small to avoid multiplier accumulation.")
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--rho-growth", type=float, default=1.0,
                        help="No growth: keep rho fixed at rho_init.")
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable expert warm-start (for ablation).")
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
        f"short+warm: outer={args.outer_steps} inner={args.inner_steps} "
        f"rho_init={args.rho_init} rho_growth={args.rho_growth} rho_max={args.rho_max} "
        f"reduction={args.reduction} warm_start={'no' if args.no_warm_start else 'expert'}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    episode = load_oracle_episode(split_dir, args.episode_idx)
    episode = slice_oracle_episode(episode, horizon=args.horizon, frame_skip=FRAME_SKIP)

    world_model, primitive_action_dim, action_repeat = make_world_model(model, model_cfg, device)

    latent_context = latent_context_from_start(world_model, episode, device)
    past_action_history = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=device,
    )
    encoded_video = encode_video(world_model, episode, args.horizon, device)
    goal_latent = world_model.encode_observation(episode["goal_obs"]).to(device)

    if args.no_warm_start:
        warm_start_actions = None
    else:
        warm_start_actions = expert_macro_actions(
            episode, action_repeat, primitive_action_dim, args.horizon
        ).to(device)
        # Diagnostic: what's the residual at the expert warm-start point?
        residuals = LatentCollocationSolver(
            world_model=world_model, config=build_alm(args)
        )._dynamics_residuals(
            latent_context=latent_context,
            past_action_context=past_action_history,
            candidate_latents=encoded_video,
            candidate_actions=warm_start_actions,
        )
        per_step = residuals.reshape(residuals.shape[0], -1).norm(dim=1)
        print(
            f"[warm-start check] residual per step: "
            f"mean={per_step.mean().item():.2f} "
            f"min={per_step.min().item():.2f} "
            f"max={per_step.max().item():.2f}"
        )

    solver = LatentCollocationSolver(world_model=world_model, config=build_alm(args))
    result = solver.solve(
        latent_context=latent_context,
        past_action_context=past_action_history,
        goal_latent=goal_latent,
        video_latents=encoded_video,
        warm_start_actions=warm_start_actions,
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
