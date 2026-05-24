"""rho_init scan.

Scans rho_init over decades on a SINGLE episode (no MPC env step), so we can
finally see whether ALM can drive goal_loss down once the penalty term is
small enough to not crush the task term.

By the wm_rollout_sanity numbers, ||L_dyn|| ~ 250/step. With sum-reduction,
0.5 * rho * 25 * 250^2 ~ 8e5 * rho dominates a task loss of O(10) unless
rho <~ 1e-5.

Usage:
    python -m src.gvpwm.examples.dino_oracle_rho_scan \
        --split val --horizon 25 --episode-idx 2 \
        --rho-init-list 1e-6 1e-5 1e-4 1e-3
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
    parser.add_argument("--rho-init-list", type=float, nargs="+",
                        default=[1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--rho-growth", type=float, default=1.9)
    parser.add_argument("--rho-max-mult", type=float, default=1000.0,
                        help="rho_max = rho_init * rho_max_mult; keeps each scan symmetric.")
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--outer-steps", type=int, default=15,
                        help="Smaller than 25 to fit all rho values in one job.")
    return parser.parse_args()


def build_alm(rho_init: float, rho_growth: float, rho_max: float,
              inner: int, outer: int) -> ALMConfig:
    return ALMConfig(
        inner_steps=inner,
        outer_steps=outer,
        learning_rate=0.05,
        rho_init=rho_init,
        rho_growth=rho_growth,
        rho_max=rho_max,
        lambda_video=1.0,
        lambda_goal=10.0,
        lambda_action=0.05,
        use_video_init=True,
        use_video_loss=True,
        fix_states_to_video=False,
        use_action_reparameterization=True,
        diagnostic_inner_interval=5,
        diagnostic_outer=True,
        residual_reduction="sum",
    )


def main():
    args = parse_args()
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

    summary = []
    for rho_init in args.rho_init_list:
        rho_max = rho_init * args.rho_max_mult
        print(f"\n================ rho_init={rho_init:.1e}  rho_max={rho_max:.1e}  growth={args.rho_growth} ================")
        cfg = build_alm(rho_init, args.rho_growth, rho_max, args.inner_steps, args.outer_steps)
        solver = LatentCollocationSolver(world_model=world_model, config=cfg)
        result = solver.solve(
            latent_context=latent_context,
            past_action_context=past_action_history,
            goal_latent=goal_latent,
            video_latents=encoded_video,
            warm_start_actions=None,
            warm_start_latents=None,
        )
        d = result.diagnostics
        line = (
            f"[FINAL rho_init={rho_init:.1e}] "
            f"residual_norm={result.dynamics_residual_norm:.4f} "
            f"goal={d.get('goal_loss', float('nan')):.6f} "
            f"video={d.get('video_loss', float('nan')):.6f} "
            f"action={d.get('action_loss', float('nan')):.4f} "
            f"final_rho={result.rho:.4e}"
        )
        print(line)
        summary.append(line)

    print("\n================ SCAN SUMMARY ================")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
