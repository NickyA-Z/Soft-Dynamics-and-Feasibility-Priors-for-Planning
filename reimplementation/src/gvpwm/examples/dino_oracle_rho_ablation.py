"""rho_max ablation for the oracle ALM evaluation.

Re-uses the same evaluate_episode flow as dino_oracle_demo, but lets us
sweep rho_max (and optionally rho_init / rho_growth) from the command line
and runs on a small slice of episodes for quick feedback.

The hypothesis: paper's rho_max=1000 is far too large for our latent scale
(see EXPERIMENT B in the diagnose log). Dropping rho_max to ~10 should let
the goal/video terms actually drive the optimization instead of being
crushed by the dynamics penalty.

Usage:
    python -m src.gvpwm.examples.dino_oracle_rho_ablation \
        --split val --horizon 25 --num-episodes 3 --rho-max 10
"""
from __future__ import annotations

import argparse

import torch

from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from . import dino_oracle_demo
from .dino_oracle_demo import (
    DATA_ROOT,
    DEFAULT_HORIZON,
    DEFAULT_SPLIT,
    FRAME_SKIP,
    candidate_episodes,
    evaluate_episode,
    load_model_once,
)


def parse_args():
    parser = argparse.ArgumentParser(description="rho_max ablation for the oracle ALM evaluation.")
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT)
    parser.add_argument("--horizon", type=int, choices=(25, 50, 80), default=DEFAULT_HORIZON)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--rho-growth", type=float, default=1.9)
    parser.add_argument("--rho-max", type=float, default=10.0,
                        help="Penalty cap. Paper uses 1000; we suspect that crushes goal/video terms in our latent scale.")
    parser.add_argument("--diagnostic-inner-interval", type=int, default=5)
    return parser.parse_args()


def make_build_planner(rho_init: float, rho_growth: float, rho_max: float,
                       diagnostic_inner_interval: int):
    """Return a build_planner() that overrides the rho schedule and turns on
    inner-loop diagnostics so the ablation log is easy to compare with the
    debug job."""
    def build_planner(world_model, horizon, **_):
        lambda_action = 0.05 if horizon == 25 else 0.1
        return GVPWMPlanner(
            world_model=world_model,
            config=PlannerConfig(
                alm=ALMConfig(
                    inner_steps=25,
                    outer_steps=25,
                    learning_rate=0.05,
                    rho_init=rho_init,
                    rho_growth=rho_growth,
                    rho_max=rho_max,
                    lambda_video=1.0,
                    lambda_goal=10.0,
                    lambda_action=lambda_action,
                    use_video_init=True,
                    use_video_loss=True,
                    fix_states_to_video=False,
                    use_action_reparameterization=True,
                    diagnostic_inner_interval=diagnostic_inner_interval,
                    diagnostic_outer=True,
                ),
                mpc=MPCConfig(
                    horizon=horizon,
                    execution_stride=1,
                    warm_start=True,
                ),
                refinement=RefinementConfig(
                    enabled=True,
                    num_samples=500,
                    noise_variance=0.3,
                ),
            ),
        )
    return build_planner


def main():
    args = parse_args()
    print(f"rho ablation: rho_init={args.rho_init} rho_growth={args.rho_growth} rho_max={args.rho_max}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    # Monkey-patch the demo's build_planner so evaluate_episode picks ours up.
    dino_oracle_demo.build_planner = make_build_planner(
        rho_init=args.rho_init,
        rho_growth=args.rho_growth,
        rho_max=args.rho_max,
        diagnostic_inner_interval=args.diagnostic_inner_interval,
    )

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=args.horizon, frame_skip=FRAME_SKIP)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]
    print(f"Evaluating split={args.split} horizon={args.horizon}")
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode ids: {eval_episodes}")

    results = []
    for idx in eval_episodes:
        print(f"\n=== Episode {idx} (rho_max={args.rho_max}) ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    start_offset=0,
                    split=args.split,
                    horizon=args.horizon,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback
            print(f"[ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Ablation Summary ===")
    for result in results:
        print(result)
    if results:
        success_rate = sum(r["success"] for r in results) / len(results)
        mean_dist = sum(r["state_dist"] for r in results) / len(results)
        mean_dyn = sum(r["dynamics_residual"] for r in results) / len(results)
        print(
            f"rho_max={args.rho_max}: success_rate={success_rate:.3f} "
            f"mean_state_dist={mean_dist:.2f} mean_dyn_res={mean_dyn:.4f}"
        )


if __name__ == "__main__":
    main()
