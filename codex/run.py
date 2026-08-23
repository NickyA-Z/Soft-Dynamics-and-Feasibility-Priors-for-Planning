from __future__ import annotations

import argparse
import json

import torch

from codex.config import ObjectiveConfig, OptimizerConfig, PlannerConfig
from codex.evaluation.recovery import action_recovery_metrics
from codex.factory import build_planners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged feasibility-only planner evaluation")
    parser.add_argument("--feasibility-checkpoint", required=True)
    parser.add_argument("--mode", choices=("fixed", "open_loop", "free"), default="fixed")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--episode-idx", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lambda-goal", type=float, default=100.0)
    parser.add_argument("--lambda-dsm", type=float, default=1.0)
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--optimizer", choices=("joint", "alternating"), default="alternating")
    return parser.parse_args()


def main() -> None:
    # These helpers are imported, not duplicated, from the existing oracle demo.
    from local.examples.dino_oracle_demo_copy import DATA_ROOT, FRAME_SKIP, load_model_once
    from local.gvpwm.examples.dino_oracle_diagnose import (
        encode_video, expert_macro_actions, make_world_model,
    )
    from local.gvpwm.examples.dino_oracle_utils import load_oracle_episode, slice_oracle_episode

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)
    adapter, primitive_dim, action_repeat = make_world_model(model, model_cfg, device)
    episode = slice_oracle_episode(
        load_oracle_episode(DATA_ROOT / args.split, args.episode_idx),
        horizon=args.horizon, frame_skip=FRAME_SKIP,
    )
    latents = encode_video(adapter, episode, args.horizon, device).detach()
    expert = expert_macro_actions(
        episode, action_repeat, primitive_dim, args.horizon
    ).to(device)
    # Preserve the dataset-derived bound of every macro-action coordinate.
    # The repeated x/y coordinates can have different valid normalized ranges.
    low = adapter.action_low.detach().clone()
    high = adapter.action_high.detach().clone()
    config = PlannerConfig(
        horizon=args.horizon,
        history_length=adapter.history_length,
        action_low=low,
        action_high=high,
        objective=ObjectiveConfig(
            lambda_goal=args.lambda_goal,
            lambda_dsm=args.lambda_dsm,
            lambda_contrastive=args.lambda_contrastive,
        ),
        optimizer=OptimizerConfig(mode=args.optimizer, joint_steps=args.steps),
    )
    planners = build_planners(adapter, args.feasibility_checkpoint, config, device)
    zero = torch.zeros_like(expert)

    if args.mode == "fixed":
        history = latents[:1].expand(adapter.history_length, *latents.shape[1:])
        fixed = planners["fixed_transition"]
        energies = fixed.energies(
            history, latents[1], {"expert": expert[0], "zero": zero[0]}
        )
        recovered = fixed.recover(history, latents[1], zero[0])
        energies["planner"] = fixed.energies(
            history, latents[1], {"planner": recovered.action}
        )["planner"]
        output = {
            "fixed_transition_energies": energies,
            "action_recovery": action_recovery_metrics(recovered.action, expert[0]),
            "recovered_action": recovered.action.detach().cpu().tolist(),
        }
    elif args.mode == "open_loop":
        result = planners["open_loop"].plan(latents, zero)
        output = {
            "objective": result.objective,
            "breakdown": result.breakdown,
            "action_recovery": action_recovery_metrics(result.actions, expert),
        }
    else:
        result = planners["free_energy"].plan(
            latents[0], latents[-1], latent_history=latents[:1]
        )
        output = {
            "objective": result.objective,
            "breakdown": result.breakdown,
            "action_recovery_diagnostic_only": action_recovery_metrics(result.actions, expert),
            "actions": result.actions.detach().cpu().tolist(),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
