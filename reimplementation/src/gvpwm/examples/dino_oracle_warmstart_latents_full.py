"""Full oracle evaluation with the warm_start_latents init bypass.

This extends the single-episode "Plan C" test to the full MPC evaluation loop.
The only behavioral change relative to dino_oracle_demo is the first solve of
each MPC replanning step:

    if no shifted latent warm start is available yet,
    pass warm_start_latents=current_video

This short-circuits initialize_latents_from_video and avoids the buggy
proprio-freezing path without patching the adapter or the solver.

Usage:
    python -m src.gvpwm.examples.dino_oracle_warmstart_latents_full \
        --split val --horizon 25 --start-index 0 --num-episodes 50
"""
from __future__ import annotations

import argparse
from typing import Any, Callable

import torch

from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..interfaces import MPCResult, MPCStepResult, VideoPlan, VideoPlanSource
from ..planner import GVPWMPlanner
from ..utils import ensure_history_length, shift_action_warm_start, shift_latent_warm_start
from ..video import temporal_resample_sequence
from . import dino_oracle_demo
from .dino_oracle_demo import (
    DATA_ROOT,
    DEFAULT_HORIZON,
    DEFAULT_NUM_EPISODES,
    DEFAULT_SPLIT,
    FRAME_SKIP,
    candidate_episodes,
    evaluate_episode,
    load_model_once,
)


class WarmStartLatentsPlanner(GVPWMPlanner):
    """Planner variant that bypasses initialize_latents_from_video on MPC entry."""

    def run_mpc(
        self,
        observation_history: Any,
        goal_observation: Any,
        step_fn: Callable[[torch.Tensor], Any],
        video_source: VideoPlanSource | None = None,
        video_plan: VideoPlan | None = None,
        past_action_history: torch.Tensor | None = None,
    ) -> MPCResult:
        if video_plan is None:
            if video_source is None:
                raise ValueError("Either video_source or video_plan must be provided.")
            initial_observation = observation_history[-1]
            video_plan = video_source.generate(
                initial_observation=initial_observation,
                goal_observation=goal_observation,
                horizon=self.config.mpc.horizon,
            )

        latent_context = self.world_model.encode_sequence(observation_history).to(self.world_model.device)
        latent_context = ensure_history_length(
            latent_context,
            self.world_model.history_length,
            pad_mode="repeat_first",
        )
        goal_latent = self.world_model.encode_observation(goal_observation).to(self.world_model.device)
        encoded_video = self._encode_video(video_plan, self.config.mpc.horizon)

        if past_action_history is None:
            past_action_history = torch.zeros(
                max(self.world_model.history_length - 1, 0),
                self.world_model.action_dim,
                device=self.world_model.device,
            )
        else:
            past_action_history = past_action_history.to(self.world_model.device)

        warm_start_latents = None
        warm_start_actions = None
        executed_actions = []
        executed_latents = [latent_context[-1]]
        steps: list[MPCStepResult] = []

        time_index = 0
        while time_index < self.config.mpc.horizon:
            remaining = self.config.mpc.horizon - time_index
            current_video = temporal_resample_sequence(
                encoded_video[time_index:],
                remaining + 1,
            )
            solver_warm_start_latents = warm_start_latents
            if solver_warm_start_latents is None:
                solver_warm_start_latents = current_video

            result = self.solver.solve(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                video_latents=current_video,
                warm_start_latents=solver_warm_start_latents,
                warm_start_actions=warm_start_actions,
            )
            planned_actions = self._refine_actions(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                actions=result.actions,
            )
            n_exec = min(self.config.mpc.execution_stride, remaining)
            executed_this_round = planned_actions[:n_exec]

            for offset in range(n_exec):
                action = executed_this_round[offset].detach()
                next_observation = step_fn(action)
                next_latent = self.world_model.encode_observation(next_observation).to(self.world_model.device)
                executed_actions.append(action)
                executed_latents.append(next_latent)
                latent_context = torch.cat([latent_context, next_latent.unsqueeze(0)], dim=0)[
                    -self.world_model.history_length :
                ]
                if self.world_model.history_length > 1:
                    past_action_history = torch.cat(
                        [past_action_history, action.unsqueeze(0)],
                        dim=0,
                    )[-(self.world_model.history_length - 1) :]
                time_index += 1

            steps.append(
                MPCStepResult(
                    step_index=time_index,
                    planned_actions=planned_actions.detach(),
                    executed_actions=executed_this_round.detach(),
                    collocation_objective=result.objective,
                    dynamics_residual_norm=result.dynamics_residual_norm,
                )
            )

            if self.config.mpc.warm_start:
                warm_start_latents = shift_latent_warm_start(result.latents, n_exec)
                warm_start_latents[0] = latent_context[-1]
                warm_start_actions = shift_action_warm_start(planned_actions, n_exec)
            else:
                warm_start_latents = None
                warm_start_actions = None

        return MPCResult(
            video_latents=encoded_video.detach(),
            executed_actions=torch.stack(executed_actions, dim=0),
            executed_latents=torch.stack(executed_latents, dim=0),
            steps=steps,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full oracle evaluation with the warm_start_latents MPC init bypass"
    )
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT)
    parser.add_argument("--horizon", type=int, choices=(25, 50, 80), default=DEFAULT_HORIZON)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--outer-steps", type=int, default=25)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--rho-growth", type=float, default=1.9)
    parser.add_argument("--rho-max", type=float, default=1000.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--debug-inner-every", type=int, default=None)
    parser.add_argument("--debug-outer", action="store_true")
    return parser.parse_args()


def make_build_planner(args):
    def build_planner(world_model, horizon, diagnostic_inner_interval=None, diagnostic_outer=False):
        lambda_action = 0.05 if horizon == 25 else 0.1
        return WarmStartLatentsPlanner(
            world_model=world_model,
            config=PlannerConfig(
                alm=ALMConfig(
                    inner_steps=args.inner_steps,
                    outer_steps=args.outer_steps,
                    learning_rate=args.lr,
                    rho_init=args.rho_init,
                    rho_growth=args.rho_growth,
                    rho_max=args.rho_max,
                    lambda_video=1.0,
                    lambda_goal=10.0,
                    lambda_action=lambda_action,
                    use_video_init=True,
                    use_video_loss=True,
                    fix_states_to_video=False,
                    use_action_reparameterization=True,
                    diagnostic_inner_interval=diagnostic_inner_interval,
                    diagnostic_outer=diagnostic_outer,
                    residual_reduction=args.reduction,
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
    print(
        "Warmstart-latents full eval: "
        f"split={args.split} horizon={args.horizon} start={args.start_index} n={args.num_episodes} "
        f"outer={args.outer_steps} inner={args.inner_steps} "
        f"rho_init={args.rho_init} growth={args.rho_growth} rho_max={args.rho_max} "
        f"lr={args.lr} reduction={args.reduction}"
    )
    print(
        "Bypass mode: if no shifted latent warm start is available, "
        "use the encoded video slice as warm_start_latents."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    dino_oracle_demo.build_planner = make_build_planner(args)

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=args.horizon, frame_skip=FRAME_SKIP)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]
    min_required_length = args.horizon * FRAME_SKIP + 1

    print(f"Evaluating split={args.split} horizon={args.horizon}")
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode ids: {eval_episodes}")

    if not eligible:
        raise ValueError(
            f"No eligible episodes found for split={args.split}, horizon={args.horizon}. "
            f"Current filter requires seq_length >= {min_required_length} "
            f"(horizon * frame_skip + 1, frame_skip={FRAME_SKIP})."
        )
    if not eval_episodes:
        raise ValueError(
            f"No episodes selected after applying start-index={args.start_index} "
            f"and num-episodes={args.num_episodes}. Eligible count={len(eligible)}."
        )

    results = []
    for idx in eval_episodes:
        print(f"\n=== Episode {idx} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    split=args.split,
                    horizon=args.horizon,
                    diagnostic_inner_interval=args.debug_inner_every,
                    diagnostic_outer=args.debug_outer,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback

            print(f"[ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Evaluation Summary ===")
    for result in results:
        print(result)
    if results:
        success_rate = sum(result["success"] for result in results) / len(results)
        mean_dist = sum(result["state_dist"] for result in results) / len(results)
        mean_dyn = sum(result["dynamics_residual"] for result in results) / len(results)
        print(
            f"Success rate: {success_rate:.3f}  "
            f"mean_state_dist: {mean_dist:.2f}  "
            f"mean_dyn_res: {mean_dyn:.4f}"
        )


if __name__ == "__main__":
    main()
