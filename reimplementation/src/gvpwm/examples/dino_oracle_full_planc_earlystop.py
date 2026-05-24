"""Plan C full-eval, variant 2: outer_steps=2 (early stop).

Same as dino_oracle_full_planc but caps outer iterations at 2. Single-
episode logs showed outer 1 was the sweet spot (goal=0.0012,
residual=47); stopping there avoids the 11-outer rho saturation that
otherwise pushes the iterate off the task optimum.

Usage:
    python -m src.gvpwm.examples.dino_oracle_full_planc_earlystop \
        --split val --horizon 25 --num-episodes 50
"""
from __future__ import annotations

import torch

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from . import dino_oracle_demo


def _planc_initialize_latents_from_video(
    self,
    current_latent: torch.Tensor,
    video_latents: torch.Tensor,
) -> torch.Tensor:
    latents = video_latents.clone()
    latents[0] = current_latent
    return latents


def _patched_build_planner(
    world_model: DinoWorldModelAdapter,
    horizon: int,
    diagnostic_inner_interval=None,
    diagnostic_outer: bool = False,
) -> GVPWMPlanner:
    lambda_action = 0.05 if horizon == 25 else 0.1
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=2,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1_000.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=lambda_action,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
                use_action_reparameterization=True,
                diagnostic_inner_interval=diagnostic_inner_interval,
                diagnostic_outer=diagnostic_outer,
                residual_reduction="mean",
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


def main():
    DinoWorldModelAdapter.initialize_latents_from_video = _planc_initialize_latents_from_video
    dino_oracle_demo.build_planner = _patched_build_planner
    print("[Plan C early-stop] adapter init patched + outer_steps=2 "
          "(lock in outer-1 sweet spot before rho saturates).")
    dino_oracle_demo.main()


if __name__ == "__main__":
    main()
