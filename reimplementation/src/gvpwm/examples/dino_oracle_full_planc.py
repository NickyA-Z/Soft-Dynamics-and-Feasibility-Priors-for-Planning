"""Full-split oracle evaluation under Plan C.

Plan C in the single-episode script bypasses _copy_current_nonvisual_state
by passing warm_start_latents=encoded_video to solver.solve. In the full
MPC loop, the first solve call has warm_start_latents=None (subsequent
calls warm-start from the previous solve), so the buggy init still fires
on the first step of every episode.

Easiest way to apply the same fix everywhere: monkey-patch
DinoWorldModelAdapter.initialize_latents_from_video at module level so it
returns video_latents with only frame 0 overwritten by current_latent
(no proprio freezing across t=1..H). This is functionally identical to
Plan A/C but plugs straight into evaluate_episode without touching the
planner or solver.

Also overrides build_planner to use the paper ALM schedule with mean
residual reduction (rho_init=1, growth=1.9, rho_max=1000, mean).

Usage:
    python -m src.gvpwm.examples.dino_oracle_full_planc \
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
                outer_steps=25,
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
    print("[Plan C full-eval] adapter init patched (no proprio freezing); "
          "ALM uses paper rho schedule + mean residual reduction.")
    dino_oracle_demo.main()


if __name__ == "__main__":
    main()
