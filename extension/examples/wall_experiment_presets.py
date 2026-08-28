from __future__ import annotations

from reimplementation.src.gvpwm.config import PlannerConfig


EXPERIMENT_CHOICES = (
    "alm_baseline",
    "alm_dsm",
    "soft_baseline",
    "soft_dsm_aux",
    "rollout_baseline",
    "rollout_baseline_no_refine",
    "rollout_dsm",
    "free_latent_dsm",
    "free_latent_combined",
)


EXPERIMENT_DESCRIPTIONS = {
    "alm_baseline": "ALM GVP-WM-style baseline without feasibility.",
    "alm_dsm": "ALM GVP-WM-style baseline with DSM auxiliary feasibility.",
    "soft_baseline": "Soft DINO-WM dynamics baseline without feasibility.",
    "soft_dsm_aux": "Soft DINO-WM dynamics with DSM auxiliary feasibility only.",
    "rollout_baseline": "Rollout baseline without feasibility.",
    "rollout_baseline_no_refine": "Rollout baseline without feasibility or random action refinement.",
    "rollout_dsm": "Pure DSM rollout: world model rollout plus DSM feasibility only.",
    "free_latent_dsm": "Free-latent feasibility-only planner with DSM only and no dynamics.",
    "free_latent_combined": (
        "Free-latent planner with DSM, transition, and contrastive feasibility "
        "guidance and no dynamics."
    ),
}


def apply_wall_experiment_preset(config: PlannerConfig, experiment: str) -> PlannerConfig:
    if experiment == "alm_baseline":
        config.alm.dynamics_mode = "alm"
        config.alm.use_dynamics_constraints = True
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = False
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "alm_dsm":
        config.alm.dynamics_mode = "alm"
        config.alm.use_dynamics_constraints = True
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = True
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "soft_baseline":
        config.alm.dynamics_mode = "soft"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 10.0

        config.feasibility.enabled = False
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "soft_dsm_aux":
        config.alm.dynamics_mode = "soft"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 10.0

        config.feasibility.enabled = True
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "rollout_baseline":
        config.alm.dynamics_mode = "rollout"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = False
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "rollout_baseline_no_refine":
        config.alm.dynamics_mode = "rollout"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0

        config.refinement.enabled = False

        config.feasibility.enabled = False
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "rollout_dsm":
        config.alm.dynamics_mode = "rollout"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = True
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "free_latent_dsm":
        config.alm.dynamics_mode = "none"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = True
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_dsm = 1.0
        config.feasibility.lambda_transition = 0.0
        config.feasibility.lambda_contrastive_plan = 0.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    if experiment == "free_latent_combined":
        config.alm.dynamics_mode = "none"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0

        config.feasibility.enabled = True
        config.feasibility.lambda_feasibility = 1.0
        config.feasibility.lambda_dsm = 1.0
        config.feasibility.lambda_transition = 1.0
        config.feasibility.lambda_contrastive_plan = 1.0
        config.feasibility.lambda_action_consistency = 0.0
        return config

    raise ValueError(f"Unknown wall experiment preset: {experiment}")
