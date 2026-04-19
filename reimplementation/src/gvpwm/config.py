from dataclasses import dataclass, field

from typing import Optional

@dataclass
class ALMConfig:
    inner_steps: int = 25
    outer_steps: int = 25
    learning_rate: float = 3e-2
    rho_init: float = 1.0
    rho_growth: float = 1.9
    rho_max: float = 1_000.0
    lambda_video: float = 1.0
    lambda_goal: float = 10.0
    lambda_action: float = 0.05
    clip_grad_norm: Optional[float] = None
    use_video_init: bool = True
    use_video_loss: bool = True
    fix_states_to_video: bool = False
    use_action_reparameterization: bool = True
    adam_eps: float = 1e-8
    diagnostic_inner_interval: Optional[int] = None
    diagnostic_outer: bool = False
    residual_reduction: str = "sum"  # "sum" (paper) or "mean" (per-element-mean of ||L_dyn||^2)
    diagnostic_grad_norms: bool = False  # if True, print grad norms of latent/action parameters per inner step


@dataclass
class RefinementConfig:
    enabled: bool = True
    num_samples: int = 500
    noise_variance: float = 0.3


@dataclass
class MPCConfig:
    horizon: int = 25
    execution_stride: int = 1
    warm_start: bool = True


@dataclass
class PlannerConfig:
    alm: ALMConfig = field(default_factory=ALMConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
