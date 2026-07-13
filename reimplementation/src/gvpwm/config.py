from dataclasses import dataclass, field

from typing import Optional


@dataclass
class SolverConfig:
    mode: str = "alm"
    inner_steps: int = 25
    learning_rate: float = 3e-2
    clip_grad_norm: Optional[float] = None
    adam_eps: float = 1e-8
    lambda_video: float = 1.0
    lambda_goal: float = 10.0
    lambda_action: float = 0.05
    lambda_action_prior: float = 0.0
    use_video_init: bool = True
    use_video_loss: bool = True
    fix_states_to_video: bool = False
    use_action_reparameterization: bool = True


@dataclass
class ALMConfig:
    enabled: bool = True
    inner_steps: int = 25
    outer_steps: int = 25
    learning_rate: float = 3e-2
    rho_init: float = 1.0
    rho_growth: float = 1.9
    rho_max: float = 1_000.0
    lambda_video: float = 1.0
    lambda_goal: float = 10.0
    lambda_action: float = 0.05
    lambda_action_prior: float = 0.0
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
    history_action_pad: str = "zeros"  # "zeros" or "repeat_available" for missing DINO action history
    pad_initial_history: bool = True  # if False, start from available context and let DINO history grow


@dataclass
class FeasibilityConfig:
    enabled: bool = False
    diagnostic_only: bool = False
    use_in_refinement: bool = False
    lambda_feasibility: float = 1.0
    hidden_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = False
    noise_level: float = 0.1

    # NEW: Contrastive learning
    lambda_contrastive_train: float = 0.0  # 0 = disabled, >0 = weight in training
    lambda_contrastive_plan: float = 0.0   # 0 = disabled, >0 = weight at planning time
    contrastive_dim: int = 128


@dataclass
class RefinementConfig:
    enabled: bool = True
    num_samples: int = 500
    noise_variance: float = 0.3
    noise_std: Optional[float] = None
    objective: str = "goal"  # "goal" (paper text) or "planner" (video+goal+action)

    def __post_init__(self) -> None:
        if self.noise_std is not None:
            self.noise_variance = self.noise_std**2


@dataclass
class MPCConfig:
    horizon: int = 25
    execution_stride: int = 1
    warm_start: bool = True
    
@dataclass
class LangevinActionConfig:
    enabled: bool = False
    num_steps: int = 20
    step_size: float = 1e-3
    temperature: float = 1e-4
    num_restarts: int = 1
    restart_noise_std: float = 0.05
    grad_clip_norm: float | None = 10.0
    add_noise: bool = True

@dataclass
class PlannerConfig:
    alm: ALMConfig = field(default_factory=ALMConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    solver: SolverConfig | None = None
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)

    langevin_action: LangevinActionConfig = field(default_factory=LangevinActionConfig)

    def __post_init__(self) -> None:
        if self.solver is None:
            return
        for name in (
            "inner_steps",
            "learning_rate",
            "clip_grad_norm",
            "adam_eps",
            "lambda_video",
            "lambda_goal",
            "lambda_action",
            "lambda_action_prior",
            "use_video_init",
            "use_video_loss",
            "fix_states_to_video",
            "use_action_reparameterization",
        ):
            setattr(self.alm, name, getattr(self.solver, name))
