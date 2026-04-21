from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverConfig:
    inner_steps: int = 25
    learning_rate: float = 3e-2
    clip_grad_norm: Optional[float] = 10.0
    adam_eps: float = 1e-8
    lambda_video: float = 1.0
    lambda_goal: float = 10.0
    lambda_action: float = 0.05
    use_video_init: bool = True
    use_video_loss: bool = True
    fix_states_to_video: bool = False
    use_action_reparameterization: bool = True


@dataclass
class ALMConfig:
    enabled: bool = True
    outer_steps: int = 25
    rho_init: float = 1.0
    rho_growth: float = 1.9
    rho_max: float = 1_000.0


@dataclass
class FeasibilityConfig:
    enabled: bool = True
    lambda_feasibility: float = 1.0
    # Model Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = False
    noise_level: float = 0.1


@dataclass
class RefinementConfig:
    enabled: bool = True
    num_samples: int = 500
    noise_std: float = 0.3


@dataclass
class MPCConfig:
    horizon: int = 25
    execution_stride: int = 1
    warm_start: bool = True


@dataclass
class PlannerConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    alm: ALMConfig = field(default_factory=ALMConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
