from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdversarialConfig:
    starts: int = 3
    steps: int = 12
    learning_rate: float = 0.08
    every_batches: int = 4
    minimum_rms_distance: float = 0.25
    distance_penalty: float = 10.0

    def __post_init__(self) -> None:
        if min(self.starts, self.steps, self.every_batches) < 1:
            raise ValueError("Adversarial starts, steps, and frequency must be positive")
        if self.learning_rate <= 0 or self.minimum_rms_distance < 0:
            raise ValueError("Invalid adversarial learning-rate or distance")


@dataclass
class TrainingConfig:
    epochs: int = 30
    dsm_warmup_epochs: int = 5
    batch_size: int = 32
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    lambda_dsm: float = 0.1
    lambda_global: float = 1.0
    lambda_local: float = 1.0
    lambda_adversarial: float = 1.0
    lambda_calibration: float = 0.1
    ranking_margin: float = 0.2
    local_ranking_margin: float = 0.02
    temperature: float = 0.1
    contrastive_noise_level: float = 0.2
    local_noise_scales: tuple[float, ...] = (0.05, 0.15, 0.4, 0.8)
    num_action_shuffles: int = 2
    num_workers: int = 4
    validation_batches: int = 8
    validation_starts: int = 3
    validation_steps: int = 30
    seed: int = 0
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)

    def __post_init__(self) -> None:
        if self.epochs < 1 or not 0 <= self.dsm_warmup_epochs < self.epochs:
            raise ValueError("Warmup must be non-negative and shorter than training")
        if self.batch_size < 2 or self.validation_batches < 1:
            raise ValueError("Batch size and validation batches must be valid")
        if self.num_action_shuffles < 1:
            raise ValueError("At least one shuffled-action negative is required")
        if self.model_dim % self.num_heads != 0 or self.num_layers < 1:
            raise ValueError("Invalid transformer dimensions")
        for name, value in vars(self).items():
            if name.startswith("lambda_") and value < 0:
                raise ValueError(f"{name} must be non-negative")
