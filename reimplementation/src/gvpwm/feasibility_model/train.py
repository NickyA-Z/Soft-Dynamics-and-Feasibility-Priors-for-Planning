"""
Training script for FeasibilityModel and ResidualFeasibilityModel.

Both models are trained offline on expert data before planning.
    FeasibilityModel trains on (h_t, a_t, z_{t+1}) tuples.
    ResidualFeasibilityModel trains on (h_t, a_t, delta_t)
        where delta_t = z_{t+1} - f_psi(h_t, a_t).

Note:
    You must first convert a dataset (such as Push-T) to the required format
    before running the script.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils.data
from gvpwm.feasibility_model import FeasibilityModel, ResidualFeasibilityModel
from torch.utils.data import DataLoader, random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    # Model architecture
    latent_dim: int = 384  # Must match world model.
    action_dim: int = 2  # Must match world model.
    history_length: int = 3  # Must match world model.
    hidden_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = False

    # Training
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Noise level sampling — sigma is sampled uniformly from [noise_level_min, noise_level_max]
    noise_level_min: float = 0.01
    noise_level_max: float = 1.0

    # Data split
    val_fraction: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 100  # save every N epochs
    keep_last_n: int = 3  # keep only the last N checkpoints, plus best
    resume: str | None = None  # path to checkpoint to resume from
    run_name: str | None = None  # optional name for the run, used in checkpoint directory

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "feasibility"  # "feasibility" or "residual"
    data_path: str = "data/expert_transitions.pt"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class ExpertTransitionDataset(torch.utils.data.Dataset):
    """
    Dataset of expert transitions (h_t, a_t, z_{t+1}) for FeasibilityModel,
    or (h_t, a_t, z_{t+1}, f_psi(h_t, a_t)) for ResidualFeasibilityModel.
    """

    def __init__(
        self,
        history: torch.Tensor,  # (N, H, Dz)
        action: torch.Tensor,  # (N, Da)
        z_next: torch.Tensor,  # (N, Dz)
        predicted_next: torch.Tensor | None = None,  # (N, Dz) — f_psi predictions
    ) -> None:
        assert (
            history.shape[0] == action.shape[0] == z_next.shape[0]
        ), "history, action, and z_next must have the same number of samples."
        if predicted_next is not None:
            assert predicted_next.shape == z_next.shape, "predicted_next must have the same shape as z_next."
        self.history = history
        self.action = action
        self.z_next = z_next
        self.predicted_next = predicted_next

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        if self.predicted_next is not None:
            return self.history[idx], self.action[idx], self.z_next[idx], self.predicted_next[idx]
        return self.history[idx], self.action[idx], self.z_next[idx]


def load_dataset(
    data_path: str,
    model_type: str,
    val_fraction: float,
    seed: int,
) -> tuple[ExpertTransitionDataset, ExpertTransitionDataset]:
    """Loads expert transition data and splits into train/val."""
    logger.info(f"Loading data from {data_path}")
    data = torch.load(data_path, weights_only=True)

    required_keys = ["history", "action", "z_next"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Data file missing required key: '{key}'.")

    if model_type == "residual" and "predicted_next" not in data:
        raise KeyError("Residual model requires 'predicted_next' in data file.")

    history = data["history"].float()
    action = data["action"].float()
    z_next = data["z_next"].float()
    predicted_next = data.get("predicted_next", None)
    if predicted_next is not None:
        predicted_next = predicted_next.float()

    n_total = history.shape[0]
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx = list(train_indices)
    val_idx = list(val_indices)

    def make_split(indices: list[int]) -> ExpertTransitionDataset:
        return ExpertTransitionDataset(
            history=history[indices],
            action=action[indices],
            z_next=z_next[indices],
            predicted_next=predicted_next[indices] if predicted_next is not None else None,
        )

    train_dataset = make_split(train_idx)
    val_dataset = make_split(val_idx)
    logger.info(f"Dataset: {n_train} train, {n_val} val samples.")
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: FeasibilityModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig,
    is_best: bool,
    keep_last_n: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    # Save periodic checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(state, best_path)
        logger.info(f"New best model saved: val_loss={val_loss:.6f}")

    # Remove old checkpoints, keeping only the last N (not including best)
    existing = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    for old_checkpoint in existing[:-keep_last_n]:
        old_checkpoint.unlink()
        logger.info(f"Removed old checkpoint: {old_checkpoint}")


def load_checkpoint(
    checkpoint_path: str,
    model: FeasibilityModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> int:
    """Loads checkpoint and returns the epoch to resume from."""
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    start_epoch = state["epoch"] + 1
    logger.info(f"Resumed from epoch {state['epoch']}, val_loss={state['val_loss']:.6f}")
    return start_epoch


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def sample_noise_levels(
    batch_size: int,
    noise_level_min: float,
    noise_level_max: float,
    device: str,
) -> torch.Tensor:
    """Samples noise levels uniformly from [noise_level_min, noise_level_max]."""
    return torch.empty(batch_size, device=device).uniform_(noise_level_min, noise_level_max)


def train_epoch(
    model: FeasibilityModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = [t.to(config.device) for t in batch]
        history, action, z_next = batch[0], batch[1], batch[2]
        predicted_next = batch[3] if len(batch) == 4 else None

        noise_level = sample_noise_levels(
            history.shape[0],
            config.noise_level_min,
            config.noise_level_max,
            config.device,
        )

        if isinstance(model, ResidualFeasibilityModel) and predicted_next is not None:
            loss = model.residual_dsm_loss(
                history=history,
                action=action,
                predicted_next_latent=predicted_next,
                true_next_latent=z_next,
                noise_level=noise_level,
            )
        else:
            loss = model.dsm_loss(
                history=history,
                action=action,
                z_next=z_next,
                noise_level=noise_level,
            )

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(
    model: FeasibilityModel,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        batch = [t.to(config.device) for t in batch]
        history, action, z_next = batch[0], batch[1], batch[2]
        predicted_next = batch[3] if len(batch) == 4 else None

        noise_level = sample_noise_levels(
            history.shape[0],
            config.noise_level_min,
            config.noise_level_max,
            config.device,
        )

        if isinstance(model, ResidualFeasibilityModel) and predicted_next is not None:
            loss = model.residual_dsm_loss(
                history=history,
                action=action,
                predicted_next_latent=predicted_next,
                true_next_latent=z_next,
                noise_level=noise_level,
            )
        else:
            loss = model.dsm_loss(
                history=history,
                action=action,
                z_next=z_next,
                noise_level=noise_level,
            )

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(config: TrainingConfig) -> Path:
    """Resolves the checkpoint directory from config.

    If run_name is provided, uses: checkpoint_dir/run_name
    Otherwise, auto-generates from model type and architecture:
        checkpoint_dir/{model_type}_h{history_length}_dz{latent_dim}_da{action_dim}
    """
    if config.run_name is not None:
        return Path(config.checkpoint_dir) / config.run_name
    auto_name = f"{config.model_type}_h{config.history_length}_dz{config.latent_dim}_da{config.action_dim}"
    return Path(config.checkpoint_dir) / auto_name


def build_model(config: TrainingConfig) -> FeasibilityModel:
    cls = ResidualFeasibilityModel if config.model_type == "residual" else FeasibilityModel
    return cls(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        history_length=config.history_length,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        use_layer_norm=config.use_layer_norm,
    ).to(config.device)


def train(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)

    # Data
    train_dataset, val_dataset = load_dataset(
        data_path=config.data_path,
        model_type=config.model_type,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model type: {config.model_type} | Parameters: {n_params:,} | Device: {config.device}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Resume
    start_epoch = 1
    if config.resume is not None:
        start_epoch = load_checkpoint(config.resume, model, optimizer, scheduler, config.device)

    checkpoint_dir = resolve_checkpoint_dir(config)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config)
        val_loss = eval_epoch(model, val_loader, config)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        logger.info(
            f"Epoch {epoch:03d}/{config.num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}" + (" *best*" if is_best else "")
        )

        if epoch % config.checkpoint_every == 0 or is_best:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
                is_best=is_best,
                keep_last_n=config.keep_last_n,
            )

    logger.info(f"Training complete. Best val_loss={best_val_loss:.6f}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train FeasibilityModel or ResidualFeasibilityModel.")
    parser.add_argument("--model_type", type=str, default="feasibility", choices=["feasibility", "residual"])
    parser.add_argument("--data_path", type=str, default="data/expert_transitions.pt")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name for the run. If not set, auto-generated from model type and architecture.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--noise_level_min", type=float, default=0.01)
    parser.add_argument("--noise_level_max", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--keep_last_n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
