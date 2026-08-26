from __future__ import annotations

import argparse

import torch

from codex.training.config import AdversarialConfig, TrainingConfig
from codex.training.trainer import PlannerAlignedTrainer
from extension.feasibility2.dataset import load_tensor_dataset
from extension.feasibility2.train import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planner-aligned feasibility training")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dsm-warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-dsm", type=float, default=0.1)
    parser.add_argument("--lambda-global", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=1.0)
    parser.add_argument("--lambda-adversarial", type=float, default=1.0)
    parser.add_argument("--adversarial-starts", type=int, default=3)
    parser.add_argument("--adversarial-steps", type=int, default=12)
    parser.add_argument("--adversarial-every", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_dataset, train_metadata = load_tensor_dataset(args.train_dataset)
    val_dataset, val_metadata = load_tensor_dataset(args.val_dataset)
    validate_dataset(train_dataset)
    validate_dataset(val_dataset)
    config = TrainingConfig(
        epochs=args.epochs,
        dsm_warmup_epochs=args.dsm_warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_dsm=args.lambda_dsm,
        lambda_global=args.lambda_global,
        lambda_local=args.lambda_local,
        lambda_adversarial=args.lambda_adversarial,
        num_workers=args.num_workers,
        adversarial=AdversarialConfig(
            starts=args.adversarial_starts,
            steps=args.adversarial_steps,
            every_batches=args.adversarial_every,
        ),
    )
    print("Device:", device)
    print("Training samples:", len(train_dataset), train_metadata.get("episode_start"),
          train_metadata.get("episode_end"))
    print("Validation samples:", len(val_dataset), val_metadata.get("episode_start"),
          val_metadata.get("episode_end"))
    print("Training configuration:", config)
    trainer = PlannerAlignedTrainer(
        train_dataset, val_dataset, train_metadata, val_metadata,
        config, device, args.checkpoint_out, args.init_checkpoint,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
