from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from nicky_dl.feasibility2.dataset import FeasibilityDataset, load_tensor_dataset
from nicky_dl.feasibility2.model import FeasibilityModel, TransformerFeasibilityModel
from nicky_dl.feasibility2.scheduler import LogUniformSigmaScheduler


def _split_batch(batch):
    if len(batch) == 4:
        history, action, z_next, _past_action_histories = batch
    elif len(batch) == 3:
        history, action, z_next = batch
    else:
        raise ValueError(f"Unexpected batch size: {len(batch)}")
    return history, action, z_next


def validate_dataset_shapes(dataset: FeasibilityDataset, architecture: str) -> None:
    if dataset.actions.ndim != 2:
        raise ValueError(f"Expected actions [N, action_dim], got {tuple(dataset.actions.shape)}")
    if not (dataset.histories.shape[0] == dataset.actions.shape[0] == dataset.next_latents.shape[0]):
        raise ValueError("histories/actions/next_latents sample count mismatch")
    if dataset.actions.shape[-1] != 10:
        raise ValueError(f"Expected action_dim=10, got {dataset.actions.shape[-1]}")
    if dataset.histories.shape[1] <= 0:
        raise ValueError(f"Expected positive history_length, got {dataset.histories.shape[1]}")

    if architecture == "mlp":
        if dataset.histories.ndim != 3 or dataset.next_latents.ndim != 2:
            raise ValueError("MLP expects histories [N,H,D] and next_latents [N,D]")
    elif architecture == "transformer":
        if dataset.histories.ndim < 4 or dataset.next_latents.ndim < 3:
            raise ValueError("Transformer expects histories [N,H,*latent_shape] and next_latents [N,*latent_shape]")
        latent_shape = tuple(dataset.next_latents.shape[1:])
        history_latent_shape = tuple(dataset.histories.shape[2:])
        if history_latent_shape != latent_shape:
            raise ValueError(
                f"History latent shape {history_latent_shape} does not match next_latents {latent_shape}"
            )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    for name, tensor in [
        ("histories", dataset.histories),
        ("actions", dataset.actions),
        ("next_latents", dataset.next_latents),
    ]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")


def train_feasibility_model(
    dataset: FeasibilityDataset,
    val_dataset: FeasibilityDataset | None,
    dataset_metadata: dict | None,
    val_dataset_metadata: dict | None,
    architecture: str,
    checkpoint_out: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    noise_level: float,
    sigma_min: float,
    sigma_max: float,
    hidden_dim: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    lambda_delta: float,
    device: str | torch.device | None,
) -> None:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n = len(dataset)
    if n < 2:
        raise ValueError("Need at least 2 samples to train/validate feasibility model.")

    if val_dataset is not None:
        train_set, val_set = dataset, val_dataset
    else:
        val_size = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
        val_size = min(val_size, n - 1)
        train_size = n - val_size
        generator = torch.Generator().manual_seed(0)
        if val_size > 0:
            train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_set, val_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False) if val_set else None

    debug_batch = next(iter(train_loader))
    debug_history, debug_action, debug_z_next = _split_batch(debug_batch)
    print("DEBUG first batch history:", tuple(debug_history.shape))
    print("DEBUG first batch action:", tuple(debug_action.shape))
    print("DEBUG first batch z_next:", tuple(debug_z_next.shape))

    if architecture == "mlp":
        history_dim = int(dataset.histories[0].numel())
        action_dim = int(dataset.actions.shape[-1])
        latent_dim = int(dataset.next_latents[0].numel())
        model = FeasibilityModel(
            history_dim=history_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            noise_level=noise_level,
        ).to(device)
    elif architecture == "transformer":
        latent_shape = tuple(dataset.next_latents.shape[1:])
        history_length = int(dataset.histories.shape[1])
        action_dim = int(dataset.actions.shape[-1])
        model = TransformerFeasibilityModel(
            action_dim=action_dim,
            latent_shape=latent_shape,
            history_length=history_length,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    print("=== Model Config ===")
    for k, v in model.config_dict().items():
        print(f"  {k}: {v}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("====================")

    scheduler = LogUniformSigmaScheduler(sigma_min=sigma_min, sigma_max=sigma_max)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    checkpoint_out = Path(checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            history, action, z_next = _split_batch(batch)
            history = history.to(device)
            action = action.to(device)
            z_next = z_next.to(device)

            dsm_loss = model.dsm_loss(history, action, z_next, scheduler=scheduler, reduction="mean")
            delta_loss = model.delta_loss(history, action, z_next, reduction="mean")
            loss = dsm_loss + lambda_delta * delta_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * history.shape[0]
            train_count += history.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = train_loss

        if val_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for batch in val_loader:
                    history, action, z_next = _split_batch(batch)
                    history = history.to(device)
                    action = action.to(device)
                    z_next = z_next.to(device)
                    dsm_loss = model.dsm_loss(history, action, z_next, noise_level=noise_level, reduction="mean")
                    delta_loss = model.delta_loss(history, action, z_next, reduction="mean")
                    loss = dsm_loss + lambda_delta * delta_loss
                    total += float(loss.cpu()) * history.shape[0]
                    count += history.shape[0]
            val_loss = total / max(count, 1)

        print(f"epoch {epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config_dict(),
                    "dataset_metadata": dataset_metadata or {},
                    "val_dataset_metadata": val_dataset_metadata or {},
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lambda_delta": lambda_delta,
                },
                checkpoint_out,
            )
            print(f"saved best checkpoint to: {checkpoint_out}")

    final_out = checkpoint_out.with_name(checkpoint_out.stem + "_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config_dict(),
            "dataset_metadata": dataset_metadata or {},
            "val_dataset_metadata": val_dataset_metadata or {},
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lambda_delta": lambda_delta,
        },
        final_out,
    )
    print(f"saved final checkpoint to: {final_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Wall feasibility model from saved tensor dataset.")
    parser.add_argument("--dataset", required=True, help="Path to .pt with histories/actions/next_latents")
    parser.add_argument("--val-dataset", default=None, help="Optional separate validation dataset .pt")
    parser.add_argument("--checkpoint-out", default="checkpoints/wall_feasibility.pt")
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda-delta", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, metadata = load_tensor_dataset(args.dataset)
    val_dataset = None
    val_metadata = None
    if args.val_dataset is not None:
        val_dataset, val_metadata = load_tensor_dataset(args.val_dataset)
    print("architecture:", args.architecture)
    validate_dataset_shapes(dataset, args.architecture)
    print("dataset:", len(dataset))
    print("histories:", tuple(dataset.histories.shape))
    print("actions:", tuple(dataset.actions.shape))
    print("next_latents:", tuple(dataset.next_latents.shape))
    print("metadata keys:", sorted(metadata.keys()))
    if val_dataset is not None:
        validate_dataset_shapes(val_dataset, args.architecture)
        print("val_dataset:", len(val_dataset))
        print("val histories:", tuple(val_dataset.histories.shape))
        print("val actions:", tuple(val_dataset.actions.shape))
        print("val next_latents:", tuple(val_dataset.next_latents.shape))
        print("val metadata keys:", sorted((val_metadata or {}).keys()))

    train_feasibility_model(
        dataset=dataset,
        val_dataset=val_dataset,
        dataset_metadata=metadata,
        val_dataset_metadata=val_metadata,
        architecture=args.architecture,
        checkpoint_out=args.checkpoint_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        noise_level=args.noise_level,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        hidden_dim=args.hidden_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lambda_delta=args.lambda_delta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
