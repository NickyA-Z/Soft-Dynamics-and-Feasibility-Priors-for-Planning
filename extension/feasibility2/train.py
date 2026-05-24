"""
Trains the feasibility model on the saved tensor dataset from build.py.
Loads the dataset, splits train/validation, runs DSM training, and saves checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import FeasibilityDataset, load_tensor_dataset
    from .model import FeasibilityModel, TransformerFeasibilityModel
    from .scheduler import SigmaScheduler, LogUniformSigmaScheduler
except ImportError:
    from nicky_dl.feasibility2.dataset import FeasibilityDataset, load_tensor_dataset
    from nicky_dl.feasibility2.model import FeasibilityModel, TransformerFeasibilityModel
    from nicky_dl.feasibility2.scheduler import SigmaScheduler, LogUniformSigmaScheduler



def train_feasibility_model(
    dataset: FeasibilityDataset,
    dataset_metadata: dict | None = None,
    architecture: str = "transformer",
    hidden_dim: int = 256,
    model_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 3,
    scheduler: SigmaScheduler = None,
    noise_level: float = 0.05,  # prioritizes scheduler over noise_level, used for debug
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_fraction: float = 0.1,
    device: str | torch.device | None = None,
    checkpoint_out: str | Path | None = None,

    delta_target_mode: str = "current_delta",
    lambda_delta: float = 1.0,
    world_model: torch.nn.Module | None = None,
) -> FeasibilityModel:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    n = len(dataset)
    if n < 2:
        raise ValueError("Need at least 2 samples to train/validate feasibility model.")

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
    if len(debug_batch) == 4:
        debug_history, debug_action, debug_z_next, debug_past_action_history = debug_batch
        print("DEBUG first batch past_action_history:", tuple(debug_past_action_history.shape))
    else:
        debug_history, debug_action, debug_z_next = next(iter(train_loader))
    print("DEBUG first batch history:", tuple(debug_history.shape))
    print("DEBUG first batch action:", tuple(debug_action.shape))
    print("DEBUG first batch z_next:", tuple(debug_z_next.shape))

    if architecture == "mlp":
        """Trains a MLP model on the given dataset."""
        """Trains a MLP model on the given dataset."""
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
        debug_history, debug_action, debug_z_next = next(iter(train_loader))

    elif architecture == "transformer":
        """Trains a transformer model on the given dataset."""
        latent_shape = tuple(dataset.next_latents.shape[1:])
        history_length = int(dataset.histories.shape[1])
        action_dim = int(dataset.actions.shape[-1])

        model = TransformerFeasibilityModel(
            action_dim=action_dim,
            latent_shape=latent_shape,  # should be (196, 394)
            history_length=history_length,  # should be 3
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(device)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # print model config
    print("=== Model Config ===")
    for k, v in model.config_dict().items():
        print(f"  {k}: {v}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("====================")
    # print model config
    print("=== Model Config ===")
    for k, v in model.config_dict().items():
        print(f"  {k}: {v}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("====================")

    print("DEBUG batch history:", tuple(debug_history.shape))
    print("DEBUG batch action:", tuple(debug_action.shape))
    print("DEBUG batch z_next:", tuple(debug_z_next.shape))
    debug_history = debug_history.to(device)
    debug_action = debug_action.to(device)
    debug_z_next = debug_z_next.to(device)
    with torch.no_grad():
        debug_loss = model.dsm_loss(
            debug_history,
            debug_action,
            debug_z_next,
            noise_level=noise_level,
            reduction="mean",
        )
    print("DEBUG dsm loss:", float(debug_loss.cpu()))

    if scheduler is None:
        scheduler = LogUniformSigmaScheduler(0.01, 0.5)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")

    # before training loop
    latent_mean = None
    latent_std = None
    if dataset_metadata is not None:
        latent_mean = dataset_metadata.get("latent_mean", None)
        latent_std = dataset_metadata.get("latent_std", None)

    # fallback if your dataset object stores them
    if latent_mean is None and hasattr(dataset, "latent_mean"):
        latent_mean = dataset.latent_mean
    if latent_std is None and hasattr(dataset, "latent_std"):
        latent_std = dataset.latent_std

    if delta_target_mode == "wm_residual":
        if world_model is None:
            raise ValueError("delta_target_mode='wm_residual' requires world_model")
        if latent_mean is None or latent_std is None:
            raise ValueError("wm_residual requires latent_mean and latent_std")
        world_model = world_model.to(device)
        world_model.eval()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        #for history, action, z_next in train_loader:
        for batch in train_loader:
            if len(batch) == 4:
                history, action, z_next, past_action_history = batch
                past_action_history = past_action_history.to(device)
            else:
                history, action, z_next = batch
                past_action_history = None

            history = history.to(device)
            action = action.to(device)
            z_next = z_next.to(device)
            #loss = model.dsm_loss(history, action, z_next, scheduler=scheduler, reduction="mean")  # mean over batch

            # old loss 
            dsm_loss = model.dsm_loss(history,action,z_next,scheduler=scheduler,reduction="mean")
            # new delta-based loss
            
            delta_loss = model.delta_loss(history,action,z_next,reduction="mean", target=delta_target_mode, world_model=world_model, latent_mean=latent_mean, latent_std=latent_std, past_action_history=past_action_history)
            loss = dsm_loss + lambda_delta * delta_loss
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * history.shape[0]
            # new Prediciting Latents loss
            #train_loss_sum += float(loss.detach().cpu()) * history.shape[0]
            train_count += history.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                # for history, action, z_next in val_loader:
                for batch in val_loader:
                    if len(batch) == 4:
                        history, action, z_next, past_action_history = batch
                        past_action_history = past_action_history.to(device)
                    else:
                        history, action, z_next = batch
                        past_action_history = None

                    history = history.to(device)
                    action = action.to(device)
                    z_next = z_next.to(device)
                    #loss = model.dsm_loss(history, action, z_next, scheduler=scheduler, reduction="mean")
                    # old loss 
                    dsm_loss = model.dsm_loss(history,action,z_next,noise_level=noise_level,reduction="mean")
                    # new delta-based loss
                    delta_loss = model.delta_loss(history,action,z_next,reduction="mean", target=delta_target_mode, world_model=world_model, latent_mean=latent_mean, latent_std=latent_std, past_action_history=past_action_history)
                    loss = dsm_loss + lambda_delta * delta_loss

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite validation loss at epoch {epoch}: {loss.item()}")
                    total += float(loss.cpu()) * history.shape[0]
                    count += history.shape[0]
            val_loss = total / max(count, 1)

        print(f"epoch {epoch:04d} train_dsm={train_loss:.6f} val_dsm={val_loss:.6f}")
        if checkpoint_out is not None and val_loss <= best_val:
            best_val = val_loss
            out = Path(checkpoint_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config_dict(),
                    "dataset_metadata": dataset_metadata or {},
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lambda_delta": lambda_delta,

                    "delta_target_mode": delta_target_mode,
                },
                out,
            )
            print(f"saved best checkpoint to: {out}")

    if checkpoint_out is not None:
        final_out = Path(checkpoint_out).with_name(Path(checkpoint_out).stem + "_final.pt")
        final_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model.config_dict(),
                "dataset_metadata": dataset_metadata or {},
                "epoch": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lambda_delta": lambda_delta,
            },
            final_out,
        )
        print(f"saved final checkpoint to: {final_out}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learned feasibility model from saved tensor dataset.")
    p.add_argument("--architecture", choices=["mlp", "transformer"], default="transformer")
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dataset", required=True, help="Path to .pt with histories/actions/next_latents")
    p.add_argument("--checkpoint-out", default="checkpoints/feasibility.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--noise-level", type=float, default=0.05)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--device", default=None)

    p.add_argument("--lambda-delta", type=float, default=1.0)
    p.add_argument("--delta-target-mode", choices=("current_delta", "wm_residual"), default="current_delta")

    return p.parse_args()


def validate_dataset_shapes(dataset: FeasibilityDataset, architecture: str) -> None:
    if dataset.actions.ndim != 2:
        raise ValueError(f"Expected actions [N, action_dim], got {tuple(dataset.actions.shape)}")

    if not (dataset.histories.shape[0] == dataset.actions.shape[0] == dataset.next_latents.shape[0]):
        raise ValueError(
            "histories/actions/next_latents sample count mismatch: "
            f"{dataset.histories.shape[0]}, "
            f"{dataset.actions.shape[0]}, "
            f"{dataset.next_latents.shape[0]}"
        )

    if dataset.actions.shape[-1] != 10:
        raise ValueError(f"Expected action_dim=10, got {dataset.actions.shape[-1]}")

    if dataset.histories.shape[1] != 3:
        raise ValueError(f"Expected history_length=3, got {dataset.histories.shape[1]}")

    if architecture == "mlp":
        if dataset.histories.ndim != 3:
            raise ValueError(f"Expected MLP histories [N, H, latent_dim], got {tuple(dataset.histories.shape)}")

        if dataset.next_latents.ndim != 2:
            raise ValueError(f"Expected MLP next_latents [N, latent_dim], got {tuple(dataset.next_latents.shape)}")

        if dataset.histories.shape[-1] != dataset.next_latents.shape[-1]:
            raise ValueError(
                f"History latent dim {dataset.histories.shape[-1]} does not match "
                f"next_latent dim {dataset.next_latents.shape[-1]}"
            )

    elif architecture == "transformer":
        if dataset.histories.ndim < 4:
            raise ValueError(
                f"Expected transformer histories [N, H, *latent_shape], got {tuple(dataset.histories.shape)}"
            )

        if dataset.next_latents.ndim < 3:
            raise ValueError(
                f"Expected transformer next_latents [N, *latent_shape], got {tuple(dataset.next_latents.shape)}"
            )

        latent_shape = tuple(dataset.next_latents.shape[1:])
        history_latent_shape = tuple(dataset.histories.shape[2:])

        if history_latent_shape != latent_shape:
            raise ValueError(
                f"History latent shape {history_latent_shape} does not match "
                f"next_latents latent shape {latent_shape}"
            )

        if latent_shape != (196, 394):
            raise ValueError(f"Expected DINO latent shape (196, 394), got {latent_shape}")
            raise ValueError(f"Expected DINO latent shape (196, 394), got {latent_shape}")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    for name, tensor in [
        ("histories", dataset.histories),
        ("actions", dataset.actions),
        ("next_latents", dataset.next_latents),
    ]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")



def main() -> None:
    args = parse_args()

    # load_tensor_dataset
    dataset, metadata = load_tensor_dataset(args.dataset)
    print("architecture:", args.architecture)
    validate_dataset_shapes(dataset, args.architecture)

    print("dataset:", len(dataset))
    print("histories:", tuple(dataset.histories.shape))
    print("actions:", tuple(dataset.actions.shape))
    print("next_latents:", tuple(dataset.next_latents.shape))

    print("history mean/std:", dataset.histories.mean().item(), dataset.histories.std().item())
    print("action mean/std:", dataset.actions.mean().item(), dataset.actions.std().item())
    print("next_latents mean/std:", dataset.next_latents.mean().item(), dataset.next_latents.std().item())

    print("metadata keys:", sorted(metadata.keys()))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    world_model = None
    if args.delta_target_mode == "wm_residual":
        from nicky_dl.feasibility2.build import load_world_model
        model_cfg = Path(metadata["model_cfg"])
        model_ckpt = Path(metadata["model_ckpt"])
        if model_cfg is None or model_ckpt is None:
            raise ValueError(
                "Dataset metadata must contain model_cfg and model_ckpt "
                "to train wm_residual."
            )
        world_model, action_repeat = load_world_model(
            device=device,
            model_cfg_path=model_cfg,
            model_ckpt_path=model_ckpt,
            primitive_action_dim=2,
            expected_action_repeat=5,
        )
        world_model.eval()
        for p in world_model.parameters():
            p.requires_grad_(False)
        print("Loaded frozen DINO-WM for wm_residual target")
        print("action repeat:", action_repeat)

    train_feasibility_model(
        dataset=dataset,
        dataset_metadata=metadata,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        noise_level=args.noise_level,
        scheduler=LogUniformSigmaScheduler(sigma_min=args.sigma_min, sigma_max=args.sigma_max),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        device=args.device,
        checkpoint_out=args.checkpoint_out,

        delta_target_mode=args.delta_target_mode,
        lambda_delta=args.lambda_delta,
        world_model=world_model,
    )


if __name__ == "__main__":
    main()
