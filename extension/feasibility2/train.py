"""
Train a transformer feasibility model from a saved tensor dataset.

Supports:
- DSM training with a log-uniform noise scheduler
- Optional delta loss
- Optional contrastive energy-ranking loss
- CUDA bfloat16 autocast
- Pinned-memory DataLoaders and non-blocking GPU transfers
- Reduced-cost contrastive validation

Checkpoint compatibility:
The DSM head keeps the existing ``out_proj`` name, preserving its state-dict
keys. Older transformer checkpoints have no ``energy_head`` parameters; they
can still be loaded for DSM inference, but their newly initialized energy head
must be trained before it is used for contrastive ranking.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

try:
    from .dataset import FeasibilityDataset, load_tensor_dataset
    from .model import TransformerFeasibilityModel
    from .scheduler import SigmaScheduler, LogUniformSigmaScheduler
except ImportError:
    from extension.feasibility2.dataset import (
        FeasibilityDataset,
        load_tensor_dataset,
    )
    from extension.feasibility2.model import TransformerFeasibilityModel
    from extension.feasibility2.scheduler import (
        SigmaScheduler,
        LogUniformSigmaScheduler,
    )


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def unpack_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 4:
        history, action, z_next, past_actions = batch
        return history, action, z_next, past_actions

    if len(batch) == 3:
        history, action, z_next = batch
        return history, action, z_next, None

    raise ValueError(f"Expected 3 or 4 tensors per batch, got {len(batch)}.")


def move_batch(
    batch: Any,
    device: torch.device,
    move_past_actions: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    history, action, z_next, past_actions = unpack_batch(batch)
    non_blocking = device.type == "cuda"

    history = history.to(device, non_blocking=non_blocking)
    action = action.to(device, non_blocking=non_blocking)
    z_next = z_next.to(device, non_blocking=non_blocking)

    if past_actions is not None and move_past_actions:
        past_actions = past_actions.to(device, non_blocking=non_blocking)
    else:
        past_actions = None

    return history, action, z_next, past_actions

def contrastive_energy_ranking_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    action: torch.Tensor,
    z_next: torch.Tensor,
    scheduler: SigmaScheduler,
    margin: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    batch_size = history.shape[0]

    if batch_size < 2:
        return history.new_zeros(())

    permutation = torch.roll(
        torch.arange(batch_size, device=history.device),
        shifts=1,
    )

    negative_action = action[permutation]

    sigma = scheduler.sample_sigmas(
        batch_size=batch_size,
        device=z_next.device,
        dtype=z_next.dtype,
    ).reshape(batch_size)

    energies = model.energy(
        torch.cat((history, history), dim=0),
        torch.cat((action, negative_action), dim=0),
        torch.cat((z_next, z_next), dim=0),
        noise_level=torch.cat((sigma, sigma), dim=0),
        reduction="none",
    )

    positive_energy, negative_energy = energies.chunk(2, dim=0)

    if debug:
        stats = torch.stack(
            [
                positive_energy.mean(),
                negative_energy.mean(),
                (negative_energy - positive_energy).mean(),
                sigma.mean(),
            ]
        ).detach().float().cpu()

        print(
            f"E_pos={stats[0].item():.6f} "
            f"E_neg={stats[1].item():.6f} "
            f"gap={stats[2].item():.6f} "
            f"sigma_mean={stats[3].item():.6f}"
        )

    return F.relu(
        margin + positive_energy - negative_energy
    ).mean()

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def save_checkpoint(
    path: Path,
    model: TransformerFeasibilityModel,
    metadata: dict | None,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config_dict(),
            "dataset_metadata": metadata or {},
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lambda_delta": args.lambda_delta,
            "delta_target_mode": args.delta_target_mode,
            "lambda_contrastive": args.lambda_contrastive,
            "contrastive_margin": args.contrastive_margin,
            "contrastive_neg_mode": args.contrastive_neg_mode,
            "contrastive_fraction": args.contrastive_fraction,
            "contrastive_every": args.contrastive_every,
        },
        path,
    )

    print(
        f"Saved checkpoint: {path} "
        f"({time.perf_counter() - start:.2f}s)"
    )


def train_feasibility_model(
    dataset: FeasibilityDataset,
    metadata: dict | None,
    scheduler: SigmaScheduler,
    world_model: torch.nn.Module | None,
    args: argparse.Namespace,
) -> TransformerFeasibilityModel:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        print("GPU:", torch.cuda.get_device_name(device))
        print("PyTorch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("bfloat16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("Warning: training on CPU.")

    sample_count = len(dataset)
    val_size = max(1, round(sample_count * args.val_fraction))
    val_size = min(val_size, sample_count - 1)
    train_size = sample_count - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    if args.max_val_samples > 0 and len(val_set) > args.max_val_samples:
        val_set = Subset(val_set, range(args.max_val_samples))

    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = TransformerFeasibilityModel(
        action_dim=int(dataset.actions.shape[-1]),
        latent_shape=tuple(dataset.next_latents.shape[1:]),
        history_length=int(dataset.histories.shape[1]),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    print("=== Model configuration ===")
    for key, value in model.config_dict().items():
        print(f"  {key}: {value}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  device: {next(model.parameters()).device}")
    print("===========================")

    latent_mean = metadata.get("latent_mean") if metadata else None
    latent_std = metadata.get("latent_std") if metadata else None

    if args.delta_target_mode == "wm_residual":
        if world_model is None:
            raise ValueError("wm_residual requires a world model.")
        if latent_mean is None or latent_std is None:
            raise ValueError(
                "wm_residual requires latent_mean and latent_std."
            )

        world_model = world_model.to(device)
        world_model.eval()

        for parameter in world_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = Path(args.checkpoint_out)
    best_checkpoint_metric = math.inf

    last_train_metrics: dict[str, float] = {}
    last_val_metrics: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        train_sums = {
            "total": 0.0,
            "dsm": 0.0,
            "delta": 0.0,
            "contrastive": 0.0,
        }
        train_count = 0
        contrastive_count = 0

        previous_step_end = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            batch_ready = time.perf_counter()
            profile = epoch == 1 and batch_index < args.profile_batches

            history, action, z_next, past_actions = move_batch(
                batch,
                device,
                move_past_actions=args.lambda_delta > 0.0,
            )

            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            transfer_done = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            with amp_context(device, not args.no_amp):
                dsm_loss = model.dsm_loss(
                    history,
                    action,
                    z_next,
                    scheduler=scheduler,
                    reduction="mean",
                )

            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            dsm_done = time.perf_counter()

            delta_loss = history.new_zeros(())

            if args.lambda_delta > 0.0:
                with amp_context(device, not args.no_amp):
                    delta_loss = model.delta_loss(
                        history,
                        action,
                        z_next,
                        reduction="mean",
                        target=args.delta_target_mode,
                        world_model=world_model,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        past_action_history=past_actions,
                    )

            contrastive_loss = history.new_zeros(())
            contrastive_weight = 0.0
            selected_count = 0

            if (
                args.lambda_contrastive > 0.0
                and batch_index % args.contrastive_every == 0
            ):
                selected_count = min(
                    history.shape[0],
                    max(
                        2,
                        round(
                            history.shape[0]
                            * args.contrastive_fraction
                        ),
                    ),
                )

                with amp_context(device, not args.no_amp):
                    contrastive_loss = contrastive_energy_ranking_loss(
                        model,
                        history[:selected_count],
                        action[:selected_count],
                        z_next[:selected_count],
                        margin=args.contrastive_margin,
                        scheduler=scheduler,
                        debug=(epoch == 1 and batch_index == 0),
                    )

                contrastive_weight = args.lambda_contrastive

            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            auxiliary_done = time.perf_counter()

            loss = (
                dsm_loss
                + args.lambda_delta * delta_loss
                + contrastive_weight * contrastive_loss
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_index}: {loss.item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            step_done = time.perf_counter()

            batch_count = history.shape[0]

            train_sums["total"] += loss.detach().float().item() * batch_count
            train_sums["dsm"] += dsm_loss.detach().float().item() * batch_count
            train_sums["delta"] += (
                delta_loss.detach().float().item() * batch_count
            )
            train_count += batch_count

            if selected_count >= 2:
                train_sums["contrastive"] += (
                    contrastive_loss.detach().float().item()
                    * selected_count
                )
                contrastive_count += selected_count

            if profile:
                print(
                    f"TIMING batch={batch_index} | "
                    f"data_wait={batch_ready - previous_step_end:.3f}s | "
                    f"transfer={transfer_done - batch_ready:.3f}s | "
                    f"dsm={dsm_done - transfer_done:.3f}s | "
                    f"aux={auxiliary_done - dsm_done:.3f}s | "
                    f"backward={step_done - auxiliary_done:.3f}s | "
                    f"total={step_done - batch_ready:.3f}s"
                )

            previous_step_end = step_done

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_end = time.perf_counter()

        train_metrics = {
            "total": train_sums["total"] / max(train_count, 1),
            "dsm": train_sums["dsm"] / max(train_count, 1),
            "delta": train_sums["delta"] / max(train_count, 1),
            "contrastive": (
                train_sums["contrastive"]
                / max(contrastive_count, 1)
            ),
        }

        model.eval()
        val_start = time.perf_counter()

        val_dsm_sum = 0.0
        val_delta_sum = 0.0
        val_count = 0
        val_contrastive_sum = 0.0
        val_contrastive_count = 0

        run_contrastive_val = (
            args.lambda_contrastive > 0.0
            and args.contrastive_val_batches > 0
            and (
                epoch == 1
                or epoch == args.epochs
                or epoch % args.contrastive_val_every == 0
            )
        )

        with torch.inference_mode():
            for val_batch_index, batch in enumerate(val_loader):
                history, action, z_next, past_actions = move_batch(
                    batch,
                    device,
                    move_past_actions=args.lambda_delta > 0.0,
                )

                with amp_context(device, not args.no_amp):
                    dsm_loss = model.dsm_loss(
                        history,
                        action,
                        z_next,
                        #noise_level=args.noise_level,
                        scheduler=scheduler,
                        reduction="mean",
                    )

                    delta_loss = history.new_zeros(())

                    if args.lambda_delta > 0.0:
                        delta_loss = model.delta_loss(
                            history,
                            action,
                            z_next,
                            reduction="mean",
                            target=args.delta_target_mode,
                            world_model=world_model,
                            latent_mean=latent_mean,
                            latent_std=latent_std,
                            past_action_history=past_actions,
                        )

                batch_count = history.shape[0]
                val_dsm_sum += dsm_loss.float().item() * batch_count
                val_delta_sum += delta_loss.float().item() * batch_count
                val_count += batch_count

                if (
                    run_contrastive_val
                    and val_batch_index < args.contrastive_val_batches
                ):
                    selected_count = min(
                        batch_count,
                        max(
                            2,
                            round(
                                batch_count
                                * args.contrastive_fraction
                            ),
                        ),
                    )

                    with amp_context(device, not args.no_amp):
                        contrastive_loss = (
                            contrastive_energy_ranking_loss(
                                model,
                                history[:selected_count],
                                action[:selected_count],
                                z_next[:selected_count],
                                margin=args.contrastive_margin,
                                scheduler=scheduler,
                                debug=(epoch == 1 and val_batch_index == 0),
                            )
                        )

                    val_contrastive_sum += (
                        contrastive_loss.float().item()
                        * selected_count
                    )
                    val_contrastive_count += selected_count

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_end = time.perf_counter()

        val_dsm = val_dsm_sum / max(val_count, 1)
        val_delta = val_delta_sum / max(val_count, 1)

        val_metrics = {
            "dsm": val_dsm,
            "delta": val_delta,
            "contrastive": (
                val_contrastive_sum / val_contrastive_count
                if val_contrastive_count > 0
                else float("nan")
            ),
        }

        # Use stable, full-validation metrics for checkpoint selection.
        checkpoint_metric = val_dsm + args.lambda_delta * val_delta

        print(
            f"epoch {epoch:04d} | "
            f"train_total={train_metrics['total']:.6f} "
            f"train_dsm={train_metrics['dsm']:.6f} "
            f"train_contrastive={train_metrics['contrastive']:.6f} | "
            f"val_dsm={val_metrics['dsm']:.6f} "
            f"val_contrastive={val_metrics['contrastive']:.6f}"
        )

        print(
            "TIMING | "
            f"train={(train_end - epoch_start) / 60:.2f} min | "
            f"validation={(val_end - val_start) / 60:.2f} min | "
            f"total={(val_end - epoch_start) / 60:.2f} min"
        )

        if checkpoint_metric <= best_checkpoint_metric:
            best_checkpoint_metric = checkpoint_metric
            save_checkpoint(
                checkpoint_path,
                model,
                metadata,
                epoch,
                train_metrics,
                val_metrics,
                args,
            )

        last_train_metrics = train_metrics
        last_val_metrics = val_metrics

    final_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_final{checkpoint_path.suffix}"
    )

    save_checkpoint(
        final_path,
        model,
        metadata,
        args.epochs,
        last_train_metrics,
        last_val_metrics,
        args,
    )

    return model


def validate_dataset(dataset: FeasibilityDataset) -> None:
    if dataset.histories.ndim != 4:
        raise ValueError(
            "Expected histories [N, H, 196, 394], got "
            f"{tuple(dataset.histories.shape)}"
        )

    if dataset.actions.ndim != 2:
        raise ValueError(
            f"Expected actions [N, 10], got {tuple(dataset.actions.shape)}"
        )

    if dataset.next_latents.ndim != 3:
        raise ValueError(
            "Expected next_latents [N, 196, 394], got "
            f"{tuple(dataset.next_latents.shape)}"
        )

    counts = {
        dataset.histories.shape[0],
        dataset.actions.shape[0],
        dataset.next_latents.shape[0],
    }

    if len(counts) != 1:
        raise ValueError("Dataset tensors have different sample counts.")

    if dataset.histories.shape[1:] != (3, 196, 394):
        raise ValueError(
            "Expected histories shape [N, 3, 196, 394], got "
            f"{tuple(dataset.histories.shape)}"
        )

    if dataset.actions.shape[1] != 10:
        raise ValueError(
            f"Expected action dimension 10, got {dataset.actions.shape[1]}"
        )

    if dataset.next_latents.shape[1:] != (196, 394):
        raise ValueError(
            "Expected next_latents shape [N, 196, 394], got "
            f"{tuple(dataset.next_latents.shape)}"
        )

    # Routine sanity check without scanning the full multi-GB dataset.
    for name, tensor in (
        ("histories", dataset.histories),
        ("actions", dataset.actions),
        ("next_latents", dataset.next_latents),
    ):
        if not torch.isfinite(tensor[:128]).all():
            raise ValueError(
                f"{name} contains NaN or Inf in the first 128 samples."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the transformer feasibility model."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/feasibility.pt",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)

    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.5)

    parser.add_argument("--lambda-delta", type=float, default=0.0)
    parser.add_argument(
        "--delta-target-mode",
        choices=("current_delta", "wm_residual"),
        default="current_delta",
    )

    parser.add_argument("--lambda-contrastive", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-neg-mode",
        choices=("wrong_action",),
        default="wrong_action",
    )
    parser.add_argument(
        "--contrastive-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--contrastive-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--contrastive-val-batches",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--contrastive-val-every",
        type=int,
        default=5,
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--profile-batches", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.contrastive_fraction <= 1.0:
        raise ValueError("--contrastive-fraction must be in (0, 1].")

    if args.contrastive_every < 1:
        raise ValueError("--contrastive-every must be at least 1.")

    if args.contrastive_val_every < 1:
        raise ValueError("--contrastive-val-every must be at least 1.")

    start = time.perf_counter()

    dataset, metadata = load_tensor_dataset(args.dataset)
    validate_dataset(dataset)

    print("Dataset samples:", len(dataset))
    print("Histories:", tuple(dataset.histories.shape))
    print("Actions:", tuple(dataset.actions.shape))
    print("Next latents:", tuple(dataset.next_latents.shape))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    world_model = None

    if args.delta_target_mode == "wm_residual":
        from extension.feasibility2.build import load_world_model

        model_cfg = metadata.get("model_cfg")
        model_ckpt = metadata.get("model_ckpt")

        if model_cfg is None or model_ckpt is None:
            raise ValueError(
                "wm_residual requires model_cfg and model_ckpt metadata."
            )

        world_model, action_repeat = load_world_model(
            device=device,
            model_cfg_path=Path(model_cfg),
            model_ckpt_path=Path(model_ckpt),
            primitive_action_dim=2,
            expected_action_repeat=5,
        )

        print("Loaded frozen DINO-WM.")
        print("Action repeat:", action_repeat)

    train_feasibility_model(
        dataset=dataset,
        metadata=metadata,
        scheduler=LogUniformSigmaScheduler(
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        ),
        world_model=world_model,
        args=args,
    )

    print(
        f"Total runtime: "
        f"{(time.perf_counter() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
