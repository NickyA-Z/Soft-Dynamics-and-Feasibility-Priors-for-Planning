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

Action DSM adds only ``action_out_proj`` to the same backbone. Old checkpoints
can initialize it, but direct action sampling requires fine-tuning with a
positive ``--lambda-action-dsm``.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split

from gvpwm.feasibility_prior.dataset import FeasibilityDataset, load_tensor_dataset
from gvpwm.feasibility_prior.model import TransformerFeasibilityModel
from gvpwm.feasibility_prior.hard_negatives import HardNegativeDataset, load_hard_negative_dataset
from gvpwm.feasibility_prior.losses import contrastive_energy_loss, paired_energy_ranking_loss
from gvpwm.feasibility_prior.scheduler import SigmaScheduler, LogUniformSigmaScheduler


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


def move_hard_negative_batch(batch: Any, device: torch.device):
    if len(batch) != 5:
        raise ValueError(f"Expected 5 hard-negative tensors, got {len(batch)}")
    non_blocking = device.type == "cuda"
    return tuple(item.to(device, non_blocking=non_blocking) for item in batch)


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

    sd = model.state_dict()

    if (args.lambda_contrastive > 0.0 or args.lambda_hard_negative > 0.0) and not any(
        k.startswith("energy_head") for k in sd
    ):
        raise RuntimeError(
            "lambda_contrastive > 0, but model.state_dict() has no energy_head."
        )

    torch.save(
        {
            "model_state_dict": sd,
            "config": model.config_dict(),
            "dataset_metadata": metadata or {},
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lambda_delta": args.lambda_delta,
            "lambda_latent_dsm": args.lambda_latent_dsm,
            "lambda_action_dsm": args.lambda_action_dsm,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "delta_target_mode": args.delta_target_mode,
            "lambda_contrastive": args.lambda_contrastive,
            "contrastive_margin": args.contrastive_margin,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_noise_level": args.contrastive_noise_level,
            "contrastive_hard_weight": args.contrastive_hard_weight,
            "contrastive_calibration_margin": args.contrastive_calibration_margin,
            "contrastive_calibration_weight": args.contrastive_calibration_weight,
            "contrastive_num_action_shuffles": args.contrastive_num_action_shuffles,
            "contrastive_latent_noise_scales": args.contrastive_latent_noise_scales,
            "contrastive_fraction": args.contrastive_fraction,
            "contrastive_every": args.contrastive_every,
            "lambda_hard_negative": args.lambda_hard_negative,
            "hard_negative_margin": args.hard_negative_margin,
            "hard_negative_temperature": args.hard_negative_temperature,
            "hard_negative_noise_level": args.hard_negative_noise_level,
        },
        path,
    )

    print(f"Saved checkpoint: {path} " f"({time.perf_counter() - start:.2f}s)")


def train_feasibility_model(
    dataset: FeasibilityDataset,
    metadata: dict | None,
    scheduler: SigmaScheduler,
    world_model: torch.nn.Module | None,
    args: argparse.Namespace,
    hard_negative_dataset: HardNegativeDataset | None = None,
    hard_negative_val_dataset: HardNegativeDataset | None = None,
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
    hard_loader = None
    hard_val_loader = None
    if hard_negative_dataset is not None:
        hard_loader = make_loader(
            hard_negative_dataset,
            args.hard_negative_batch_size or args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    if hard_negative_val_dataset is not None:
        hard_val_loader = make_loader(
            hard_negative_val_dataset,
            args.hard_negative_batch_size or args.batch_size,
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

    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device)
        incompatible = model.load_state_dict(
            initial["model_state_dict"], strict=False
        )
        allowed_missing = {
            key for key in incompatible.missing_keys
            if key.startswith(("energy_head.", "action_out_proj."))
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible initialization checkpoint: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(f"Initialized model from {args.init_checkpoint}")

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
            raise ValueError("wm_residual requires latent_mean and latent_std.")

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
            "action_dsm": 0.0,
            "delta": 0.0,
            "contrastive": 0.0,
            "hard_negative": 0.0,
            "hard_gap": 0.0,
            "hard_ranking_accuracy": 0.0,
            "hard_margin_accuracy": 0.0,
        }
        train_count = 0
        contrastive_count = 0
        hard_negative_count = 0
        hard_iterator = iter(hard_loader) if hard_loader is not None else None

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
                action_dsm_loss = history.new_zeros(())
                if args.lambda_action_dsm > 0.0:
                    action_dsm_loss = model.action_dsm_loss(
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
                        round(history.shape[0] * args.contrastive_fraction),
                    ),
                )

                with amp_context(device, not args.no_amp):
                    contrastive_loss = contrastive_energy_loss(
                        model,
                        history[:selected_count],
                        action[:selected_count],
                        z_next[:selected_count],
                        noise_level=args.contrastive_noise_level,
                        temperature=args.contrastive_temperature,
                        ranking_margin=args.contrastive_margin,
                        hard_negative_weight=args.contrastive_hard_weight,
                        calibration_margin=args.contrastive_calibration_margin,
                        calibration_weight=args.contrastive_calibration_weight,
                        num_action_shuffles=args.contrastive_num_action_shuffles,
                        latent_noise_scales=tuple(args.contrastive_latent_noise_scales),
                        debug=(epoch == 1 and batch_index == 0),
                    )

                contrastive_weight = args.lambda_contrastive

            hard_negative_loss = history.new_zeros(())
            hard_metrics = None
            hard_batch_count = 0
            if hard_iterator is not None and args.lambda_hard_negative > 0.0:
                try:
                    hard_batch = next(hard_iterator)
                except StopIteration:
                    hard_iterator = iter(hard_loader)
                    hard_batch = next(hard_iterator)
                (
                    hard_history,
                    hard_positive_action,
                    hard_positive_next,
                    hard_negative_action,
                    hard_negative_next,
                ) = move_hard_negative_batch(hard_batch, device)
                hard_batch_count = hard_history.shape[0]
                with amp_context(device, not args.no_amp):
                    hard_negative_loss, hard_metrics = paired_energy_ranking_loss(
                        model,
                        hard_history,
                        hard_positive_action,
                        hard_positive_next,
                        hard_negative_action,
                        hard_negative_next,
                        noise_level=args.hard_negative_noise_level,
                        temperature=args.hard_negative_temperature,
                        ranking_margin=args.hard_negative_margin,
                    )

            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            auxiliary_done = time.perf_counter()

            loss = (
                args.lambda_latent_dsm * dsm_loss
                + args.lambda_action_dsm * action_dsm_loss
                + args.lambda_delta * delta_loss
                + contrastive_weight * contrastive_loss
                + args.lambda_hard_negative * hard_negative_loss
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
            train_sums["action_dsm"] += (
                action_dsm_loss.detach().float().item() * batch_count
            )
            train_sums["delta"] += delta_loss.detach().float().item() * batch_count
            train_count += batch_count

            if selected_count >= 2:
                train_sums["contrastive"] += (
                    contrastive_loss.detach().float().item() * selected_count
                )
                contrastive_count += selected_count
            if hard_metrics is not None:
                train_sums["hard_negative"] += (
                    hard_negative_loss.detach().float().item() * hard_batch_count
                )
                train_sums["hard_gap"] += (
                    hard_metrics["energy_gap"].float().item() * hard_batch_count
                )
                train_sums["hard_ranking_accuracy"] += (
                    hard_metrics["ranking_accuracy"].float().item() * hard_batch_count
                )
                train_sums["hard_margin_accuracy"] += (
                    hard_metrics["margin_accuracy"].float().item() * hard_batch_count
                )
                hard_negative_count += hard_batch_count

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
            "action_dsm": train_sums["action_dsm"] / max(train_count, 1),
            "delta": train_sums["delta"] / max(train_count, 1),
            "contrastive": (train_sums["contrastive"] / max(contrastive_count, 1)),
            "hard_negative": train_sums["hard_negative"] / max(hard_negative_count, 1),
            "hard_gap": train_sums["hard_gap"] / max(hard_negative_count, 1),
            "hard_ranking_accuracy": train_sums["hard_ranking_accuracy"]
            / max(hard_negative_count, 1),
            "hard_margin_accuracy": train_sums["hard_margin_accuracy"]
            / max(hard_negative_count, 1),
        }

        model.eval()
        val_start = time.perf_counter()

        val_dsm_sum = 0.0
        val_action_dsm_sum = 0.0
        val_delta_sum = 0.0
        val_count = 0
        val_contrastive_sum = 0.0
        val_contrastive_count = 0
        val_hard_sums = {"loss": 0.0, "gap": 0.0, "ranking": 0.0, "margin": 0.0}
        val_hard_count = 0

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
                        # noise_level=args.noise_level,
                        scheduler=scheduler,
                        reduction="mean",
                    )
                    action_dsm_loss = history.new_zeros(())
                    if args.lambda_action_dsm > 0.0:
                        action_dsm_loss = model.action_dsm_loss(
                            history,
                            action,
                            z_next,
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
                val_action_dsm_sum += (
                    action_dsm_loss.float().item() * batch_count
                )
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
                            round(batch_count * args.contrastive_fraction),
                        ),
                    )

                    with amp_context(device, not args.no_amp):
                        contrastive_loss = contrastive_energy_loss(
                            model,
                            history[:selected_count],
                            action[:selected_count],
                            z_next[:selected_count],
                            noise_level=args.contrastive_noise_level,
                            temperature=args.contrastive_temperature,
                            ranking_margin=args.contrastive_margin,
                            hard_negative_weight=args.contrastive_hard_weight,
                            calibration_margin=args.contrastive_calibration_margin,
                            calibration_weight=args.contrastive_calibration_weight,
                            num_action_shuffles=args.contrastive_num_action_shuffles,
                            latent_noise_scales=tuple(
                                args.contrastive_latent_noise_scales
                            ),
                            debug=(epoch == 1 and val_batch_index == 0),
                        )

                    val_contrastive_sum += (
                        contrastive_loss.float().item() * selected_count
                    )
                    val_contrastive_count += selected_count

            if hard_val_loader is not None and args.lambda_hard_negative > 0.0:
                for hard_batch_index, hard_batch in enumerate(hard_val_loader):
                    if (
                        args.hard_negative_val_batches > 0
                        and hard_batch_index >= args.hard_negative_val_batches
                    ):
                        break
                    hard_tensors = move_hard_negative_batch(hard_batch, device)
                    with amp_context(device, not args.no_amp):
                        hard_loss, hard_metrics = paired_energy_ranking_loss(
                            model,
                            *hard_tensors,
                            noise_level=args.hard_negative_noise_level,
                            temperature=args.hard_negative_temperature,
                            ranking_margin=args.hard_negative_margin,
                        )
                    count = hard_tensors[0].shape[0]
                    val_hard_sums["loss"] += hard_loss.float().item() * count
                    val_hard_sums["gap"] += (
                        hard_metrics["energy_gap"].float().item() * count
                    )
                    val_hard_sums["ranking"] += (
                        hard_metrics["ranking_accuracy"].float().item() * count
                    )
                    val_hard_sums["margin"] += (
                        hard_metrics["margin_accuracy"].float().item() * count
                    )
                    val_hard_count += count

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_end = time.perf_counter()

        val_dsm = val_dsm_sum / max(val_count, 1)
        val_action_dsm = val_action_dsm_sum / max(val_count, 1)
        val_delta = val_delta_sum / max(val_count, 1)

        val_metrics = {
            "dsm": val_dsm,
            "action_dsm": val_action_dsm,
            "delta": val_delta,
            "contrastive": (
                val_contrastive_sum / val_contrastive_count
                if val_contrastive_count > 0
                else float("nan")
            ),
            "hard_negative": (
                val_hard_sums["loss"] / val_hard_count
                if val_hard_count
                else float("nan")
            ),
            "hard_gap": (
                val_hard_sums["gap"] / val_hard_count
                if val_hard_count
                else float("nan")
            ),
            "hard_ranking_accuracy": (
                val_hard_sums["ranking"] / val_hard_count
                if val_hard_count
                else float("nan")
            ),
            "hard_margin_accuracy": (
                val_hard_sums["margin"] / val_hard_count
                if val_hard_count
                else float("nan")
            ),
        }

        checkpoint_metric = (
            args.lambda_latent_dsm * val_dsm
            + args.lambda_action_dsm * val_action_dsm
            + args.lambda_delta * val_delta
        )
        if args.lambda_contrastive > 0.0:
            val_contrastive = val_metrics["contrastive"]
            checkpoint_metric = (
                checkpoint_metric + args.lambda_contrastive * val_contrastive
                if math.isfinite(val_contrastive)
                else math.inf
            )
        if args.lambda_hard_negative > 0.0 and hard_val_loader is not None:
            checkpoint_metric += (
                args.lambda_hard_negative * val_metrics["hard_negative"]
            )

        print(
            f"epoch {epoch:04d} | "
            f"train_total={train_metrics['total']:.6f} "
            f"train_dsm={train_metrics['dsm']:.6f} "
            f"train_action_dsm={train_metrics['action_dsm']:.6f} "
            f"train_contrastive={train_metrics['contrastive']:.6f} | "
            f"train_hard={train_metrics['hard_negative']:.6f} "
            f"train_hard_acc={train_metrics['hard_margin_accuracy']:.3f} | "
            f"val_dsm={val_metrics['dsm']:.6f} "
            f"val_action_dsm={val_metrics['action_dsm']:.6f} "
            f"val_contrastive={val_metrics['contrastive']:.6f} "
            f"val_hard={val_metrics['hard_negative']:.6f} "
            f"val_hard_acc={val_metrics['hard_margin_accuracy']:.3f}"
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

    if dataset.histories.shape[1] < 1:
        raise ValueError(
            "Expected histories with a positive history length, got "
            f"{tuple(dataset.histories.shape)}"
        )

    if dataset.histories.shape[2:] != dataset.next_latents.shape[1:]:
        raise ValueError(
            "History latent shape must match next-latent shape: "
            f"histories={tuple(dataset.histories.shape)}, "
            f"next_latents={tuple(dataset.next_latents.shape)}"
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
            raise ValueError(f"{name} contains NaN or Inf in the first 128 samples.")


def validate_hard_negative_dataset(
    hard_dataset: HardNegativeDataset,
    positive_dataset: FeasibilityDataset,
) -> None:
    tensors = hard_dataset.tensors
    expected_history_shape = tuple(positive_dataset.histories.shape[1:])
    expected_action_shape = tuple(positive_dataset.actions.shape[1:])
    expected_next_shape = tuple(positive_dataset.next_latents.shape[1:])
    expected = {
        "histories": expected_history_shape,
        "positive_actions": expected_action_shape,
        "negative_actions": expected_action_shape,
        "positive_next_latents": expected_next_shape,
        "negative_next_latents": expected_next_shape,
    }
    for key, expected_shape in expected.items():
        actual_shape = tuple(tensors[key].shape[1:])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Hard-negative {key} shape {actual_shape} != expected {expected_shape}"
            )
        if not torch.isfinite(tensors[key][:128]).all():
            raise ValueError(f"Hard-negative {key} contains NaN or Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the transformer feasibility model."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--hard-negative-dataset",
        default=None,
        help="Paired, normalized planner-negative artifact used for training.",
    )
    parser.add_argument(
        "--hard-negative-val-dataset",
        default=None,
        help="Episode-disjoint paired artifact used for validation.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/feasibility.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional existing feasibility checkpoint to fine-tune.",
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
        "--lambda-latent-dsm",
        type=float,
        default=1.0,
        help="Weight of the existing next-latent denoising objective.",
    )
    parser.add_argument(
        "--lambda-action-dsm",
        type=float,
        default=0.0,
        help="Weight of positive-only action denoising conditioned on the transition.",
    )
    parser.add_argument(
        "--delta-target-mode",
        choices=("current_delta", "wm_residual"),
        default="current_delta",
    )

    parser.add_argument("--lambda-contrastive", type=float, default=0.0)
    parser.add_argument("--lambda-hard-negative", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.1)
    parser.add_argument("--hard-negative-temperature", type=float, default=0.1)
    parser.add_argument("--hard-negative-noise-level", type=float, default=0.0)
    parser.add_argument("--hard-negative-batch-size", type=int, default=0)
    parser.add_argument(
        "--hard-negative-val-batches",
        type=int,
        default=0,
        help="Maximum validation batches; 0 evaluates the complete artifact.",
    )
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-noise-level", type=float, default=0.2)
    parser.add_argument("--contrastive-hard-weight", type=float, default=0.5)
    parser.add_argument(
        "--contrastive-calibration-margin",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--contrastive-calibration-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--contrastive-num-action-shuffles",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--contrastive-latent-noise-scales",
        type=float,
        nargs="*",
        default=(0.05, 0.1),
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

    if args.lambda_latent_dsm < 0.0 or args.lambda_action_dsm < 0.0:
        raise ValueError("DSM loss weights must be non-negative.")
    if args.sigma_min <= 0.0 or args.sigma_max <= args.sigma_min:
        raise ValueError("Require 0 < --sigma-min < --sigma-max.")

    if not 0.0 < args.contrastive_fraction <= 1.0:
        raise ValueError("--contrastive-fraction must be in (0, 1].")

    if args.contrastive_every < 1:
        raise ValueError("--contrastive-every must be at least 1.")

    if args.contrastive_val_every < 1:
        raise ValueError("--contrastive-val-every must be at least 1.")

    if args.contrastive_temperature <= 0.0:
        raise ValueError("--contrastive-temperature must be positive.")

    if args.lambda_hard_negative < 0.0:
        raise ValueError("--lambda-hard-negative must be non-negative.")
    if args.hard_negative_temperature <= 0.0:
        raise ValueError("--hard-negative-temperature must be positive.")
    if args.hard_negative_batch_size < 0:
        raise ValueError("--hard-negative-batch-size must be non-negative.")
    if args.lambda_hard_negative > 0.0 and not args.hard_negative_dataset:
        raise ValueError("--lambda-hard-negative requires --hard-negative-dataset.")

    if args.contrastive_hard_weight < 0.0:
        raise ValueError("--contrastive-hard-weight must be non-negative.")

    if args.contrastive_calibration_weight < 0.0:
        raise ValueError("--contrastive-calibration-weight must be non-negative.")

    if args.contrastive_num_action_shuffles < 1:
        raise ValueError("--contrastive-num-action-shuffles must be at least 1.")

    if any(scale <= 0.0 for scale in args.contrastive_latent_noise_scales):
        raise ValueError("--contrastive-latent-noise-scales must all be positive.")

    start = time.perf_counter()

    dataset, metadata = load_tensor_dataset(args.dataset)
    validate_dataset(dataset)

    hard_negative_dataset = None
    hard_negative_val_dataset = None
    if args.hard_negative_dataset:
        hard_negative_dataset, hard_metadata = load_hard_negative_dataset(
            args.hard_negative_dataset
        )
        validate_hard_negative_dataset(hard_negative_dataset, dataset)
        print(
            "Hard-negative training samples:",
            len(hard_negative_dataset),
            "metadata:",
            hard_metadata,
        )
    if args.hard_negative_val_dataset:
        hard_negative_val_dataset, hard_val_metadata = load_hard_negative_dataset(
            args.hard_negative_val_dataset
        )
        validate_hard_negative_dataset(hard_negative_val_dataset, dataset)
        print(
            "Hard-negative validation samples:",
            len(hard_negative_val_dataset),
            "metadata:",
            hard_val_metadata,
        )

    print("Dataset samples:", len(dataset))
    print("Histories:", tuple(dataset.histories.shape))
    print("Actions:", tuple(dataset.actions.shape))
    print("Next latents:", tuple(dataset.next_latents.shape))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    world_model = None

    if args.delta_target_mode == "wm_residual":
        from scripts.pusht.build_pusht import load_world_model

        model_cfg = metadata.get("model_cfg")
        model_ckpt = metadata.get("model_ckpt")

        if model_cfg is None or model_ckpt is None:
            raise ValueError("wm_residual requires model_cfg and model_ckpt metadata.")

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
        hard_negative_dataset=hard_negative_dataset,
        hard_negative_val_dataset=hard_negative_val_dataset,
    )

    print(f"Total runtime: " f"{(time.perf_counter() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
