from __future__ import annotations

import math
from pathlib import Path

import torch

from codex.training.config import TrainingConfig
from codex.training.losses import calibrated_energy_loss, multi_action_ranking_loss
from codex.training.mining import (
    global_action_negatives,
    local_action_negatives,
    mine_adversarial_actions,
)
from codex.training.validation import action_recovery_batch
from extension.feasibility2.dataset import FeasibilityDataset
from extension.feasibility2.model import TransformerFeasibilityModel
from extension.feasibility2.scheduler import LogUniformSigmaScheduler
from extension.feasibility2.train import amp_context, make_loader, move_batch


class PlannerAlignedTrainer:
    """Train the existing feasibility transformer for action minimization.

    Model construction, DSM loss, log-uniform sigma scheduling, DataLoader
    construction, batch transfer, and AMP context are reused directly from
    ``extension.feasibility2``. The new logic is limited to fixed-transition
    action negatives, online adversarial mining, and recovery-based validation.
    """

    def __init__(
        self,
        train_dataset: FeasibilityDataset,
        val_dataset: FeasibilityDataset,
        metadata: dict,
        val_metadata: dict,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_out: str | Path,
        init_checkpoint: str | Path | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.metadata = dict(metadata)
        self.val_metadata = dict(val_metadata)
        self.config = config
        self.device = device
        self.checkpoint_out = Path(checkpoint_out)
        self._validate_data_contract()
        self.low = train_dataset.actions.amin(dim=0).to(device)
        self.high = train_dataset.actions.amax(dim=0).to(device)
        if not torch.all(self.low < self.high):
            raise ValueError("At least one action coordinate has no training range")
        if init_checkpoint is None:
            self.model = TransformerFeasibilityModel(
                action_dim=int(train_dataset.actions.shape[-1]),
                latent_shape=tuple(train_dataset.next_latents.shape[1:]),
                history_length=int(train_dataset.histories.shape[1]),
                model_dim=config.model_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
            ).to(device)
        else:
            self.model = TransformerFeasibilityModel.from_checkpoint(
                init_checkpoint, map_location=device
            ).to(device)
        self.scheduler = LogUniformSigmaScheduler(0.05, 0.5)

    def fit(self) -> TransformerFeasibilityModel:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        train_loader = make_loader(
            self.train_dataset, cfg.batch_size, True, cfg.num_workers,
            self.device.type == "cuda",
        )
        val_loader = make_loader(
            self.val_dataset, cfg.batch_size, False, cfg.num_workers,
            self.device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        best_metric = math.inf
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            sums = {
                "total": 0.0, "dsm": 0.0, "global": 0.0,
                "local": 0.0, "adversarial": 0.0,
            }
            count = 0
            warmup = epoch <= cfg.dsm_warmup_epochs
            for batch_index, batch in enumerate(train_loader):
                history, action, z_next, _ = move_batch(batch, self.device, False)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(self.device, True):
                    dsm = self.model.dsm_loss(
                        history, action, z_next,
                        scheduler=self.scheduler, reduction="mean",
                    )
                local = history.new_zeros(())
                global_ranking = history.new_zeros(())
                adversarial = history.new_zeros(())
                calibration = history.new_zeros(())
                if not warmup:
                    global_actions = global_action_negatives(
                        action, self.low, self.high, cfg.num_action_shuffles
                    )
                    local_actions = local_action_negatives(
                        action, self.low, self.high, cfg.local_noise_scales
                    )
                    with amp_context(self.device, True):
                        global_ranking, _ = multi_action_ranking_loss(
                            self.model, history, action, z_next, global_actions,
                            noise_level=cfg.contrastive_noise_level,
                            margin=cfg.ranking_margin,
                            temperature=cfg.temperature,
                        )
                        local, _ = multi_action_ranking_loss(
                            self.model, history, action, z_next, local_actions,
                            noise_level=cfg.contrastive_noise_level,
                            # Nearby actions can be alternative feasible controls;
                            # require ordering but only a small local margin.
                            margin=cfg.local_ranking_margin,
                            temperature=cfg.temperature,
                        )
                    if (
                        (cfg.lambda_adversarial > 0.0 or cfg.lambda_calibration > 0.0)
                        and batch_index % cfg.adversarial.every_batches == 0
                    ):
                        adversarial_actions = mine_adversarial_actions(
                            self.model, history, action, z_next, self.low, self.high,
                            noise_level=cfg.contrastive_noise_level,
                            starts=cfg.adversarial.starts,
                            steps=cfg.adversarial.steps,
                            learning_rate=cfg.adversarial.learning_rate,
                            minimum_rms_distance=cfg.adversarial.minimum_rms_distance,
                            distance_penalty=cfg.adversarial.distance_penalty,
                        )
                        with amp_context(self.device, True):
                            adversarial, _ = multi_action_ranking_loss(
                                self.model, history, action, z_next,
                                adversarial_actions.unsqueeze(1),
                                noise_level=cfg.contrastive_noise_level,
                                margin=cfg.ranking_margin,
                                temperature=cfg.temperature,
                            )
                            positive_energy = self.model.energy(
                                history, action, z_next,
                                cfg.contrastive_noise_level, "none",
                            )
                            negative_energy = self.model.energy(
                                history, adversarial_actions, z_next,
                                cfg.contrastive_noise_level, "none",
                            )
                            calibration = calibrated_energy_loss(
                                positive_energy, negative_energy
                            )
                adversarial_frequency_correction = (
                    cfg.adversarial.every_batches
                    if (
                        not warmup
                        and (cfg.lambda_adversarial > 0.0 or cfg.lambda_calibration > 0.0)
                        and batch_index % cfg.adversarial.every_batches == 0
                    )
                    else 0.0
                )
                loss = (
                    dsm if warmup else
                    cfg.lambda_dsm * dsm
                    + cfg.lambda_global * global_ranking
                    + cfg.lambda_local * local
                    + cfg.lambda_adversarial * adversarial_frequency_correction * adversarial
                    + cfg.lambda_calibration * adversarial_frequency_correction * calibration
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_norm)
                optimizer.step()
                batch_count = history.shape[0]
                count += batch_count
                for name, value in (("total", loss), ("dsm", dsm),
                                    ("global", global_ranking),
                                    ("local", local), ("adversarial", adversarial)):
                    sums[name] += float(value.detach()) * batch_count
            train_metrics = {name: value / max(count, 1) for name, value in sums.items()}
            val_metrics = self._validate(val_loader)
            selection_metric = (
                val_metrics["recovery_ratio"]
                + 2.0 * (1.0 - val_metrics["expert_beats_zero"])
                + 2.0 * (1.0 - val_metrics["expert_beats_optimized"])
                - 0.1 * val_metrics["cosine"]
            )
            print(
                f"epoch={epoch:03d} warmup={warmup} "
                f"train={train_metrics} val={val_metrics} "
                f"selection={selection_metric:.6f}"
            )
            if selection_metric < best_metric:
                best_metric = selection_metric
                self._save(epoch, train_metrics, val_metrics, best=True)
        self._save(cfg.epochs, train_metrics, val_metrics, best=False)
        return self.model

    def _validate(self, loader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        batches = 0
        # Gradients with respect to candidate actions are required even though
        # model parameters are not updated during validation.
        for batch_index, batch in enumerate(loader):
            if batch_index >= self.config.validation_batches:
                break
            history, action, z_next, _ = move_batch(batch, self.device, False)
            metrics = action_recovery_batch(
                self.model, history, action, z_next, self.low, self.high,
                noise_level=self.config.contrastive_noise_level,
                starts=self.config.validation_starts,
                steps=self.config.validation_steps,
                learning_rate=self.config.adversarial.learning_rate,
                minimum_rms_distance=0.0, distance_penalty=0.0,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
        self.model.train()
        return {name: value / max(batches, 1) for name, value in totals.items()}

    def _save(self, epoch, train_metrics, val_metrics, *, best: bool) -> None:
        path = self.checkpoint_out if best else self.checkpoint_out.with_name(
            f"{self.checkpoint_out.stem}_final{self.checkpoint_out.suffix}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["action_low"] = self.low.detach().cpu()
        metadata["action_high"] = self.high.detach().cpu()
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config_dict(),
            "dataset_metadata": metadata,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "training_setup": "codex_planner_aligned_v1",
            "lambda_delta": 0.0,
            "lambda_contrastive": self.config.lambda_local,
            "lambda_global": self.config.lambda_global,
            "lambda_hard_negative": self.config.lambda_adversarial,
            "contrastive_noise_level": self.config.contrastive_noise_level,
            "planner_selection_metric": (
                "recovery_ratio+2*(1-expert_beats_zero)+"
                "2*(1-expert_beats_optimized)-0.1*cosine"
            ),
        }, path)
        print(f"Saved {'best' if best else 'final'} checkpoint: {path}")

    def _validate_data_contract(self) -> None:
        train_shapes = (
            self.train_dataset.histories.shape[1:],
            self.train_dataset.actions.shape[1:],
            self.train_dataset.next_latents.shape[1:],
        )
        val_shapes = (
            self.val_dataset.histories.shape[1:],
            self.val_dataset.actions.shape[1:],
            self.val_dataset.next_latents.shape[1:],
        )
        if train_shapes != val_shapes:
            raise ValueError(f"Train/validation shapes differ: {train_shapes} vs {val_shapes}")
        train_range = (
            self.metadata.get("episode_start"), self.metadata.get("episode_end")
        )
        val_range = (
            self.val_metadata.get("episode_start"), self.val_metadata.get("episode_end")
        )
        if None not in (*train_range, *val_range):
            overlap = max(train_range[0], val_range[0]) < min(train_range[1], val_range[1])
            if overlap:
                raise ValueError(
                    f"Training episodes {train_range} overlap validation episodes {val_range}"
                )
        for key in ("latent_mean", "latent_std"):
            if key not in self.metadata or key not in self.val_metadata:
                raise ValueError(f"Both datasets must contain {key}")
            if not torch.allclose(
                torch.as_tensor(self.metadata[key]),
                torch.as_tensor(self.val_metadata[key]),
            ):
                raise ValueError(
                    f"Validation {key} differs from training statistics; rebuild validation "
                    "with --norm-stats pointing to the training dataset"
                )
