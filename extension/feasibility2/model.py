"""
Defines the learned feasibility model.

Trains a denoising network:
    epsilon_theta(h_t, a_t, z_{t+1}^{noisy}, sigma)

The transformer backbone is shared by two task-specific heads:
    - a token-wise DSM head that predicts noise
    - a scalar energy head used for contrastive transition ranking

The original DSM-derived penalty remains available for planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import SinusoidalSigmaEmbedding
from .scheduler import SigmaScheduler

Reduction = Literal["none", "mean", "sum"]


@dataclass
class TransformerFeasibilityConfig:
    action_dim: int
    latent_shape: tuple[int, ...]
    history_length: int = 3
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    noise_level: float = 0.05
    lambda_delta: float = 1.0


def _prod(shape: tuple[int, ...]) -> int:
    result = 1
    for x in shape:
        result *= int(x)
    return result


def _infer_token_shape(latent_shape: tuple[int, ...]) -> tuple[int, int]:
    """Infer [num_tokens, token_dim] from a latent shape."""
    if len(latent_shape) == 1:
        return 1, int(latent_shape[0])
    if len(latent_shape) == 2:
        return int(latent_shape[0]), int(latent_shape[1])
    if len(latent_shape) == 3:
        c, h, w = map(int, latent_shape)
        return h * w, c
    raise ValueError(
        f"Unsupported latent_shape={latent_shape}. Expected [D], [N, D], or [C, H, W]."
    )


class TransformerFeasibilityModel(nn.Module):
    """Transformer conditional denoising model for learned feasibility."""

    def __init__(
        self,
        action_dim: int,
        latent_shape: tuple[int, ...],
        history_length: int = 3,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by num_heads={num_heads}"
            )

        self.action_dim = int(action_dim)
        self.latent_shape = tuple(int(x) for x in latent_shape)
        self.latent_dim = _prod(self.latent_shape)
        self.history_length = int(history_length)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)

        self.num_tokens, self.token_dim = _infer_token_shape(self.latent_shape)

        self.token_proj = nn.Linear(self.token_dim, self.model_dim)
        self.action_proj = nn.Linear(self.action_dim, self.model_dim)
        self.sigma_embed = SinusoidalSigmaEmbedding(self.model_dim)

        total_tokens = 2 + (self.history_length + 1) * self.num_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.model_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        self.out_proj = nn.Linear(self.model_dim, self.token_dim)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.delta_query = nn.Parameter(torch.zeros(1, self.num_tokens, self.model_dim))
        self.delta_out_proj = nn.Linear(self.model_dim, self.token_dim)

    def _batchify(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        single = False

        expected_history_ndim = 1 + len(self.latent_shape)
        expected_batched_history_ndim = 2 + len(self.latent_shape)
        expected_batched_z_ndim = 1 + len(self.latent_shape)

        if history.ndim == expected_history_ndim:
            history = history.unsqueeze(0)
            action = action.unsqueeze(0)
            z_next_noisy = z_next_noisy.unsqueeze(0)
            single = True

        if history.ndim != expected_batched_history_ndim:
            raise ValueError(
                f"history must be [H, *latent_shape] or [B, H, *latent_shape], got {tuple(history.shape)}"
            )
        if action.ndim != 2:
            raise ValueError(
                f"action must be [B, action_dim] after batching, got {tuple(action.shape)}"
            )
        if z_next_noisy.ndim != expected_batched_z_ndim:
            raise ValueError(
                f"z_next_noisy must be [B, *latent_shape] after batching, got {tuple(z_next_noisy.shape)}"
            )
        if history.shape[1] != self.history_length:
            raise ValueError(
                f"history length mismatch: got {history.shape[1]}, expected {self.history_length}"
            )
        if tuple(history.shape[2:]) != self.latent_shape:
            raise ValueError(
                f"history latent shape mismatch: got {tuple(history.shape[2:])}, expected {self.latent_shape}"
            )
        if tuple(z_next_noisy.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"z_next_noisy latent shape mismatch: got {tuple(z_next_noisy.shape[1:])}, expected {self.latent_shape}"
            )
        if action.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action.shape[1]}, expected {self.action_dim}"
            )

        return history, action, z_next_noisy, single

    def _noise_tensor(self, noise_level, batch_size: int, device, dtype) -> torch.Tensor:
        if noise_level is None:
            raise ValueError("noise_level must be provided explicitly; model has no default sigma.")

        if not torch.is_tensor(noise_level):
            noise = torch.full(
                (batch_size, 1),
                float(noise_level),
                device=device,
                dtype=dtype,
            )
        else:
            noise = noise_level.to(device=device, dtype=dtype)
            if noise.ndim == 0:
                noise = noise.expand(batch_size).reshape(batch_size, 1)
            elif noise.ndim == 1:
                noise = noise.reshape(-1, 1)
            if noise.shape[0] == 1 and batch_size != 1:
                noise = noise.expand(batch_size, 1)

        return noise

    def _latent_to_tokens(self, z: torch.Tensor) -> torch.Tensor:
        if len(self.latent_shape) == 1:
            return z.reshape(z.shape[0], 1, self.token_dim)
        if len(self.latent_shape) == 2:
            return z
        if len(self.latent_shape) == 3:
            return z.permute(0, 2, 3, 1).reshape(z.shape[0], self.num_tokens, self.token_dim)
        raise ValueError(f"Unsupported latent_shape={self.latent_shape}")

    def _tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        if len(self.latent_shape) in (1, 2):
            return tokens.reshape(tokens.shape[0], *self.latent_shape)
        if len(self.latent_shape) == 3:
            c, h, w = self.latent_shape
            return tokens.reshape(tokens.shape[0], h, w, c).permute(0, 3, 1, 2)
        raise ValueError(f"Unsupported latent_shape={self.latent_shape}")

    def _history_to_tokens(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]
        history_flat = history.reshape(batch_size * self.history_length, *self.latent_shape)
        tokens = self._latent_to_tokens(history_flat)
        return tokens.reshape(
            batch_size,
            self.history_length * self.num_tokens,
            self.token_dim,
        )

    def _encode_next_features(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level: torch.Tensor | float,
    ) -> tuple[torch.Tensor, bool]:
        """Encode a candidate next latent with the shared transformer."""
        history, action, z_next, single = self._batchify(
            history,
            action,
            z_next,
        )
        batch_size = history.shape[0]

        sigma = self._noise_tensor(
            noise_level,
            batch_size=batch_size,
            device=history.device,
            dtype=history.dtype,
        )

        history_tokens = self.token_proj(self._history_to_tokens(history))
        next_tokens = self.token_proj(self._latent_to_tokens(z_next))

        action_token = self.action_proj(action).unsqueeze(1)
        sigma_token = self.sigma_embed(sigma).unsqueeze(1)

        x = torch.cat(
            [action_token, sigma_token, history_tokens, next_tokens],
            dim=1,
        )
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer(x)

        next_start = 2 + self.history_length * self.num_tokens
        next_end = next_start + self.num_tokens

        return x[:, next_start:next_end], single

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next_noisy: torch.Tensor,
        noise_level: torch.Tensor | float,
    ) -> torch.Tensor:
        next_features, single = self._encode_next_features(
            history,
            action,
            z_next_noisy,
            noise_level,
        )
        pred_tokens = self.out_proj(next_features)
        pred = self._tokens_to_latent(pred_tokens)

        if single:
            pred = pred.squeeze(0)
        return pred

    def energy(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level: torch.Tensor | float,
        reduction: Reduction = "none",
    ) -> torch.Tensor:
        """Predict one scalar contrastive energy per transition."""
        next_features, single = self._encode_next_features(
            history,
            action,
            z_next,
            noise_level,
        )
        energy = self.energy_head(next_features.mean(dim=1)).squeeze(-1)

        if single:
            energy = energy.squeeze(0)

        if reduction == "none":
            return energy
        if reduction == "sum":
            return energy.sum()
        if reduction == "mean":
            return energy.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def dsm_loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        scheduler: SigmaScheduler | None = None,
        noise_level=None,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        history_b, action_b, z_b, _ = self._batchify(history, action, z_next)

        if scheduler is not None:
            sigma = scheduler.sample_sigmas(z_b.shape[0], z_b.device, z_b.dtype)
            z_noisy, eps = scheduler.add_noise(z_b, sigma)
            weight = scheduler.loss_weight(sigma).reshape(-1)
        else:
            sigma = self._noise_tensor(noise_level, z_b.shape[0], z_b.device, z_b.dtype)
            eps = torch.randn_like(z_b)
            z_noisy = z_b + sigma.reshape(-1, *([1] * (z_b.ndim - 1))) * eps
            weight = torch.ones(z_b.shape[0], device=z_b.device, dtype=z_b.dtype)

        pred_eps = self.forward(history_b, action_b, z_noisy, sigma)

        loss_per_item = F.mse_loss(pred_eps, eps, reduction="none")
        loss_per_item = loss_per_item.reshape(loss_per_item.shape[0], -1).mean(dim=-1)
        loss_per_item = loss_per_item * weight

        if reduction == "none":
            return loss_per_item
        if reduction == "sum":
            return loss_per_item.sum()
        if reduction == "mean":
            return loss_per_item.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def penalty(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level=None,
        reduction: Reduction = "none",
    ) -> torch.Tensor:
        history_b, action_b, z_b, single = self._batchify(history, action, z_next)

        batch_size = z_b.shape[0]
        sigma = self._noise_tensor(noise_level, batch_size, z_b.device, z_b.dtype)
        sigma_expanded = sigma.reshape(-1, *([1] * (z_b.ndim - 1)))

        eps = torch.randn_like(z_b)
        z_noisy = z_b + sigma_expanded * eps
        pred_eps = self.forward(history_b, action_b, z_noisy, sigma)

        energy = pred_eps.reshape(pred_eps.shape[0], -1).pow(2).mean(dim=-1)

        if single:
            energy = energy.squeeze(0)

        if reduction == "none":
            return energy
        if reduction == "sum":
            return energy.sum()
        if reduction == "mean":
            return energy.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    @torch.no_grad()
    def energy_no_grad(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level=None,
        reduction: Reduction = "none",
    ) -> torch.Tensor:
        return self.penalty(history, action, z_next, noise_level, reduction)

    def config_dict(self) -> dict:
        return {
            "architecture": "transformer",
            "action_dim": self.action_dim,
            "latent_shape": self.latent_shape,
            "latent_dim": self.latent_dim,
            "history_length": self.history_length,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        }

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        cfg = dict(checkpoint["config"])
        cfg.pop("architecture", None)
        cfg.pop("latent_dim", None)
        model = cls(**cfg)
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("energy_head.")
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible transformer checkpoint: "
                f"missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model.eval()
        return model

    def predict_delta(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized latent delta z_next - z_current."""
        if history.ndim == 1 + len(self.latent_shape):
            dummy_z = history[-1].clone()
            action_in = action
        else:
            dummy_z = history[:, -1].clone()
            action_in = action

        history, action_in, dummy_z, single = self._batchify(history, action_in, dummy_z)
        batch_size = history.shape[0]

        history_tokens = self.token_proj(self._history_to_tokens(history))
        action_token = self.action_proj(action_in).unsqueeze(1)

        sigma = torch.zeros(batch_size, 1, device=history.device, dtype=history.dtype)
        sigma_token = self.sigma_embed(sigma).unsqueeze(1)
        delta_tokens = self.delta_query.expand(batch_size, -1, -1)

        x = torch.cat(
            [action_token, sigma_token, history_tokens, delta_tokens],
            dim=1,
        )
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer(x)

        delta_start = 2 + self.history_length * self.num_tokens
        delta_end = delta_start + self.num_tokens

        delta_out = x[:, delta_start:delta_end]
        pred_tokens = self.delta_out_proj(delta_out)
        pred_delta = self._tokens_to_latent(pred_tokens)

        if single:
            pred_delta = pred_delta.squeeze(0)
        return pred_delta

    def delta_loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        reduction: Reduction = "mean",
        target: str = "current_delta",
        world_model=None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        past_action_history = None,
    ) -> torch.Tensor:
        """Train transition head to predict a normalized latent residual."""
        history_b, action_b, z_b, _ = self._batchify(history, action, z_next)

        if target == "current_delta":
            z_pred = history_b[:, -1]
        elif target == "wm_residual":
            if world_model is None:
                raise ValueError("World model is required for wm_residual target")
            if latent_mean is None or latent_std is None:
                raise ValueError(
                    "latent_mean and latent_std are required for target='wm_residual' because DINO-WM expects raw latents."
                )
            if past_action_history is None:
                raise ValueError("past_action_history is required for wm_residual target")

            past_action_history = past_action_history.to(
                device=history_b.device,
                dtype=history_b.dtype,
            )
            latent_mean = latent_mean.to(device=history_b.device, dtype=history_b.dtype)
            latent_std = latent_std.to(device=history_b.device, dtype=history_b.dtype)

            latent_shape = tuple(history_b.shape[2:])
            mean_hist = latent_mean.view(1, 1, *latent_shape)
            std_hist = latent_std.view(1, 1, *latent_shape)
            mean_next = latent_mean.view(1, *latent_shape)
            std_next = latent_std.view(1, *latent_shape)

            history_raw = history_b * std_hist + mean_hist
            preds_raw = []
            with torch.no_grad():
                for i in range(history_raw.shape[0]):
                    z_pred_i = world_model.predict_next_latent(
                        history_raw[i],
                        past_action_history[i],
                    )
                    preds_raw.append(z_pred_i)

            z_pred_raw = torch.stack(preds_raw, dim=0)
            z_pred = (z_pred_raw - mean_next) / std_next
        else:
            raise ValueError(f"Unknown target: {target}")

        target_delta = z_b - z_pred
        pred_delta = self.predict_delta(history_b, action_b)

        loss_per_item = F.mse_loss(pred_delta, target_delta, reduction="none")
        loss_per_item = loss_per_item.reshape(loss_per_item.shape[0], -1).mean(dim=-1)

        if reduction == "none":
            return loss_per_item
        if reduction == "sum":
            return loss_per_item.sum()
        if reduction == "mean":
            return loss_per_item.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def transition_penalty(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        reduction: Reduction = "mean",
        target: str = "current_delta",
        world_model=None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        past_action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Planning-time learned transition consistency penalty."""
        history_b, action_b, z_b, single = self._batchify(history, action, z_next)

        if target == "current_delta":
            z_pred = history_b[:, -1]
        elif target == "wm_residual":
            if world_model is None:
                raise ValueError("World model is required for wm_residual target")
            if latent_mean is None or latent_std is None:
                raise ValueError(
                    "latent_mean and latent_std are required for target='wm_residual' because DINO-WM expects raw latents."
                )
            if past_action_history is None:
                raise ValueError("past_action_history is required for wm_residual target")

            past_action_history = past_action_history.to(
                device=history_b.device,
                dtype=history_b.dtype,
            )
            latent_mean = latent_mean.to(device=history_b.device, dtype=history_b.dtype)
            latent_std = latent_std.to(device=history_b.device, dtype=history_b.dtype)

            latent_shape = tuple(history_b.shape[2:])
            mean_hist = latent_mean.view(1, 1, *latent_shape)
            std_hist = latent_std.view(1, 1, *latent_shape)
            mean_next = latent_mean.view(1, *latent_shape)
            std_next = latent_std.view(1, *latent_shape)

            history_raw = history_b * std_hist + mean_hist
            preds_raw = []
            with torch.no_grad():
                for i in range(history_raw.shape[0]):
                    z_pred_i = world_model.predict_next_latent(
                        history_raw[i],
                        past_action_history[i],
                    )
                    preds_raw.append(z_pred_i)

            z_pred_raw = torch.stack(preds_raw, dim=0)
            z_pred = (z_pred_raw - mean_next) / std_next
        else:
            raise ValueError(f"Unknown target: {target}")

        target_delta = z_b - z_pred
        pred_delta = self.predict_delta(history_b, action_b)

        loss_per_item = F.mse_loss(pred_delta, target_delta, reduction="none")
        loss_per_item = loss_per_item.reshape(loss_per_item.shape[0], -1).mean(dim=-1)

        if single:
            loss_per_item = loss_per_item.squeeze(0)

        if reduction == "none":
            return loss_per_item
        if reduction == "sum":
            return loss_per_item.sum()
        if reduction == "mean":
            return loss_per_item.mean()
        raise ValueError(f"Unknown reduction: {reduction}")


def load_feasibility_model_from_checkpoint(path, map_location="cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=map_location)
    cfg = dict(checkpoint["config"])

    architecture = cfg.get("architecture", "mlp")

    if architecture == "mlp":
        cfg.pop("architecture", None)
        cfg.pop("history_length", None)
        model = FeasibilityModel(**cfg)
    elif architecture == "transformer":
        cfg.pop("architecture", None)
        cfg.pop("latent_dim", None)
        model = TransformerFeasibilityModel(**cfg)
    else:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")

    if architecture == "transformer":
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("energy_head.")
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible transformer checkpoint: "
                f"missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@dataclass
class FeasibilityConfig:
    history_dim: int
    action_dim: int
    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    noise_level: float = 0.05


class FeasibilityModel(nn.Module):
    """Conditional denoising model for learned feasibility."""

    def __init__(
        self,
        history_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        noise_level: float = 0.05,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.history_dim = int(history_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.noise_level = float(noise_level)

        if self.history_dim % self.latent_dim != 0:
            raise ValueError(
                f"history_dim must be divisible by latent_dim, got history_dim={self.history_dim}, latent_dim={self.latent_dim}"
            )

        self.history_length = self.history_dim // self.latent_dim

        input_dim = self.history_dim + self.action_dim + self.latent_dim + 1
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, self.latent_dim))
        self.net = nn.Sequential(*layers)


    def _batchify(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        single = False

        if history.ndim == 2:
            history = history.unsqueeze(0)
            action = action.unsqueeze(0)
            z_next_noisy = z_next_noisy.unsqueeze(0)
            single = True

        if history.ndim != 3:
            raise ValueError(
                f"history must be [H, latent_dim] or [B, H, latent_dim], got {tuple(history.shape)}"
            )
        if action.ndim != 2:
            raise ValueError(
                f"action must be [B, action_dim] after batching, got {tuple(action.shape)}"
            )
        if z_next_noisy.ndim != 2:
            raise ValueError(
                f"z_next_noisy must be [B, latent_dim] after batching, got {tuple(z_next_noisy.shape)}"
            )
        if history.shape[1] != self.history_length:
            raise ValueError(
                f"history_length mismatch: got {history.shape[1]}, expected {self.history_length}"
            )
        if history.shape[2] != self.latent_dim:
            raise ValueError(
                f"history latent_dim mismatch: got {history.shape[2]}, expected {self.latent_dim}"
            )
        if action.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action.shape[1]}, expected {self.action_dim}"
            )
        if z_next_noisy.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_next_noisy dim mismatch: got {z_next_noisy.shape[1]}, expected {self.latent_dim}"
            )

        return history, action, z_next_noisy, single

    def _noise_tensor(self, noise_level, batch_size: int, device, dtype) -> torch.Tensor:
        if noise_level is None:
            noise_level = self.noise_level
        if not torch.is_tensor(noise_level):
            noise = torch.full((batch_size, 1), float(noise_level), device=device, dtype=dtype)
        else:
            noise = noise_level.to(device=device, dtype=dtype)
            if noise.ndim == 0:
                noise = noise.expand(batch_size).reshape(batch_size, 1)
            elif noise.ndim == 1:
                noise = noise.reshape(-1, 1)
            if noise.shape[0] == 1 and batch_size != 1:
                noise = noise.expand(batch_size, 1)
        return noise

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next_noisy: torch.Tensor,
        noise_level: torch.Tensor | float,
    ) -> torch.Tensor:
        history, action, z_next_noisy, single = self._batchify(history, action, z_next_noisy)
        batch_size = history.shape[0]

        history_flat = history.reshape(batch_size, -1)
        sigma = self._noise_tensor(noise_level, batch_size, history.device, history.dtype)

        x = torch.cat([history_flat, action, z_next_noisy, sigma], dim=-1)
        out = self.net(x)

        if single:
            out = out.squeeze(0)
        return out

    def dsm_loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level=None,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        history_b, action_b, z_b, _ = self._batchify(history, action, z_next)
        sigma = self._noise_tensor(noise_level, z_b.shape[0], z_b.device, z_b.dtype)
        eps = torch.randn_like(z_b)
        z_noisy = z_b + sigma * eps
        pred_eps = self.forward(history_b, action_b, z_noisy, sigma)
        loss_per_item = F.mse_loss(pred_eps, eps, reduction="none").mean(dim=-1)

        if reduction == "none":
            return loss_per_item
        if reduction == "sum":
            return loss_per_item.sum()
        if reduction == "mean":
            return loss_per_item.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def penalty(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level=None,
        reduction: Reduction = "none",
    ) -> torch.Tensor:
        pred_eps = self.forward(history, action, z_next, noise_level)
        if pred_eps.ndim == 1:
            energy = pred_eps.pow(2).sum().unsqueeze(0)
        else:
            energy = pred_eps.pow(2).sum(dim=-1)

        if reduction == "none":
            return energy
        if reduction == "sum":
            return energy.sum()
        if reduction == "mean":
            return energy.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    @torch.no_grad()
    def energy_no_grad(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level=None,
        reduction: Reduction = "none",
    ) -> torch.Tensor:
        return self.penalty(history, action, z_next, noise_level, reduction)

    def config_dict(self) -> dict:
        return {
            "architecture": "mlp",
            "history_dim": self.history_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "noise_level": self.noise_level,
            "history_length": self.history_length,
        }

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        cfg = dict(checkpoint["config"])
        cfg.pop("history_length", None)
        cfg.pop("architecture", None)
        model = cls(**cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
