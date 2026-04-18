"""
This module implements the FeasibilityModel and ResidualFeasibilityModel for
learning soft feasibility penalties in the GVP-WM framework.

FeasibilityModel learns epsilon_theta(z_{t+1}^k, h_t, a_t, k), which is used
to compute a soft feasibility penalty P_theta = ||epsilon_theta(.)||^2 which
replaces hard constraints enforced by ALM with a learned energy term.

ResidualFeasibilityModel is a diagnostic control trained that learns by
training on residuals delta_t = z_{t+1} - f_psi(h_t, a_t).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeasibilityModel(nn.Module):
    """
    Learns epsilon_theta(z_{t+1}^k, h_t, a_t, k) for the conditional density
    p(z_{t+1} | h_t, a_t).

    At test time, the soft feasibility penalty is

        P_theta = ||epsilon_theta(.)||^2

    which can replace a hard ALM constraint with a learned energy term.

    Terminology:
        history: the last history_length H latents, h_t = z_{t-H:t}
        action: the current action a_t
        z_next: the next latent z_{t+1}
        z_noisy: the noisy next latent z_{t+1}^k
        noise_level: the noise level k that was added to z_next to get z_noisy

    Dimensions:
        history:     (..., history_length, latent_dim)
        action:      (..., action_dim)
        z_next:      (..., latent_dim)
        z_noisy:     (..., latent_dim)
        noise_level: (...) or (..., 1)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        history_length: int,
        hidden_dim: int,
        num_layers: int,
        use_layer_norm: bool = False,
    ) -> None:
        """Initializes the FeasibilityModel. Ensure that latent_dim, action_dim,
        and history_length match those of the world model and planner."""
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

        input_dim = history_length * latent_dim + action_dim + latent_dim + 1

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.latent_dim))
        self.net = nn.Sequential(*layers)

    def _flatten_history(self, history: torch.Tensor) -> torch.Tensor:
        """Flattens the history of latents into a single feature vector for each sample."""
        expected = (self.history_length, self.latent_dim)
        if history.shape[-2:] != expected:
            raise ValueError("Expected history to end with " f"{expected}, got {tuple(history.shape[-2:])}.")
        return history.reshape(*history.shape[:-2], self.history_length * self.latent_dim)

    def _prepare_noise_level(
        self,
        noise_level: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Ensures noise_level has the right shape and dtype for concatenation with other features."""
        if noise_level.ndim == reference.ndim - 1:
            noise_level = noise_level.unsqueeze(-1)
        elif noise_level.ndim != reference.ndim:
            raise ValueError("noise_level must have shape (...,) or (..., 1) matching the batch dims.")
        return noise_level.to(dtype=reference.dtype, device=reference.device)

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_noisy: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the noise epsilon_theta for a batch of samples.

        Args:
            history(torch.Tensor): history of latents.
            action(torch.Tensor): current action.
            z_noisy(torch.Tensor): noisy next latent z_{t+1}^k
            noise_level(torch.Tensor): noise level for each sample.

        Returns:
            torch.Tensor: predicted noise epsilon_theta(z_{t+1}^k, h_t, a_t, k) for each sample in batch.
        """
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        if z_noisy.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z_noisy.shape[-1]}.")

        history_features = self._flatten_history(history)
        noise_features = self._prepare_noise_level(noise_level, z_noisy)
        features = torch.cat([history_features, action, z_noisy, noise_features], dim=-1)
        return self.net(features)

    def score(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_noisy: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the score function s_theta = -epsilon_theta / sigma for a batch of samples.
        Used for Langevin sampling at test time.

        Low penalty -> network predicts near-zero noise, feasible.
        high penalty -> network predicts large noise, infeasible.

        Args:
            history(torch.Tensor): history of latents.
            action(torch.Tensor): current action.
            z_noisy(torch.Tensor): noisy next latent z_{t+1}^k
            noise_level(torch.Tensor): noise level for each sample.

        Returns:
            torch.Tensor: scores s_theta(z_{t+1}^k, h_t, a_t, k) for each sample in batch.
        """
        predicted_noise = self.forward(history, action, z_noisy, noise_level)
        sigma = self._prepare_noise_level(noise_level, z_noisy).clamp_min(1e-6)
        return -predicted_noise / sigma

    def penalty(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Learned Soft Feasibility Penalty P_theta = ||epsilon_theta(.)||^2.
        Replaces hard ALM constraints with a learned energy term.

        Low penalty -> network predicts near-zero noise, feasible.
        high penalty -> network predicts large noise, infeasible.

        Args:
            history(torch.Tensor): history of latents.
            action(torch.Tensor): current action.
            z_noisy(torch.Tensor): noisy next latent z_{t+1}^k
            noise_level(torch.Tensor): noise level for each sample.
            reduction(str): how to reduce the loss in the batch dimension.

        Returns:
            torch.Tensor: feasibility penalty
        """
        predicted_noise = self.forward(history, action, z_noisy, noise_level)
        penalty = predicted_noise.pow(2).sum(dim=-1)
        if reduction == "none":
            return penalty
        if reduction == "mean":
            return penalty.mean()
        if reduction == "sum":
            return penalty.sum()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def sample_noisy_target(
        self,
        z_next: torch.Tensor,
        noise_level: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a (batch of) next latent(s) z_{t+1}, sample noise and add it
        to z_next according to the noise level.

        Args:
            z_next(torch.Tensor): next latent z_{t+1}.
            noise_level(torch.Tensor): noise level for each sample.
            noise(torch.Tensor | None): optional noise to add to z_next, random if None.

        Returns:
            torch.Tensor: noisy next latent z_{t+1}^k
            torch.Tensor: the true noise that was added
        """
        sigma = self._prepare_noise_level(noise_level, z_next)
        if noise is None:
            noise = torch.randn_like(z_next)
        z_noisy = z_next + sigma * noise
        return z_noisy, noise

    def dsm_loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        noise_level: torch.Tensor,
        noise: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the denoising score matching loss for a batch of samples.

        Args:
            history(torch.Tensor): history of latents.
            action(torch.Tensor): current action.
            z_next(torch.Tensor): next latent z_{t+1}.
            noise_level(torch.Tensor): noise level for each sample.
            noise(torch.Tensor | None): optional noise to add to z_next, random if None.
            reduction(str): how to reduce the loss in the batch dimension.

        Returns:
            torch.Tensor: denoising score matchin loss.
        """
        z_noisy, true_noise = self.sample_noisy_target(z_next, noise_level, noise=noise)
        predicted_noise = self.forward(history, action, z_noisy, noise_level)
        loss = (predicted_noise - true_noise).pow(2).mean(dim=-1)
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction}")


class ResidualFeasibilityModel(FeasibilityModel):
    """
    Diagnostic control trained on residuals delta_t = z_{t+1} - f_psi(h_t, a_t).

    This does not replace the main model. It is useful for comparing energy
    landscapes and measuring whether the pretrained dynamics model drifts away
    from the feasible manifold over a planning horizon.

    Intuition:
        The residual model's penalty tells you how surprised it is by f_psi's
        error at each step. So if you run it along a planning horizon:
            - Early steps: f_psi is likely accurate, delta_t is small -> low penalty
            - Later steps: if f_psi drifts, delta_t grows -> penalty spikes

        This lets you quantify exactly where and how much f_psi leaves the
        feasible manifold over the horizon.
    """

    def residual_target(
        self,
        predicted_next_latent: torch.Tensor,
        true_next_latent: torch.Tensor,
    ) -> torch.Tensor:
        return true_next_latent - predicted_next_latent

    def residual_dsm_loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        predicted_next_latent: torch.Tensor,
        true_next_latent: torch.Tensor,
        noise_level: torch.Tensor,
        noise: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        residual = self.residual_target(predicted_next_latent, true_next_latent)
        return self.dsm_loss(
            history=history,
            action=action,
            z_next=residual,
            noise_level=noise_level,
            noise=noise,
            reduction=reduction,
        )
