from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class FeasibilityScorer(nn.Module):
    """Planning-only wrapper with an explicit normalization contract.

    The checkpoint loading is reused from
    ``local.feasibility2.integrate.load_feasibility_scorer``. Unlike the old
    integration layer, every public scoring method here requires actions that
    are already normalized in the DINO-WM convention.
    """

    def __init__(self, model: nn.Module, checkpoint_path: str | Path) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not hasattr(model, "latent_mean") or not hasattr(model, "latent_std"):
            raise ValueError("Checkpoint must contain latent normalization statistics")
        if not hasattr(model, "energy"):
            raise ValueError("Checkpoint model has no contrastive energy head")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device,
    ) -> "FeasibilityScorer":
        # Lazy import keeps planning utilities usable off-cluster; the reused
        # loader itself depends on the external DINO-WM ``datasets`` package.
        from local.feasibility2.integrate import load_feasibility_scorer

        path = Path(checkpoint_path).expanduser().resolve()
        model = load_feasibility_scorer(path, device=device, freeze=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", {})
        if not any(key.startswith("energy_head.") for key in state):
            raise ValueError("Checkpoint does not contain trained energy-head weights")
        return cls(model, path)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.model.latent_mean.to(latents)
        std = self.model.latent_std.to(latents).clamp_min(1e-6)
        return (latents - mean) / std

    def dsm_energy(
        self,
        history_raw: torch.Tensor,
        action_normalized: torch.Tensor,
        next_latent_raw: torch.Tensor,
        *,
        noise_level: float,
        mode: str = "paper_clean",
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history = self.normalize_latents(history_raw)
        next_latent = self.normalize_latents(next_latent_raw)
        if mode == "paper_clean":
            model_input = next_latent
        elif mode == "noisy_probe":
            if noise is None:
                raise ValueError("noisy_probe requires fixed noise")
            model_input = next_latent + float(noise_level) * noise.to(next_latent)
        else:
            raise ValueError(f"Unknown DSM mode: {mode}")
        prediction = self.model(
            history,
            action_normalized,
            model_input,
            noise_level,
        )
        return prediction.reshape(-1).pow(2).mean()

    def contrastive_energy(
        self,
        history_raw: torch.Tensor,
        action_normalized: torch.Tensor,
        next_latent_raw: torch.Tensor,
        *,
        noise_level: float,
    ) -> torch.Tensor:
        history = self.normalize_latents(history_raw)
        next_latent = self.normalize_latents(next_latent_raw)
        return self.model.energy(
            history,
            action_normalized,
            next_latent,
            noise_level=noise_level,
            reduction="mean",
        )
