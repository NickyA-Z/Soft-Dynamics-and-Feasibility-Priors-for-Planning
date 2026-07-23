from __future__ import annotations

import torch


def make_initial_action_parameters(
    base_parameter: torch.Tensor,
    num_starts: int,
    noise_std: float,
    *,
    include_base: bool = True,
    raw_action_limit: float | None = 3.0,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Create a base initialization plus independently perturbed copies."""

    if num_starts < 1:
        raise ValueError("num_starts must be at least 1")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    initializations: list[torch.Tensor] = []
    if include_base:
        initializations.append(base_parameter.detach().clone())

    while len(initializations) < num_starts:
        noise = torch.randn(
            base_parameter.shape,
            device=base_parameter.device,
            dtype=base_parameter.dtype,
            generator=generator,
        )
        candidate = base_parameter.detach().clone().add_(noise, alpha=noise_std)
        if raw_action_limit is not None:
            candidate.clamp_(-raw_action_limit, raw_action_limit)
        initializations.append(candidate)

    return initializations
