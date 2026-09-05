from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def ensure_history_length(
    sequence: torch.Tensor,
    target_length: int,
    pad_mode: Literal["repeat_first", "zeros"] = "repeat_first",
) -> torch.Tensor:
    if sequence.shape[0] >= target_length:
        return sequence[-target_length:].clone()
    pad_count = target_length - sequence.shape[0]
    if pad_mode == "repeat_first":
        pad_value = sequence[:1].expand(pad_count, *sequence.shape[1:])
    elif pad_mode == "zeros":
        pad_value = torch.zeros(
            pad_count,
            *sequence.shape[1:],
            dtype=sequence.dtype,
            device=sequence.device,
        )
    else:
        raise ValueError(f"Unsupported pad mode: {pad_mode}")
    return torch.cat([pad_value, sequence], dim=0)


def shift_latent_warm_start(latents: torch.Tensor, executed: int) -> torch.Tensor:
    if executed <= 0:
        return latents.clone()
    if executed >= latents.shape[0]:
        return latents[-1:].expand_as(latents).clone()
    tail = latents[executed:]
    pad = latents[-1:].expand(executed, *latents.shape[1:])
    return torch.cat([tail, pad], dim=0)


def shift_action_warm_start(actions: torch.Tensor, executed: int) -> torch.Tensor:
    if executed <= 0:
        return actions.clone()
    if executed >= actions.shape[0]:
        return torch.zeros_like(actions)
    tail = actions[executed:]
    pad = torch.zeros(
        executed,
        *actions.shape[1:],
        dtype=actions.dtype,
        device=actions.device,
    )
    return torch.cat([tail, pad], dim=0)


def center_pad_to_aspect_ratio(
    images: torch.Tensor,
    target_aspect_ratio: float,
    fill: float = 0.0,
) -> torch.Tensor:
    height = images.shape[-2]
    width = images.shape[-1]
    current_ratio = width / height
    if abs(current_ratio - target_aspect_ratio) < 1e-6:
        return images
    if current_ratio < target_aspect_ratio:
        new_width = int(round(height * target_aspect_ratio))
        pad_total = new_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padding = (pad_left, pad_right, 0, 0)
    else:
        new_height = int(round(width / target_aspect_ratio))
        pad_total = new_height - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padding = (0, 0, pad_top, pad_bottom)
    return F.pad(images, padding, value=fill)


def center_crop(images: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    height = images.shape[-2]
    width = images.shape[-1]
    if target_height > height or target_width > width:
        raise ValueError("Target crop must fit inside the input tensor.")
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return images[..., top : top + target_height, left : left + target_width]
