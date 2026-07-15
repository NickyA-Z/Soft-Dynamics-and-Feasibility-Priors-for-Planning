from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .interfaces import VideoPlan, VideoPlanSource


def temporal_resample_sequence(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[0] == target_length:
        return sequence.clone()
    if sequence.shape[0] == 1:
        return sequence.expand(target_length, *sequence.shape[1:]).clone()
    flat = sequence.reshape(sequence.shape[0], -1).T.unsqueeze(0)
    resized = F.interpolate(flat, size=target_length, mode="linear", align_corners=True)
    return resized.squeeze(0).T.reshape(target_length, *sequence.shape[1:])


def apply_motion_blur(video: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return video.clone()
    blurred = []
    for index in range(video.shape[0]):
        start = max(0, index - window // 2)
        end = min(video.shape[0], start + window)
        start = max(0, end - window)
        blurred.append(video[start:end].mean(dim=0))
    return torch.stack(blurred, dim=0)


@dataclass
class PrecomputedVideoPlanSource(VideoPlanSource):
    data: Any
    encoded: bool = False

    def generate(self, initial_observation: Any, goal_observation: Any, horizon: int) -> VideoPlan:
        return VideoPlan(data=self.data, encoded=self.encoded)


@dataclass
class CallableVideoPlanSource(VideoPlanSource):
    generator: Callable[[Any, Any, int], Any]
    encoded: bool = False

    def generate(self, initial_observation: Any, goal_observation: Any, horizon: int) -> VideoPlan:
        result = self.generator(initial_observation, goal_observation, horizon)
        if isinstance(result, VideoPlan):
            return result
        return VideoPlan(data=result, encoded=self.encoded)
