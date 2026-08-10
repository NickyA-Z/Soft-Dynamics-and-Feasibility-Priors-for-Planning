"""Typed storage for paired planner-generated hard negatives.

Artifacts contain the exact normalized tensors consumed by ``model.energy``.
Normalization therefore belongs at the planner/export boundary, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


TENSOR_KEYS = (
    "histories",
    "positive_actions",
    "positive_next_latents",
    "negative_actions",
    "negative_next_latents",
)


class HardNegativeDataset(Dataset):
    def __init__(self, **tensors: torch.Tensor) -> None:
        missing = set(TENSOR_KEYS) - set(tensors)
        if missing:
            raise ValueError(f"Missing hard-negative tensors: {sorted(missing)}")
        counts = {int(tensors[key].shape[0]) for key in TENSOR_KEYS}
        if len(counts) != 1:
            raise ValueError("Hard-negative tensors have different sample counts")
        self.tensors = {key: tensors[key].float() for key in TENSOR_KEYS}
        if (
            self.tensors["positive_actions"].shape
            != self.tensors["negative_actions"].shape
        ):
            raise ValueError("Positive and negative action shapes differ")
        if (
            self.tensors["positive_next_latents"].shape
            != self.tensors["negative_next_latents"].shape
        ):
            raise ValueError("Positive and negative next-latent shapes differ")

    def __len__(self) -> int:
        return int(self.tensors["histories"].shape[0])

    def __getitem__(self, index: int):
        return tuple(self.tensors[key][index] for key in TENSOR_KEYS)


def save_hard_negative_dataset(
    path: str | Path,
    *,
    metadata: dict | None = None,
    **tensors: torch.Tensor,
) -> None:
    dataset = HardNegativeDataset(**tensors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **{key: value.cpu() for key, value in dataset.tensors.items()},
            "metadata": metadata or {},
            "format_version": 1,
            "normalized": True,
        },
        path,
    )


def load_hard_negative_dataset(path: str | Path) -> tuple[HardNegativeDataset, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data.get("normalized", False):
        raise ValueError("Hard-negative artifact must contain normalized model inputs")
    dataset = HardNegativeDataset(**{key: data[key] for key in TENSOR_KEYS})
    return dataset, data.get("metadata", {})


def collate_hard_negative_records(records: Iterable[dict]) -> dict[str, torch.Tensor]:
    """Stack planner-export records into arguments accepted by the save helper."""
    records = list(records)
    if not records:
        raise ValueError("Cannot collate an empty hard-negative record sequence")
    singular = {
        "histories": "history",
        "positive_actions": "positive_action",
        "positive_next_latents": "positive_next_latent",
        "negative_actions": "negative_action",
        "negative_next_latents": "negative_next_latent",
    }
    return {
        plural: torch.stack([record[key] for record in records])
        for plural, key in singular.items()
    }


def save_hard_negative_records(
    path: str | Path,
    records: Iterable[dict],
    *,
    metadata: dict | None = None,
) -> None:
    """Save records returned by ``mining.make_hard_negative_record``."""
    records = list(records)
    tensors = collate_hard_negative_records(records)
    diagnostics = [
        {
            key: record[key]
            for key in ("positive_energy", "negative_energy", "hardness")
            if key in record
        }
        for record in records
    ]
    artifact_metadata = dict(metadata or {})
    artifact_metadata["mining_diagnostics"] = diagnostics
    save_hard_negative_dataset(path, metadata=artifact_metadata, **tensors)
