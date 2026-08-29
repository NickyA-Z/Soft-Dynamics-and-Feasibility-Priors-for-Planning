from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch


# Importing the legacy dataset module normally loads the full Push-T dataset
# dependency stack only to obtain these two constants. Keep this unit test
# independent of optional video packages such as decord.
pusht_stub = types.ModuleType("dino_wm.datasets.pusht_dset")
pusht_stub.ACTION_MEAN = torch.zeros(2)
pusht_stub.ACTION_STD = torch.ones(2)
sys.modules.setdefault("dino_wm.datasets.pusht_dset", pusht_stub)

from extension.feasibility2.train import validate_dataset  # noqa: E402


def make_dataset(history_length: int, history_latent_shape=(196, 394)):
    return SimpleNamespace(
        histories=torch.zeros(1, history_length, *history_latent_shape),
        actions=torch.zeros(1, 10),
        next_latents=torch.zeros(1, 196, 394),
    )


@pytest.mark.parametrize("history_length", [1, 3])
def test_validate_dataset_accepts_wall_and_pusht_history_lengths(
    history_length: int,
) -> None:
    validate_dataset(make_dataset(history_length))


def test_validate_dataset_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="positive history length"):
        validate_dataset(make_dataset(0))


def test_validate_dataset_rejects_mismatched_history_latent_shape() -> None:
    with pytest.raises(ValueError, match="History latent shape"):
        validate_dataset(make_dataset(1, history_latent_shape=(195, 394)))
