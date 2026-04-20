"""
Shared fixtures and sys.modules patching for tests that import dino_oracle_demo.

dino_oracle_demo.py has module-level imports from the dino_wm repo
(plan, datasets.pusht_dset) that are not available outside the cluster
checkpoint environment. We patch sys.modules before any import so that
all test files in this directory can safely import from the demo module.
"""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import torch

# ---------------------------------------------------------------------------
# Patch dino_wm repo dependencies before any test imports the demo module
# ---------------------------------------------------------------------------
_fake_plan = types.ModuleType("plan")
_fake_plan.load_model = None  # tests that use evaluate_episode must skip
_fake_plan.__spec__ = ModuleSpec(name="plan", loader=None)
sys.modules.setdefault("plan", _fake_plan)

_fake_datasets = types.ModuleType("datasets")
_fake_datasets.__spec__ = ModuleSpec(name="datasets", loader=None)
_fake_pusht = types.ModuleType("datasets.pusht_dset")
_fake_pusht.__spec__ = ModuleSpec(name="datasets.pusht_dset", loader=None)
_fake_pusht.ACTION_MEAN = torch.zeros(2, dtype=torch.float32)
_fake_pusht.ACTION_STD = torch.ones(2, dtype=torch.float32)
sys.modules.setdefault("datasets", _fake_datasets)
sys.modules.setdefault("datasets.pusht_dset", _fake_pusht)
