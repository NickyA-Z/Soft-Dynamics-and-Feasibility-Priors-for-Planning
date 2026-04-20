"""
Tests for dino_oracle_demo.parse_args.

conftest.py pre-patches sys.modules so the dino_wm imports don't fail.
"""
from __future__ import annotations

import pytest

from gvpwm.examples.dino_oracle_demo import parse_args


def _parse(args: list[str]):
    """Call parse_args with an explicit argv list (bypasses sys.argv)."""
    import sys
    old_argv = sys.argv
    sys.argv = ["prog"] + args
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_defaults():
    args = _parse([])
    assert args.split == "val"
    assert args.horizon == 25
    assert args.start_index == 0
    assert args.num_episodes == 50


def test_num_episodes_override():
    args = _parse(["--num-episodes", "1"])
    assert args.num_episodes == 1


def test_start_index():
    args = _parse(["--start-index", "5", "--num-episodes", "3"])
    assert args.start_index == 5
    assert args.num_episodes == 3


def test_split_and_horizon_overrides():
    args = _parse([
        "--split", "train",
        "--horizon", "80",
        "--start-index", "7",
    ])
    assert args.split == "train"
    assert args.horizon == 80
    assert args.start_index == 7


def test_invalid_episode_type_raises():
    with pytest.raises(SystemExit):
        _parse(["--num-episodes", "not_an_int"])
