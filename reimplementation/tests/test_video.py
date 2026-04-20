"""Tests for temporal_resample_sequence."""
import torch
import pytest

from gvpwm.video import temporal_resample_sequence


def make_seq(n, d=4):
    """Linearly spaced sequence so interpolation is exact."""
    return torch.stack([torch.full((d,), float(i)) for i in range(n)])


def test_same_length_returns_clone():
    seq = make_seq(5)
    out = temporal_resample_sequence(seq, 5)
    assert out.shape == seq.shape
    assert torch.allclose(out, seq)
    assert out.data_ptr() != seq.data_ptr()  # clone, not same tensor


def test_upsample_doubles_length():
    seq = make_seq(3, d=2)
    out = temporal_resample_sequence(seq, 6)
    assert out.shape[0] == 6


def test_downsample_halves_length():
    seq = make_seq(10)
    out = temporal_resample_sequence(seq, 5)
    assert out.shape[0] == 5


def test_single_frame_expands_by_repeat():
    frame = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    out = temporal_resample_sequence(frame, 4)
    assert out.shape == (4, 3)
    assert torch.allclose(out, frame.expand(4, -1))


def test_endpoints_preserved_under_resampling():
    """Linear interpolation must keep first and last frames exact."""
    seq = make_seq(5)
    out = temporal_resample_sequence(seq, 9)
    assert torch.allclose(out[0], seq[0], atol=1e-5)
    assert torch.allclose(out[-1], seq[-1], atol=1e-5)


def test_output_feature_dims_unchanged():
    seq = torch.randn(7, 3, 4)  # (T, P, D)
    out = temporal_resample_sequence(seq, 11)
    assert out.shape == (11, 3, 4)
