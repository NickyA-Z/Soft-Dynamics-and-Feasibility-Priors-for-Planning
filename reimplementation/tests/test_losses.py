import torch

from gvpwm.losses import scale_invariant_alignment


def test_scale_invariant_alignment_is_zero_under_scaling():
    reference = torch.tensor([1.0, 2.0, 3.0])
    scaled = reference * 7.5
    loss = scale_invariant_alignment(reference, scaled)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
