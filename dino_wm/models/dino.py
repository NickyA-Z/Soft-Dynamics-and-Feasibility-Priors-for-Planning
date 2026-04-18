import os
from pathlib import Path
import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


def _load_dinov2_offline(name: str) -> torch.nn.Module:
    """Load DINOv2 without network access by loading cached weights manually."""
    model = torch.hub.load("facebookresearch/dinov2", name, pretrained=False)
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / f"{name}_pretrain.pth"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"DINOv2 pretrained weights not found at {weights_path}. "
            "Run once with internet access to cache them."
        )
    return model


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = _load_dinov2_offline(name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb