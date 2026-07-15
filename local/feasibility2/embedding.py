import torch
import torch.nn as nn
import math


class SinusoidalSigmaEmbedding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, embed_dim: int):
        super().__init__()

        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        half = embed_dim // 2

        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(10000), half)))

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B, 1]
        sigma = torch.log(sigma.clamp(min=1e-6))

        args = sigma * self.freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)
