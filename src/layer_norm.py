"""Layer normalization used by transformer blocks."""

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.shift
