"""Feed-forward network from the GPT transformer block."""

import torch
from torch import Tensor, nn


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return 0.5 * inputs * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=inputs.device))
                * (inputs + 0.044715 * torch.pow(inputs, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)
