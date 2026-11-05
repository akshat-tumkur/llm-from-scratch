"""Transformer block with pre-layer normalization and residual paths."""

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.feed_forward = FeedForward(config["emb_dim"])
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.dropout(self.attention(self.norm1(inputs)))
        return inputs + self.dropout(self.feed_forward(self.norm2(inputs)))
