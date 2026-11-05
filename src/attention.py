"""Causal multi-head self-attention."""
import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        queries = self.query(inputs).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = self.key(inputs).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = self.value(inputs).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill_(
            self.mask.bool()[:sequence_length, :sequence_length],
            -torch.inf,
        )
        attention_weights = torch.softmax(
            attention_scores / self.head_dim**0.5,
            dim=-1,
        )
        context = (self.dropout(attention_weights) @ values).transpose(1, 2)
        context = context.contiguous().view(batch_size, sequence_length, -1)
        return self.output_projection(context)
