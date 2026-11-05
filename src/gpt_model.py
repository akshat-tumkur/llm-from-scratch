import torch
from torch import Tensor, nn

from .layer_norm import LayerNorm
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx: Tensor) -> Tensor:
        batch_size, sequence_length = in_idx.shape
        token_embeddings = self.tok_emb(in_idx)
        position_embeddings = self.pos_emb(
            torch.arange(sequence_length, device=in_idx.device)
        )
        x = self.drop_emb(token_embeddings + position_embeddings)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
