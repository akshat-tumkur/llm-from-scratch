"""GPT-2 byte-pair tokenizer."""

from typing import List, Sequence

import tiktoken


class BPETokenizer:
    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.encoding.decode(list(token_ids))
