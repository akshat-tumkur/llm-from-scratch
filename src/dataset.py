from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        token_ids: Sequence[int],
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids: List[Tensor] = []
        self.target_ids: List[Tensor] = []
        for start in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(
                    token_ids[start : start + max_length],
                    dtype=torch.long,
                )
            )
            self.target_ids.append(
                torch.tensor(
                    token_ids[start + 1 : start + max_length + 1],
                    dtype=torch.long,
                )
            )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.input_ids[index], self.target_ids[index]
