from typing import Sequence, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import GPTDatasetV1


def create_dataloader(
    token_ids: Sequence[int],
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader[Tuple[Tensor, Tensor]]:
    dataset = GPTDatasetV1(token_ids, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )
