"""Cross-entropy loss utilities."""

from typing import Iterable, Tuple

import torch
from torch import Tensor

from .gpt_model import GPTModel


def calc_loss_batch(
    input_batch: Tensor,
    target_batch: Tensor,
    model: GPTModel,
    device: torch.device,
) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


def calc_loss_loader(
    data_loader: Iterable[Tuple[Tensor, Tensor]],
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if hasattr(data_loader, "__len__") and len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)  # type: ignore[arg-type]
    total_batches = min(num_batches, len(data_loader))  # type: ignore[arg-type]
    with torch.no_grad():
        for index, (input_batch, target_batch) in enumerate(data_loader):
            if index >= total_batches:
                break
            total_loss += calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            ).item()
    return total_loss / total_batches
