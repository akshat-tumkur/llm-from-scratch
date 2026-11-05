"""GPT pretraining loop."""

from typing import List, Tuple

import torch

from .generation import generate_and_print_sample
from .loss import calc_loss_batch, calc_loss_loader
from .tokenizer import BPETokenizer


def evaluate_model(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    eval_iter: int,
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        validation_loss = calc_loss_loader(
            validation_loader,
            model,
            device,
            eval_iter,
        )
    model.train()
    return train_loss, validation_loss


def train_model_simple(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: BPETokenizer,
) -> Tuple[List[float], List[float], List[int]]:
    train_losses: List[float] = []
    validation_losses: List[float] = []
    tokens_seen: List[int] = []
    total_tokens = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total_tokens += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(
                    model,
                    train_loader,
                    validation_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                tokens_seen.append(total_tokens)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {validation_loss:.3f}"
                )

        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )

    return train_losses, validation_losses, tokens_seen
