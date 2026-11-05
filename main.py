"""Train the GPT model on the local text corpus."""

import torch

from src.config import GPT_CONFIG_124M
from src.dataloader import create_dataloader
from src.gpt_model import GPTModel
from src.generation import save_model
from src.tokenizer import BPETokenizer
from src.training import train_model_simple


def main() -> None:
    with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = BPETokenizer()
    token_ids = tokenizer.encode(text_data)
    print("Characters:", len(text_data))
    print("Tokens:", len(token_ids))

    train_ratio = 0.90
    split_index = int(train_ratio * len(token_ids))
    train_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]
    config = GPT_CONFIG_124M.copy()

    torch.manual_seed(123)
    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=True,
        drop_last=True,
    )
    validation_loader = create_dataloader(
        validation_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1,
    )
    train_model_simple(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    save_model(model, "model.pth")
    print("Model weights saved to model.pth")


if __name__ == "__main__":
    main()
