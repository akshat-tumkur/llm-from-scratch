import torch
from torch import Tensor

from .config import GPT_CONFIG_124M
from .gpt_model import GPTModel
from .tokenizer import BPETokenizer


def text_to_token_ids(text: str, tokenizer: BPETokenizer) -> Tensor:
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: Tensor, tokenizer: BPETokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(
    model: torch.nn.Module,
    token_ids: Tensor,
    max_new_tokens: int,
    context_size: int,
) -> Tensor:
    for _ in range(max_new_tokens):
        context = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(context)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids


def generate(
    model: torch.nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Generate tokens with optional temperature and top-k sampling."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond) # forward pass
        logits = logits[:, -1, :]

        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive")
            top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    device: torch.device,
    start_context: str,
) -> None:
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    generated = generate_text_simple(
        model,
        encoded,
        max_new_tokens=50,
        context_size=model.pos_emb.weight.shape[0],
    )
    print(token_ids_to_text(generated, tokenizer).replace("\n", " "))
    model.train()


def save_model(model: GPTModel, path: str = "model.pth") -> None:
    """Save model weights as a PyTorch state dictionary."""
    torch.save(model.state_dict(), path)


def load_model(
    path: str = "model.pth",
    config: dict = GPT_CONFIG_124M,
    device: torch.device | None = None,
) -> GPTModel:
    """Create a GPT model and load weights saved with save_model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model