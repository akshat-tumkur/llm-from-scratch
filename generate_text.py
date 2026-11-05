import torch

from src.config import GPT_CONFIG_124M
from src.generation import generate, load_model, text_to_token_ids, token_ids_to_text
from src.tokenizer import BPETokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BPETokenizer()
model = load_model("model.pth", GPT_CONFIG_124M, device)

prompt = input("Prompt: ")
input_ids = text_to_token_ids(prompt, tokenizer).to(device)

output_ids = generate(
    model=model,
    idx=input_ids,
    max_new_tokens=50,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.0,
    top_k=25,
    eos_id=tokenizer.encoding.eot_token,
)

print(token_ids_to_text(output_ids, tokenizer))