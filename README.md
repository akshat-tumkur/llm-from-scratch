# LLM from Scratch

A modular GPT-style language model implemented in PyTorch for learning and
experimentation.

## Features

- GPT-2 BPE tokenization with `tiktoken`
- Sliding-window next-token prediction datasets
- Train/validation data loaders
- Causal multi-head self-attention
- Token and positional embeddings
- Layer normalization and residual connections
- GELU feed-forward networks
- Transformer blocks and GPT model
- Cross-entropy training with AdamW
- Greedy generation
- Temperature and top-k sampling
- Saving and loading model weights

## Project structure

```text
llm-from-scratch/
├── data/
│   └── the-verdict.txt
├── src/
│   ├── config.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── attention.py
│   ├── layer_norm.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   ├── gpt_model.py
│   ├── generation.py
│   ├── loss.py
│   └── training.py
├── main.py
├── generate_text.py
├── requirements.txt
└── README.md
```

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Training

Run from the project root:

```powershell
python main.py
```

Training uses:

- The corpus in `data/the-verdict.txt`
- A 90/10 training and validation split
- Context length of 256 tokens
- Batch size of 2
- AdamW with learning rate `0.0004`
- Weight decay of `0.1`
- 10 epochs
- Loss evaluation every 5 steps
- Text generation after each epoch

After training, model weights are saved to `model.pth`.

## Text generation

Run:

```powershell
python generate_text.py
```

The generation utility loads `model.pth`, asks for a prompt, and supports
greedy decoding or temperature/top-k sampling.

