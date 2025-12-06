import sys
import argparse

import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
# sampling utils
# -----------------------------------------------------------------------------

def top_k_logits(logits: torch.Tensor, k: int | None):
    """
    Keep only top-k logits, set the rest to -inf so they get zero prob.
    logits: (B, vocab_size)
    """
    if k is None:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[:, -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def generate(model: GPT,
             idx: torch.Tensor,
             max_new_tokens: int,
             temperature: float = 1.0,
             top_k: int | None = 50) -> torch.Tensor:
    """
    Autoregressive generation.

    Args:
        model: GPT model
        idx: (B, T) int64 prompt tokens on the correct device
        max_new_tokens: how many tokens to generate
        temperature: softmax temperature
        top_k: top-k sampling (None disables)

    Returns:
        (B, T + max_new_tokens) tensor of token ids
    """
    model.eval()

    for _ in range(max_new_tokens):
        # only feed last n_ctx tokens into the model
        idx_cond = idx[:, -model.config.n_ctx:]

        logits, _ = model(idx_cond)          # (B, T_cond, vocab_size)
        logits = logits[:, -1, :]           # (B, vocab_size) – last time step

        # temperature + top-k
        logits = logits / max(temperature, 1e-8)
        logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # append to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx


# -----------------------------------------------------------------------------
# state_dict cleaning
# -----------------------------------------------------------------------------

def clean_state_dict(sd: dict) -> dict:
    """
    Remove wrappers like '_orig_mod.' and 'module.' from checkpoint keys,
    so they match the plain GPT model.
    """
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        new_sd[new_k] = v
    return new_sd


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to checkpoint (.pt) file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (<=0 -> greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k for sampling (None for no top-k)",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional text prompt",
    )
    args = parser.parse_args()

    # device selection
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    print("Using device:", device)

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # rebuild config & model
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)

    # fix state_dict keys (_orig_mod., module., etc.)
    raw_state_dict = ckpt["model"]
    cleaned_state_dict = clean_state_dict(raw_state_dict)

    # load weights
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = "Once upon a time,"
    print("Prompt:", repr(prompt))

    # encode
    input_ids = enc.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # generate
    out = generate(
        model,
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # decode
    out_ids = out[0].tolist()
    text = enc.decode(out_ids)

    print("\n=== OUTPUT ===\n")
    print(text)


if __name__ == "__main__":
    main()
