import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048
    n_ctx: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = nn.Dropout(config.dropout)

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        ## batchsize, n ctx, d model
        B, T, C  = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        d_head = C // self.n_head
        q = q.view(B, T, self.n_head, d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_head).transpose(1, 2)


        # # attention = (q @ k^t) / sqrt(d_k) @ v
        # attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
        # attn = attn.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v
        # faster version from pytorch
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output proj
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        # fc is up proj c proj is down proj
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x) ** 2 
        x = self.c_proj(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
        return self.weight * x / (norm + self.eps)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.ln_2 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # token and positional embedding weights, also heads
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.n_ctx, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.d_model)
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying for slightly better perplexity
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.n_ctx
        device = idx.device
        pos = torch.arange(0, T, device=device).unsqueeze(0) # (1, T)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        # put x through all blocks
        for block in self.transformer.h:
            x = block(x)
        # final layer norm and head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # softcap logits
        logits = 30.0 * torch.tanh(logits / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, block_size: int):
        # don't upcast the entire thing to int64
        self.tokens = tokens  # keep as np.int32
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx):
        start = idx
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]                  # numpy slice
        chunk = torch.from_numpy(chunk.astype(np.int64))  # small conversion here
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
        
def main():
    config = GPTConfig()
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))  # (B=2, T=16)
    logits, loss = model(x, x)
    print(logits.shape, loss.item())

if __name__ == "__main__":
    main()