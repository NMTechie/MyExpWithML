"""Character-level causal language model on Tiny Shakespeare.

Context window 128, d_model=128, 4 heads, 4 layers (assignment spec).
Reuses causal_mask / TransformerEncoder / SinusoidalPositionalEncoding from
the rest of Problem 2 -- no attention math is reimplemented here.
"""

import os
import urllib.request
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import causal_mask
from encoder import TransformerEncoder
from positional_encoding import SinusoidalPositionalEncoding

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_tiny_shakespeare(data_dir: str) -> str:
    """Download Tiny Shakespeare into data_dir/input.txt if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    text_path = os.path.join(data_dir, "input.txt")
    if not os.path.isfile(text_path):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, text_path)
    return text_path


def load_text(text_path: str) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Return (sorted unique chars, char->index, index->char)."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)  # (len(text),)


def decode(indices: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in indices)


def train_val_split(data: torch.Tensor, val_fraction: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contiguous split (not shuffled) -- correct for a language-modeling stream."""
    split = int(len(data) * (1 - val_fraction))
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor, context_len: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences for next-char prediction.

    input  = data[i : i+context_len]
    target = data[i+1 : i+context_len+1]  (shifted by one character)
    """
    max_start = len(data) - context_len - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + context_len] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + context_len + 1] for i in ix])  # (B, T)
    return x.to(device), y.to(device)


class CharTransformerLM(nn.Module):
    """Token embedding + sinusoidal PE + causal encoder stack + LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)   # (vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)          # (d_model,) -> (vocab_size,)

    def forward(self, idx: torch.Tensor, use_mask: bool = True) -> torch.Tensor:
        """idx: (B, T) long token ids -> logits: (B, T, vocab_size).

        use_mask=False disables the causal mask (used only by the Step 6
        no-mask ablation; training here always calls with use_mask=True).
        """
        batch, seq_len = idx.shape
        x = self.token_embed(idx)                              # (B, T, d_model)
        x = self.pos_encoding(x)                                # (B, T, d_model)
        mask = causal_mask(seq_len).to(idx.device) if use_mask else None  # (T, T) or None
        x = self.encoder(x, mask=mask)                          # (B, T, d_model)
        return self.lm_head(x)                                  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """idx: (B, T0) seed context -> (B, T0 + max_new_tokens), one token at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]                   # crop to the context window
            logits = self(idx_cond)                             # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature             # (B, vocab_size); last position only
            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)       # (B, top_k)
                logits[logits < top_values[:, [-1]]] = float("-inf")  # mask everything outside top-k
            probs = F.softmax(logits, dim=-1)                   # (B, vocab_size)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)             # (B, T+1)
        return idx


@torch.no_grad()
def estimate_loss(
    model: CharTransformerLM, data: torch.Tensor, context_len: int,
    batch_size: int, eval_iters: int, device: torch.device, use_mask: bool = True,
) -> float:
    """Average cross-entropy loss (nats) over eval_iters random batches.

    use_mask=False disables the causal mask (Step 6c no-mask ablation only).
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        xb, yb = get_batch(data, context_len, batch_size, device)
        logits = model(xb, use_mask=use_mask)                   # (B, T, vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

def train_char_lm(
    model: CharTransformerLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    context_len: int,
    batch_size: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    lr: float,
    device: torch.device,
    use_mask: bool = True,
) -> Dict[str, object]:
    """Shared training loop for Step 5's main run and Step 6's ablations (6b/6c).
    Returns a dict with train_loss_history (list of (iter, loss)),
    val_loss_history (list of (iter, loss)), sec_per_block, and final_val_loss."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    block_times = []
    train_loss_history = []
    val_loss_history = []
    block_start = time.time()

    for it in range(1, max_iters + 1):
        xb, yb = get_batch(train_data, context_len, batch_size, device)
        logits = model(xb, use_mask=use_mask)                    # (B, T, vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters:
            block_times.append(time.time() - block_start)
            block_start = time.time()
            val_loss = estimate_loss(
                model, val_data, context_len, batch_size, eval_iters, device, use_mask=use_mask)
            train_loss_history.append((it, loss.item()))
            val_loss_history.append((it, val_loss))
            print(f"  iter {it}/{max_iters} - train_loss={loss.item():.4f} - "
                  f"val_loss={val_loss:.4f} nats - {block_times[-1]:.1f}s/{eval_interval}steps")

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "sec_per_block": float(np.mean(block_times)),
        "final_val_loss": val_loss_history[-1][1],
    }