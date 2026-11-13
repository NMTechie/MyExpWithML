"""Scaled dot-product attention, built from matrix operations only."""

import math
from typing import Optional, Tuple, Union

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_weights: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """q, k, v: (..., seq_len, d_k) -> output: (..., seq_len, d_k).

    mask: optional boolean tensor, True = keep, False = block (set to -inf
    before softmax). Must broadcast to (..., seq_len_q, seq_len_k).
    """
    d_k = q.shape[-1]
    # (..., seq_len_q, seq_len_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = weights @ v                     # (..., seq_len_q, d_k)
    if return_weights:
        return output, weights
    return output


def causal_mask(seq_len: int) -> torch.Tensor:
    """Boolean mask (seq_len, seq_len), True where query i may attend to key j (j <= i)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
