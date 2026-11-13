"""Multi-head attention built on top of attention.py's scaled dot-product core.

Reuses scaled_dot_product_attention/causal_mask directly so there is a single
source of truth for the attention math (verified by the Task 1 unit tests).
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """W_Q, W_K, W_V, W_O projections with head splitting/merging.

    d_model must be divisible by n_heads; d_k = d_model // n_heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # weight (d_model, d_model), bias (d_model,)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """(B, T, d_model) -> (B, n_heads, T, d_k)."""
        x = x.view(batch, seq_len, self.n_heads, self.d_k)  # (B, T, H, d_k)
        return x.transpose(1, 2)                            # (B, H, T, d_k)

    def _merge_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """(B, n_heads, T, d_k) -> (B, T, d_model)."""
        x = x.transpose(1, 2).contiguous()      # (B, T, H, d_k)
        return x.view(batch, seq_len, self.d_model)  # (B, T, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, d_model) -> (B, T, d_model). Self-attention only (encoder use case).

        mask: None, or (T, T) boolean (True = keep, False = block); broadcasts
        against (B, n_heads, T, T) attention scores.

        return_weights=True additionally returns the (B, n_heads, T, T)
        attention weights (used for the Step 6 attention heatmaps); default
        False preserves the exact return type used by every earlier step.
        """
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.W_Q(x), batch, seq_len)  # (B, H, T, d_k)
        k = self._split_heads(self.W_K(x), batch, seq_len)  # (B, H, T, d_k)
        v = self._split_heads(self.W_V(x), batch, seq_len)  # (B, H, T, d_k)

        if return_weights:
            attn_out, weights = scaled_dot_product_attention(
                q, k, v, mask=mask, return_weights=True)  # (B,H,T,d_k), (B,H,T,T)
            merged = self._merge_heads(attn_out, batch, seq_len)  # (B, T, d_model)
            return self.W_O(merged), weights
        else:
            attn_out = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, H, T, d_k)
            merged = self._merge_heads(attn_out, batch, seq_len)         # (B, T, d_model)
            return self.W_O(merged)                                     # (B, T, d_model)