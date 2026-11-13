"""Pre-LayerNorm transformer encoder block and a stack of N blocks.

Pre-LN: LayerNorm is applied BEFORE each sub-layer (not after, as in the
original 2017 Transformer's post-LN). This keeps the residual stream clean
for gradient flow, which matters for the 4-layer stack in Task B.
"""

from typing import Optional

import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """One pre-LN block: LN -> MHA -> residual, then LN -> FFN(4d wide) -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # weight+bias, each (d_model,)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.drop1 = nn.Dropout(dropout)

        # weight+bias, each (d_model,)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),   # weight (d_ff, d_model), bias (d_ff,)
            nn.GELU(),
            nn.Dropout(dropout),
            # weight (d_model, d_ff), bias (d_model,)
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). mask: None, or (T, T) boolean (True=keep)."""
        attn_out = self.mha(self.ln1(x), mask=mask)  # (B, T, d_model)
        x = x + self.drop1(attn_out)                 # residual 1

        ff_out = self.ff(self.ln2(x))                # (B, T, d_model)
        x = x + self.drop2(ff_out)                   # residual 2
        return x


class TransformerEncoder(nn.Module):
    """Stack of n_layers identical EncoderBlocks, with a final LayerNorm (pre-LN convention)."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        # pre-LN stacks need one final norm before the head
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_ln(x)
