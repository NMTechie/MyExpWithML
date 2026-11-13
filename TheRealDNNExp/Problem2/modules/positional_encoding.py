"""Sinusoidal and learned positional encodings.

Sinusoidal is a registered buffer -> 0 trainable parameters.
Learned is an nn.Embedding -> max_len * d_model trainable parameters.
The contrast matters for the Task A positional-information ablation
(none / sinusoidal / learned).
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)); PE[pos, 2i+1] = cos(pos / 10000^(2i/d))."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            max_len, dtype=torch.float32).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        # (1, max_len, d_model); buffer, not Parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> same shape, with position info added."""
        seq_len = x.shape[1]
        # (1, T, d_model) broadcasts over batch
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """nn.Embedding lookup over position indices 0..max_len-1 (trainable)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # (max_len, d_model) trainable weight
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> same shape, with learned position vectors added."""
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)       # (T,)
        # (1, T, d_model) broadcasts
        return x + self.embedding(positions).unsqueeze(0)


class NoPositionalEncoding(nn.Module):
    """Identity module -- used as the 'none' arm of the Task A ablation.

    Accepts (and ignores) d_model/max_len so it matches the constructor
    signature of SinusoidalPositionalEncoding/LearnedPositionalEncoding --
    this lets POSITIONAL_ENCODINGS[mode](d_model=..., max_len=...) in
    har_encoder.py call all three variants uniformly.
    """

    def __init__(self, d_model: int = 0, max_len: int = 0) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
