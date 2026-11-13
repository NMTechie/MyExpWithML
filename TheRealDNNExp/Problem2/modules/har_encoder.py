"""Patch-based HAR encoder classifier: patch embed -> CLS token -> encoder stack -> head.

Input windows are (window_len=128, n_channels=9). We patchify into
window_len/patch_len=16 patches of 8 time steps each (flattened to
patch_len*n_channels=72 raw features per patch) rather than embedding every
single time step, per the assignment's "(or each patch of 8 steps)" option --
this keeps the attended sequence short (16 patches + 1 CLS = 17 tokens).
"""

import torch
import torch.nn as nn

from encoder import TransformerEncoder
from positional_encoding import (
    NoPositionalEncoding, SinusoidalPositionalEncoding, LearnedPositionalEncoding,
)

POSITIONAL_ENCODINGS = {
    "none": NoPositionalEncoding,
    "sinusoidal": SinusoidalPositionalEncoding,
    "learned": LearnedPositionalEncoding,
}


class HAREncoderClassifier(nn.Module):
    """Classifies HAR windows by patchifying, prepending a CLS token, running a
    bidirectional (no mask) transformer encoder, and classifying from the CLS
    token's final hidden state.
    """

    def __init__(
        self,
        n_channels: int = 9,
        window_len: int = 128,
        patch_len: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        n_classes: int = 6,
        pos_encoding: str = "sinusoidal",  # "none" | "sinusoidal" | "learned"
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_len % patch_len != 0:
            raise ValueError(f"window_len={window_len} must be divisible by patch_len={patch_len}")
        if pos_encoding not in POSITIONAL_ENCODINGS:
            raise ValueError(f"unknown pos_encoding={pos_encoding!r}")

        self.patch_len = patch_len
        self.n_patches = window_len // patch_len          # 128 / 8 = 16
        patch_dim = patch_len * n_channels                # 8 * 9 = 72 raw features per patch

        self.patch_embed = nn.Linear(patch_dim, d_model)              # (72,) -> (d_model,)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))     # (1, 1, d_model), prepended
        nn.init.normal_(self.cls_token, std=0.02)

        # +1 slot for the CLS token's own position
        self.pos_encoding = POSITIONAL_ENCODINGS[pos_encoding](
            d_model=d_model, max_len=self.n_patches + 1
        )
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window_len, n_channels) -> logits: (B, n_classes)."""
        batch, window_len, n_channels = x.shape
        # (B, window_len, n_channels) -> (B, n_patches, patch_len * n_channels)
        x = x.reshape(batch, self.n_patches, self.patch_len * n_channels)
        x = self.patch_embed(x)                                 # (B, n_patches, d_model)

        cls = self.cls_token.expand(batch, -1, -1)              # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                          # (B, n_patches+1, d_model); CLS at index 0

        x = self.pos_encoding(x)                                # (B, n_patches+1, d_model)
        x = self.encoder(x, mask=None)                          # bidirectional, no causal mask

        cls_out = x[:, 0, :]                                    # (B, d_model); CLS token's final state
        return self.classifier(cls_out)                         # (B, n_classes)