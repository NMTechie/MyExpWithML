# models.py
"""Custom RNN, LSTM and GRU cells built from matrix operations only.

Each cell mirrors the parameter layout of PyTorch's RNNCell/LSTMCell/
GRUCell (separate input and hidden biases, same gate ordering) so that
weights can be copied directly from the library layers for the Step 5
equivalence check. No nn.RNN, nn.LSTM, nn.GRU or their *Cell classes
are used inside these implementations.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MyRNNCell(nn.Module):
    """Vanilla RNN cell: h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(
            torch.empty(hidden_size, input_size))   # (h, d)
        self.weight_hh = nn.Parameter(
            torch.empty(hidden_size, hidden_size))  # (h, h)
        self.bias_ih = nn.Parameter(torch.empty(
            hidden_size))                 # (h,)
        self.bias_hh = nn.Parameter(torch.empty(
            hidden_size))                 # (h,)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        k = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (batch, input_size), h_prev: (batch, hidden_size) -> h_t: (batch, hidden_size)."""
        gate_x = x_t @ self.weight_ih.T + self.bias_ih      # (batch, h)
        gate_h = h_prev @ self.weight_hh.T + self.bias_hh   # (batch, h)
        h_t = torch.tanh(gate_x + gate_h)                   # (batch, h)
        return h_t


class MyLSTMCell(nn.Module):
    """LSTM cell with PyTorch's gate order: input, forget, cell, output."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(
            4 * hidden_size, input_size))   # (4h, d)
        self.weight_hh = nn.Parameter(torch.empty(
            4 * hidden_size, hidden_size))  # (4h, h)
        self.bias_ih = nn.Parameter(torch.empty(
            4 * hidden_size))                 # (4h,)
        self.bias_hh = nn.Parameter(torch.empty(
            4 * hidden_size))                 # (4h,)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        k = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(
        self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_t: (batch, input_size); state = (h_prev, c_prev), each (batch, hidden_size)."""
        h_prev, c_prev = state
        gates = (
            x_t @ self.weight_ih.T + self.bias_ih
            + h_prev @ self.weight_hh.T + self.bias_hh
        )  # (batch, 4h)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(
            4, dim=1)  # each (batch, h)
        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_t = f_t * c_prev + i_t * g_t   # (batch, h)
        h_t = o_t * torch.tanh(c_t)      # (batch, h)
        return h_t, c_t


class MyGRUCell(nn.Module):
    """GRU cell with PyTorch's gate order: reset, update, new (candidate)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(
            3 * hidden_size, input_size))   # (3h, d)
        self.weight_hh = nn.Parameter(torch.empty(
            3 * hidden_size, hidden_size))  # (3h, h)
        self.bias_ih = nn.Parameter(torch.empty(
            3 * hidden_size))                 # (3h,)
        self.bias_hh = nn.Parameter(torch.empty(
            3 * hidden_size))                 # (3h,)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        k = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (batch, input_size), h_prev: (batch, hidden_size) -> h_t: (batch, hidden_size)."""
        gate_x = x_t @ self.weight_ih.T + self.bias_ih     # (batch, 3h)
        gate_h = h_prev @ self.weight_hh.T + self.bias_hh  # (batch, 3h)

        x_r, x_z, x_n = gate_x.chunk(3, dim=1)  # each (batch, h)
        h_r, h_z, h_n = gate_h.chunk(3, dim=1)  # each (batch, h)

        r_t = torch.sigmoid(x_r + h_r)         # reset gate, (batch, h)
        z_t = torch.sigmoid(x_z + h_z)         # update gate, (batch, h)
        # candidate state; r_t gates only the hidden path, (batch, h)
        n_t = torch.tanh(x_n + r_t * h_n)
        h_t = (1.0 - z_t) * n_t + z_t * h_prev  # (batch, h)
        return h_t
