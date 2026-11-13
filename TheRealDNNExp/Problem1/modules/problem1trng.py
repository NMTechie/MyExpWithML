# training.py
"""Training, evaluation, and results-logging utilities for Step 6/7."""

import csv
import os
import time
from typing import Dict, Tuple,Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

RESULTS_COLUMNS = [
    "problem", "model", "seed", "seq_len", "test_acc", "macro_f1",
    "test_mse", "params", "sec_per_epoch", "val_loss_nats",
    "bits_per_char", "notes",
]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Run one training epoch, updating model parameters in place."""
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, predicted_labels) over an entire DataLoader."""
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_true.append(yb.numpy())
        all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train for a fixed number of epochs; return average seconds per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []
    for epoch in range(epochs):
        start = time.time()
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_times.append(time.time() - start)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            y_true, y_pred = evaluate(model, val_loader, device)
            val_acc = accuracy_score(y_true, y_pred)
            print(
                f"  epoch {epoch + 1}/{epochs} - val_acc={val_acc:.4f} - {epoch_times[-1]:.1f}s")

    return float(np.mean(epoch_times))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return test accuracy and macro-F1."""
    return {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def append_result_row(csv_path: str, row: Dict[str, object]) -> None:
    """Append one row to results.csv, writing the header if the file is new."""
    file_exists = os.path.isfile(csv_path)
    full_row = {col: row.get(col, "") for col in RESULTS_COLUMNS}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)



def train_one_epoch_regression(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch for a regression model; return average training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (true_values, predicted_values) over an entire DataLoader."""
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        all_true.append(yb.numpy())
        all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true and predicted values."""
    return float(np.mean((y_true - y_pred) ** 2))


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    track_history: bool = False,
) -> Tuple[float, Dict[str, list]]:
    """Train a regression model for a fixed number of epochs.

    Returns (avg_sec_per_epoch, history). history contains 'train_loss' and
    'val_loss' lists (one entry per epoch) only when track_history=True;
    otherwise both lists stay empty to skip unnecessary per-epoch evaluation.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    epoch_times = []

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch_regression(model, train_loader, optimizer, criterion, device)
        epoch_times.append(time.time() - start)

        is_log_epoch = (epoch + 1) % 10 == 0 or epoch == epochs - 1
        if track_history or is_log_epoch:
            y_true, y_pred = evaluate_regression(model, val_loader, device)
            val_loss = compute_mse(y_true, y_pred)
            if track_history:
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
            if is_log_epoch:
                print(f"  epoch {epoch + 1}/{epochs} - val_mse={val_loss:.4f} - {epoch_times[-1]:.1f}s")

    return float(np.mean(epoch_times)), history


def compute_gradient_norms(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Return the mean ||dL/dh_t|| across a batch, for every time step t."""
    with torch.enable_grad():
        model.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output, _ = model.rnn(x_batch)  # output: (batch, T, hidden_size)
        output.retain_grad()

        last_hidden = output[:, -1, :]            # slice OUTPUT itself, not h_n/state
        pred = model.head(last_hidden).squeeze(-1)

        criterion = nn.MSELoss()
        loss = criterion(pred, y_batch)
        loss.backward()

        

        grad_norms = output.grad.norm(dim=-1)
        mean_grad_norms = grad_norms.mean(dim=0).detach().cpu().numpy()
    return mean_grad_norms



def train_one_epoch_regression(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
) -> float:
    """Run one training epoch for a regression model; return average training loss.

    If max_grad_norm is set, gradients are clipped to that norm before the
    optimizer step (Step 9c gradient-clipping ablation).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    track_history: bool = False,
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, Dict[str, list]]:
    """Train a regression model for a fixed number of epochs.

    Returns (avg_sec_per_epoch, history). history contains 'train_loss' and
    'val_loss' lists (one entry per epoch) only when track_history=True.
    max_grad_norm, if set, enables gradient clipping (Step 9c ablation).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    epoch_times = []

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch_regression(
            model, train_loader, optimizer, criterion, device, max_grad_norm=max_grad_norm
        )
        epoch_times.append(time.time() - start)

        is_log_epoch = (epoch + 1) % 10 == 0 or epoch == epochs - 1
        if track_history or is_log_epoch:
            y_true, y_pred = evaluate_regression(model, val_loader, device)
            val_loss = compute_mse(y_true, y_pred)
            if track_history:
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
            if is_log_epoch:
                print(f"  epoch {epoch + 1}/{epochs} - val_mse={val_loss:.4f} - {epoch_times[-1]:.1f}s")

    return float(np.mean(epoch_times)), history

