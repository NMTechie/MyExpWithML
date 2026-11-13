# plotting.py
"""Visualization helpers for Problem 1 figures (class balance, etc.)."""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from typing import List, Sequence,Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_activity_labels(data_dir: str) -> List[str]:
    """Read activity_labels.txt and return class names ordered 0-5.

    The raw file maps labels 1-6 to names (e.g. "1 WALKING"); index 0
    here corresponds to raw label 1 since datasets.load_har() shifts
    labels down by one for nn.CrossEntropyLoss.
    """
    try:
        labels_path = os.path.join(data_dir, "activity_labels.txt")
        names = []
        with open(labels_path) as f:
            for line in f:
                _, name = line.strip().split(maxsplit=1)
                names.append(name)
        return names
    except Exception as e:
        raise Exception(f"Error loading activity labels: {e}")
    finally:
        print("Finished loading activity labels")


def plot_class_balance(
    y: np.ndarray,
    class_names: Sequence[str],
    title: str,
    save_path: str,
) -> None:
    """Bar chart of label counts for one split, saved to save_path."""
    try:
        counts = np.bincount(y, minlength=len(class_names))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(class_names, counts)
        ax.set_xlabel("Activity")
        ax.set_ylabel("Window count")
        ax.set_title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        raise Exception(f"Error plotting class balance: {e}")
    finally:
        print(f"Finished plotting class balance to '{save_path}'")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    save_path: str,
) -> None:
    """Confusion matrix heatmap for one run, saved to save_path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Finished plotting confusion matrix to '{save_path}'")





def plot_learning_curves(
    histories: Dict[str, Dict[str, list]],
    title: str,
    save_path: str,
) -> None:
    """Train-loss and val-MSE curves for multiple models, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_type, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=model_type.upper())
        axes[1].plot(epochs, history["val_loss"], label=model_type.upper())

    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation MSE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_mse_vs_t(summary_df: pd.DataFrame, title: str, save_path: str) -> None:
    """MSE vs T, one line per model, with std error bars.

    summary_df must have columns: model, seq_len, mse_mean, mse_std.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_type in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_type].sort_values("seq_len")
        ax.errorbar(
            sub["seq_len"], sub["mse_mean"], yerr=sub["mse_std"],
            marker="o", capsize=4, label=model_type.upper(),
        )
    ax.axhline(0.167, color="gray", linestyle="--", label="Constant-prediction baseline (0.167)")
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Test MSE")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_gradient_norms(
    before_norms: Dict[str, np.ndarray],
    after_norms: Dict[str, np.ndarray],
    title: str,
    save_path: str,
) -> None:
    """Combined before/after ||dL/dh_t|| plot, log-scale y-axis, one panel each."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for model_type, norms in before_norms.items():
        axes[0].plot(range(1, len(norms) + 1), norms, label=model_type.upper())
    for model_type, norms in after_norms.items():
        axes[1].plot(range(1, len(norms) + 1), norms, label=model_type.upper())

    axes[0].set_title("Before Training")
    axes[1].set_title("After Training")
    for ax in axes:
        ax.set_xlabel("Time step t")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("||dL/dh_t|| (log scale)")

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_hidden_size_sweep(summary_df: pd.DataFrame, title: str, save_path: str) -> None:
    """Macro-F1 vs hidden size, one line per model, with std error bars.

    summary_df must have columns: model, hidden_size, f1_mean, f1_std.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_type in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_type].sort_values("hidden_size")
        ax.errorbar(
            sub["hidden_size"], sub["f1_mean"], yerr=sub["f1_std"],
            marker="o", capsize=4, label=model_type.upper(),
        )
    ax.set_xlabel("Hidden size (h)")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_gradient_clipping_ablation(
    histories: Dict[str, Dict[str, list]],
    summary_df: pd.DataFrame,
    title: str,
    save_path: str,
) -> None:
    """Two-panel ablation figure: training-loss curves (left), test-MSE comparison (right).

    histories keys: 'none', 'clipped'. summary_df must have columns: clip, mse_mean, mse_std.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for clip_label, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=clip_label)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()

    axes[1].bar(summary_df["clip"], summary_df["mse_mean"], yerr=summary_df["mse_std"], capsize=6)
    axes[1].set_title("Test MSE (mean +/- std, 3 seeds)")
    axes[1].set_ylabel("Test MSE")

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

