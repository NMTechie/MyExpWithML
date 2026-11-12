# plotting.py
"""Visualization helpers for Problem 1 figures (class balance, etc.)."""

import os
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
