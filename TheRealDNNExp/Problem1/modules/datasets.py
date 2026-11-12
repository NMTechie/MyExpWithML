# datasets.py
"""HAR (Human Activity Recognition) data pipeline.

Builds the subject-wise train/val/test splits, per-channel
normalization, and PyTorch Dataset/DataLoader objects described
in Problem 1, Task 1 (P1.1) of the assignment.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Fixed channel order required by the assignment.
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_har(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load signals, labels, and subject IDs for one split ('train' or 'test').

    Labels in the raw files are 1-6; they are shifted to 0-5 here so they
    work directly with nn.CrossEntropyLoss.
    """
    try:
        x = stack_har_signals(data_dir, split)
        y = np.loadtxt(os.path.join(data_dir, split,
                       f"y_{split}.txt")).astype(np.int64) - 1
        subjects = np.loadtxt(os.path.join(
            data_dir, split, f"subject_{split}.txt")).astype(np.int64)
        return x, y, subjects
    except Exception as e:
        raise Exception(f"Error loading HAR dataset: {e}")
    finally:
        print(f"Finished loading HAR dataset for split '{split}'")


def stack_har_signals(data_dir: str, split: str) -> np.ndarray:
    """Stack the nine Inertial Signals files into shape (N, 128, 9)."""
    try:
        signal_dir = os.path.join(data_dir, split, "Inertial Signals")
        channel_arrays = []
        for channel in CHANNELS:
            file_path = os.path.join(signal_dir, f"{channel}_{split}.txt")
            channel_arrays.append(np.loadtxt(file_path))  # shape (N, 128)
        # (N, 128, 9)
        return np.stack(channel_arrays, axis=-1).astype(np.float32)
    except Exception as e:
        print(f"Error loading HAR dataset: {e}")
        raise Exception(f"Error loading HAR dataset: {e}")
    finally:
        print(f"Finished stacking HAR signals for split '{split}'")


def create_subject_split(train_subjects: np.ndarray, seed: int, n_val_subjects: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out n_val_subjects distinct subjects from the training pool.

    Returns boolean masks (train_mask, val_mask), each aligned with
    train_subjects, using the exact RNG call specified by the assignment.
    """
    try:
        unique_subjects = np.unique(train_subjects)
        rng = np.random.default_rng(seed)
        val_subject_ids = rng.choice(
            unique_subjects, n_val_subjects, replace=False)
        val_mask = np.isin(train_subjects, val_subject_ids)
        train_mask = ~val_mask
        return train_mask, val_mask
    except Exception as e:
        raise Exception(f"Error creating subject split: {e}")
    finally:
        print(f"Finished creating subject split with seed '{seed}'")


def compute_normalization_stats(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std from training windows only.

    x_train has shape (N, 128, 9); stats are reduced over N and the
    128 time steps, keeping one value per channel, shape (1, 1, 9).
    """
    try:
        mean = np.mean(x_train, axis=(0, 1), keepdims=True)
        std = np.std(x_train, axis=(0, 1), keepdims=True)
        return mean, std
    except Exception as e:
        raise Exception(f"Error in computing normalization stats: {e}")
    finally:
        print("Finished computing normalization stats")


def normalize_data(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize data using the provided mean and std."""
    try:
        return (x - mean) / std
    except Exception as e:
        raise Exception(f"Error in normalizing data: {e}")
    finally:
        print(
            f"Finished normalizing data with provided mean-{mean} and std-{std}")


class HARDataset(Dataset):
    """Wraps normalized HAR windows and labels as a PyTorch Dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()   # (N, 128, 9)
        self.y = torch.from_numpy(y).long()    # (N,)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders. Only the train loader is shuffled."""

    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    except Exception as e:
        raise Exception(f"Error in building dataloaders: {e}")
    finally:
        print("Finished building dataloaders")
