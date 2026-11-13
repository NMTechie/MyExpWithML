"""Dedup-safe results.csv read/write helpers for Problem 2 runs.

Problem 1's append_result_row() always appends, which duplicates rows if a
cell is re-run (Colab session resume, re-running a single ablation arm,
etc.). upsert_result_row() instead replaces any existing row with the same
identity key before writing, so results.csv never accumulates duplicates
across repeated 'Restart and Run All' executions.
"""

import csv
import os
from typing import Dict, List

RESULTS_COLUMNS = [
    "problem", "model", "seed", "seq_len", "test_acc", "macro_f1",
    "test_mse", "params", "sec_per_epoch", "val_loss_nats",
    "bits_per_char", "notes",
]

# Columns that together identify "the same run". A re-run with the same
# problem/model/seed/seq_len replaces the old row instead of duplicating it.
KEY_COLUMNS = ["problem", "model", "seed", "seq_len"]


def _row_key(row: Dict[str, object]) -> tuple:
    return tuple(str(row.get(col, "")) for col in KEY_COLUMNS)


def read_all_rows(csv_path: str) -> List[Dict[str, str]]:
    """Return all existing rows as a list of dicts; [] if the file doesn't exist yet."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def upsert_result_row(csv_path: str, row: Dict[str, object]) -> None:
    """Write one row to results.csv, replacing any existing row with the same
    (problem, model, seed, seq_len) key in place, so re-running a cell never
    duplicates rows. A genuinely new key is appended at the end.
    """
    existing_rows = read_all_rows(csv_path)
    new_key = _row_key(row)
    full_row = {col: str(row.get(col, "")) for col in RESULTS_COLUMNS}

    replaced = False
    for i, existing_row in enumerate(existing_rows):
        if _row_key(existing_row) == new_key:
            existing_rows[i] = full_row
            replaced = True
            break
    if not replaced:
        existing_rows.append(full_row)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({col: existing_row.get(col, "") for col in RESULTS_COLUMNS})