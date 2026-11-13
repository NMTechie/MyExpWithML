"""Analysis helpers for Step 6 (P2.5): attention heatmaps, and
context-length time/memory profiling.
"""

import os
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from attention import causal_mask


@torch.no_grad()
def get_two_head_attention(
    model: nn.Module,
    sentence: str,
    stoi: Dict[str, int],
    device: torch.device,
    layer_idx: int = 0,
    head_indices: Tuple[int, int] = (0, 1),
) -> Tuple[torch.Tensor, List[str]]:
    """Run `sentence` through one encoder layer's MHA and return its attention
    weights for two heads.

    Returns (weights, tokens): weights has shape (2, T, T); tokens is the
    list of characters in the sentence (for axis labels).
    """
    unknown = [ch for ch in sentence if ch not in stoi]
    if unknown:
        raise ValueError(
            f"sentence contains characters not in the training vocabulary: {set(unknown)}")

    model.eval()
    idx = torch.tensor([[stoi[ch] for ch in sentence]], dtype=torch.long, device=device)  # (1, T)
    seq_len = idx.shape[1]

    x = model.token_embed(idx)              # (1, T, d_model)
    x = model.pos_encoding(x)                # (1, T, d_model)

    layer = model.encoder.layers[layer_idx]
    ln_x = layer.ln1(x)                      # pre-LN input, same as EncoderBlock.forward
    mask = causal_mask(seq_len).to(device)   # (T, T); this model was trained causally
    _, weights = layer.mha(ln_x, mask=mask, return_weights=True)  # weights: (1, H, T, T)

    selected = weights[0, list(head_indices), :, :]  # (2, T, T)
    tokens = list(sentence)
    return selected.cpu(), tokens


def attention_distance(weights_one_head: torch.Tensor) -> float:
    """Mean |i - j| distance, weighted by attention mass. weights_one_head: (T, T)."""
    seq_len = weights_one_head.shape[0]
    positions = torch.arange(seq_len)
    i_idx = positions.view(-1, 1).expand(seq_len, seq_len).float()
    j_idx = positions.view(1, -1).expand(seq_len, seq_len).float()
    distances = (i_idx - j_idx).abs()
    return float((weights_one_head * distances).sum() / weights_one_head.sum(dim=-1).sum())


def plot_attention_heatmaps(
    weights: torch.Tensor,
    tokens: List[str],
    head_indices: Tuple[int, int],
    save_path: str,
) -> None:
    """weights: (2, T, T). Saves a side-by-side heatmap figure to save_path."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, head_weights, head_idx in zip(axes, weights, head_indices):
        im = ax.imshow(head_weights.numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=8, rotation=90)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel("Key position (attended to)")
        ax.set_ylabel("Query position")
        ax.set_title(f"Head {head_idx}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved attention heatmaps to {save_path}")


def plot_heads_ablation(heads_list: Sequence[int], val_losses: Sequence[float], save_path: str) -> None:
    """Line plot of final validation loss vs number of heads (fixed d_model)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(heads_list, val_losses, marker="o")
    ax.set_xlabel("Number of heads (d_model fixed)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Heads ablation: validation loss vs n_heads")
    ax.set_xticks(list(heads_list))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved heads-ablation plot to {save_path}")


def plot_train_loss_comparison(
    causal_history: List[Tuple[int, float]],
    no_mask_history: List[Tuple[int, float]],
    save_path: str,
) -> None:
    """Overlay training-loss curves: causal (masked) vs no-mask ablation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    iters_c, loss_c = zip(*causal_history)
    iters_n, loss_n = zip(*no_mask_history)
    ax.plot(iters_c, loss_c, label="Causal mask (correct)", marker="o")
    ax.plot(iters_n, loss_n, label="No mask (ablation)", marker="s")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training loss (nats)")
    ax.set_title("Training loss: causal mask vs no mask")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved causal-vs-no-mask train loss plot to {save_path}")


def measure_time_memory(
    model: nn.Module,
    context_lengths: Sequence[int],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    n_repeats: int = 5,
) -> Dict[int, Dict[str, float]]:
    """Forward+backward time and peak memory at each context length T.

    Returns {T: {"time_sec": avg_seconds_per_step, "memory_mb": peak_mb_or_nan}}.
    Uses a single model instance sized for the largest T (its positional
    encoding/causal mask are built per-call for the requested T, no re-init
    needed since SinusoidalPositionalEncoding and causal_mask both slice to
    seq_len).
    """
    results: Dict[int, Dict[str, float]] = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for context_len in context_lengths:
        idx = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, context_len), device=device)

        # Warmup (not timed): lets CUDA kernels/caches settle.
        for _ in range(2):
            logits = model(idx)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        start = time.time()
        for _ in range(n_repeats):
            logits = model(idx)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / n_repeats

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mb = float("nan")  # not tracked on CPU

        results[context_len] = {"time_sec": elapsed, "memory_mb": peak_mb}
        print(f"  T={context_len:4d}  time/step={elapsed * 1000:.1f}ms  peak_mem={peak_mb:.1f}MB")

    return results


def plot_time_memory_vs_context(
    results: Dict[int, Dict[str, float]], save_path_time: str, save_path_memory: str
) -> None:
    """Plots measured time/memory vs T, each with an O(T^2) reference curve
    (scaled to match at the smallest T) to visualize the quadratic growth.
    """
    context_lengths = sorted(results.keys())
    times = [results[T]["time_sec"] for T in context_lengths]
    memories = [results[T]["memory_mb"] for T in context_lengths]

    t0, time0 = context_lengths[0], times[0]
    quad_ref_time = [time0 * (T / t0) ** 2 for T in context_lengths]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(context_lengths, times, marker="o", label="Measured time/step")
    ax.plot(context_lengths, quad_ref_time, linestyle="--", label=r"$O(T^2)$ reference")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Time per step (s)")
    ax.set_title("Time vs context length")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path_time), exist_ok=True)
    fig.savefig(save_path_time)
    plt.close(fig)

    if not any(m != m for m in memories):  # NaN check for CPU runs
        m0 = memories[0]
        quad_ref_mem = [m0 * (T / t0) ** 2 for T in context_lengths]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(context_lengths, memories, marker="o", label="Measured peak memory")
        ax.plot(context_lengths, quad_ref_mem, linestyle="--", label=r"$O(T^2)$ reference")
        ax.set_xlabel("Context length T")
        ax.set_ylabel("Peak memory (MB)")
        ax.set_title("Memory vs context length")
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path_memory), exist_ok=True)
        fig.savefig(save_path_memory)
        plt.close(fig)
        print(f"Saved time/memory-vs-context plots to {save_path_time} and {save_path_memory}")
    else:
        print(f"Saved time-vs-context plot to {save_path_time} (memory skipped: running on CPU)")