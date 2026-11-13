# DNN Assignment 2
- [DNN Assignment 2](#dnn-assignment-2)
  - [Student Details](#student-details)
  - [Configuration](#configuration)
    - [Problem 1](#problem-1)
    - [Problem 2](#problem-2)
  - [Environment](#environment)
  - [Execution Order](#execution-order)
    - [Run problem1.ipynb](#run-problem1ipynb)
    - [Run problem2.ipynb](#run-problem2ipynb)
  - [Results](#results)
  - [Runtime](#runtime)
  - [Assumptions](#assumptions)
  - [Known Limitations](#known-limitations)

## Student Details

Name: Nilesh Ranjan Mitra

Student ID: 2026aj05496

## Configuration

The assignment said to ignore the variants table (the one that assigns hidden size/width/seeds based on the last digit of your student ID) and just pick my own values instead. So these numbers below are not derived from my ID, I just picked something reasonable and used it consistently everywhere.

### Problem 1

Hidden size (h): 64 — used for RNN, LSTM and GRU, same value for all three so the comparison stays fair.

Seeds: 42, 123, 999 — used for every "3 seeds" requirement (HAR training, adding-problem training, hidden-size sweep, the ablation).

### Problem 2

Model dimension (d): 128

Heads: 4 (so d/heads = 32, which satisfies the "d divisible by heads" requirement)

Layers: 4

## Environment

Python: 3.12.10 (this is what I have locally; Problem 1 actually ran on Google Colab's T4 GPU runtime, so the exact PyTorch/NumPy/pandas build there could differ slightly)

PyTorch: *(need to fill this in — forgot to run `!pip show torch` in the Colab session before it recycled. Will grab this from the final clean run and put the exact version here before submitting.)*

NumPy: *(same as above — to be filled from the final run)*

Pandas: *(same as above — to be filled from the final run)*

scikit-learn: used for `accuracy_score`, `f1_score`, and the confusion matrix — version also to be confirmed.

## Execution Order

### Run problem1.ipynb

For the big-5 tasks in Problem 1, I split things up so debugging would be easier instead of having one giant notebook with everything inline. Basically I tried to follow Single Responsibility / Separation of Concerns — `problem1.ipynb` is just the "orchestrator": it calls functions and stitches results together, but the actual logic (data loading, model definitions, training loops, plotting) lives in separate `.py` files under `Problem1/modules/`. The notebook adds that folder to `sys.path` at the very top, so whoever runs this shouldn't need to do anything extra before executing — it works the same way whether it's opened locally or in Colab (there's a small `IN_COLAB` check at the top that mounts Google Drive only if needed).

**On the parameter-count / bias thing (this tripped me up for a bit, so documenting it properly):**

The assignment gives reference parameter counts for `h=64` as RNN=4,736 / LSTM=18,944 / GRU=14,208. When I first built my cells and counted parameters, mine came out *higher* than this (4,800 / 19,200 / 14,400), and I was worried I'd done something wrong. Turns out this is expected, and it comes down to how PyTorch structures its bias terms internally.

PyTorch's `nn.RNN` / `nn.LSTM` / `nn.GRU` each keep **two separate bias vectors per gate** — one for the input path (`bias_ih`) and one for the hidden path (`bias_hh`). But mathematically:

x @ W_ih + b_ih + h @ W_hh + b_hh == x @ W_ih + h @ W_hh + (b_ih + b_hh)


So those two biases could always be combined into a single one without losing any model capacity — it's a redundant extra `h` parameters per gate that PyTorch keeps around anyway (probably for implementation reasons, e.g. cuDNN kernel compatibility, not because it needs them mathematically).

I deliberately built `MyRNNCell` / `MyLSTMCell` / `MyGRUCell` to copy PyTorch's structure exactly (both biases, same gate order) instead of using the "minimal" single-bias version. The reason: it makes the Step 5 weight-copying check dead simple — I just copy `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` straight across from the library layer with no risk of a subtle off-by-something mismatch (this matters especially for GRU, where the reset gate only multiplies the *hidden* bias, not the input one — a detail you genuinely can't collapse into a single bias term without rewriting the gate equations).

So my actual parameter counts (4,800 / 19,200 / 14,400) are slightly higher than the assignment's quoted numbers, and the difference is exactly `gates × h` in each case:

| Model | Gates | My actual count | Assignment's "minimal" count | Difference (= gates × h) |
|---|---|---|---|---|
| RNN  | 1 | 4,800  | 4,736  | 64  |
| LSTM | 4 | 19,200 | 18,944 | 256 |
| GRU  | 3 | 14,400 | 14,208 | 192 |

I derived both formulas by hand:
- Actual (two biases per gate): `gates * (h*d + h*h + 2h)`
- Minimal (one combined bias per gate): `gates * (h*d + h*h + h)`

and confirmed in code that `count_params(my_cell)` exactly matches the "actual" formula above — that was my sanity check that the hand-derivation actually lines up with what the code is doing, not just a coincidence. This verification table (plus the max-error-vs-PyTorch numbers, all ≤9e-8) is saved as notebook output and is what I'm using as the "small verification table" the report asks for.

### Run problem2.ipynb

Same philosophy as `problem1.ipynb`: the notebook is an orchestrator, the actual logic lives in `.py` files under `Problem2/modules/`, and the notebook adds that folder to `sys.path` at the top so it runs the same way locally or on Colab. Problem 2 ended up GPU-heavy enough (the character-level LM especially) that I moved to Colab's T4 runtime partway through, same as Problem 1.

**Step 2 — Attention from scratch (P2.1).** Scaled dot-product attention — `softmax(QK^T/√d_k)·V` — built from matrix operations only, no `nn.MultiheadAttention` anywhere near it. Unit tests: each row of the attention weights sums to 1 (softmax guarantee, but good to actually check); output matches `F.scaled_dot_product_attention` to 1.19e-07 (well inside the 1e-5 the assignment asks for); a causal mask zeroes out every future position, and the masked output matches PyTorch's own `is_causal=True` path to the same 1.19e-07. I also hand-worked a 3-token, d_k=2 toy example on paper (no instructor example was given, so I made one up: Q=K=[[1,0],[0,1],[1,1]], V=[[1,0],[0,1],[0.5,0.5]]) and checked the code against my hand arithmetic — max error 3.19e-05, so the two agree.

**Step 3 — Multi-head attention + pre-LN encoder block (P2.2).** `MultiHeadAttention` wraps the Step 2 attention core: it projects through `W_Q/W_K/W_V/W_O`, reshapes `d_model` into `n_heads × d_k`, runs attention per head, concatenates, and mixes with `W_O`. The encoder block is **pre-LayerNorm** (LayerNorm *before* each sub-layer, not after) — this is the GPT-style convention rather than the original 2017 Transformer's post-LN, and it keeps the residual stream cleaner across the 4-layer stack I'm using everywhere in Problem 2.

This is where I hit the same kind of "wait, my number doesn't match the assignment's" moment as the Problem 1 bias thing. The assignment gives a weights-only formula, `4d² + 2·d·d_ff`, and says that comes to 196,608 for d=128, d_ff=512 — and it does, exactly. But that formula doesn't count biases or LayerNorm parameters, so `sum(p.numel())` on a real `EncoderBlock` comes out higher:

| Component | Count |
|---|---|
| Weights only (4d² + 2·d·d_ff) | 196,608 |
| + MHA biases (4d) | 512 |
| + FFN biases (d_ff + d) | 640 |
| + LayerNorm γ/β (2 norms × 2 × d) | 512 |
| **Actual `sum(p.numel())` per block** | **198,272** |

Same lesson as the RNN/LSTM/GRU bias gap in Problem 1: the "clean" formula in the assignment is deliberately the weights-only part, and the gap (1,664 here) is fully accounted for once you add back biases and LayerNorm — I printed both numbers side by side in the notebook as the reconciliation check. The full 4-layer stack (my config: d=128, heads=4, d_ff=512) comes to 793,344 trainable parameters.

**Step 4 — HAR encoder classifier + positional-encoding ablation (P2.3).** Reused Problem 1's HAR data pipeline, subject-wise split, and generic `train_model`/`evaluate` loop completely unchanged — I wanted architecture to be the *only* thing different between this and the Problem 1 comparison. Instead of embedding all 128 time steps individually, I patchified each window into 16 patches of 8 steps (72 raw features per patch, per the assignment's "or each patch of 8 steps" option), linearly projected each patch to d=128, prepended a learned CLS token, and classified off the CLS token's final state — 17 tokens total, bidirectional (no causal mask, since classification should see the whole window).

Ablated positional information (none / sinusoidal / learned), 3 seeds each, same 30-epoch budget as Problem 1's HAR runs:

| Positional encoding | Test acc | Macro-F1 | Std (acc) | Params | Sec/epoch |
|---|---|---|---|---|---|
| None | 0.9216 | 0.9211 | 0.0040 | 803,590 | 1.33 |
| Sinusoidal | 0.9102 | 0.9096 | 0.0057 | 803,590 | 1.34 |
| Learned | 0.9077 | 0.9086 | 0.0155 | 805,766 | 1.36 |
| **Best Problem 1 (GRU, seed 999)** | **0.9226** | **0.9228** | — | **14,790** | **0.257** |

The "none" result genuinely surprised me — I expected sinusoidal to at least tie it. My best explanation: with only 17 tokens and full bidirectional attention, there's very little order-ambiguity left for the model to resolve; the patches' own content probably already disambiguates their position well enough. The learned-PE variant also has noticeably higher seed-to-seed variance (std 0.0155 vs 0.0040 for none) — with only 30 epochs and ~2,176 extra parameters to learn from scratch, that part of the model seems to be the least reliably trained. And the headline comparison: the best transformer here is basically tied with the best Problem 1 GRU on accuracy, but needs **54× more parameters** and **~5.2× more time per epoch** — for short, low-dimensional sensor windows like this, the small recurrent model is just a much better accuracy-per-parameter deal.

**Step 5 — Character-level causal LM on Tiny Shakespeare (P2.4).** Tiny Shakespeare is 1,115,394 characters, confirmed exactly 65 distinct characters in code. Split 90/10 by position (not shuffled — shuffling a language-modeling stream would leak future context into earlier windows), giving 1,003,854 train / 111,540 val characters. Model: causal 4-layer encoder, d=128, 4 heads, context window 128, sinusoidal PE, 810,049 total parameters, trained 5,000 iterations with AdamW (lr=3e-4).

Validation loss fell smoothly and landed at **1.6024 nats** (2.3118 bits/char) — right in the assignment's 1.5–1.8 sanity range, and well below the uniform-guess baseline of ln(65) = 4.1744 nats:

| Iteration | Val loss (nats) | Bits/char |
|---|---|---|
| 500 | 2.1698 | 3.1303 |
| 1,500 | 1.8497 | 2.6686 |
| 3,000 | 1.6828 | 2.4278 |
| 5,000 | 1.6024 | 2.3118 |

Generated 500 characters at three temperatures (top-k=40 — the assignment doesn't pin down k, so I picked a standard value and noted it as an assumption):
- **T=0.2**: grammatically plausible but stuck in loops on filler words ("the the state... shall be the shall be") — low temperature sharpens the distribution so much the model keeps picking its favorite continuation over and over.
- **T=0.8**: the best balance by far — correct play-script structure (character names in caps followed by a colon, e.g. "HENRY BOLINGBROKE:"), varied vocabulary, reasonable rhythm.
- **T=1.5**: noisy and mostly incoherent (invented words like "muddital", "brunkshic"), though it still loosely keeps the play-script shape.

**Step 6 — Analysis and ablations (P2.5).**
- **(a) Attention heatmaps** — two heads, layer 0, on "To be, or not to be: that is the question" (43 characters). Both heads show mostly local/near-diagonal attention plus a bright column at position 0 (every later position gives some weight back to the first token — a common "attention sink" pattern in causal transformers). Mean attention-weighted distance |i−j|: Head 0 = 2.39, Head 1 = 2.20 — Head 0 slightly more long-range, but neither is doing dramatic long-distance attention yet, which makes sense for an early layer.
- **(b) Heads = 1/4/8** at fixed d=128, matched 2,000-iteration budget: val loss 1.7592 / 1.7638 / 1.7837 nats respectively, with sec/500-iters climbing from 14.1s to 21.0s as heads increase. Important catch here: all three configurations have the *exact same* parameter count (810,049) — more heads doesn't add capacity, it just changes how the fixed-size `W_Q/K/V/O` projections get split into smaller attention sub-spaces. Fewer heads trained faster and reached marginally lower loss in this short budget, though I'd want a longer run before calling that a general result rather than a budget artifact.
- **(c) Removing the causal mask** (same 2,000-iter budget): causal val loss stayed at 1.7638 nats; no-mask collapsed to 0.0185 nats. That number looked suspiciously good, and it is — without the mask, position t can attend directly to position t+1, which is literally its own training target. The model learned to copy the future instead of predicting it, a shortcut that only exists during training (the whole sequence is visible at once) and vanishes at generation time (there is no future token yet). Confirmed by generation: the causal model produced readable play-script-like text; the no-mask model produced an endless repeat of the single character "O". Good concrete reminder that a low training loss doesn't automatically mean a working model.
- **(d) Time/memory vs context length** (T = 64/128/256/512, batch=16): time-per-step went 16.1→15.3→20.8→58.2 ms, peak memory went 110.6→160.5→297.7→751.9 MB. Going from T=64 to T=512 is an 8× increase in context length, but only ~3.6× more time and ~6.8× more memory — both far below the 64× (8²) a pure O(n²·d) attention cost would predict. At these context lengths and this model size, the costs that scale only linearly with T (embeddings, the W_Q/K/V/O and feed-forward projections) still make up most of the total; the quadratic attention term is there (the growth is clearly super-linear, especially T=256→512) but it would only dominate outright at much larger context lengths than tested here.

**One engineering decision worth calling out:** Problem 1's `append_result_row()` always appends to `results.csv`, which is fine if you only ever run each cell once, but Problem 2 involves re-running individual ablation arms (fixing a bug, bumping iteration counts, etc.) on Colab, and a plain append would have left duplicate rows every time I re-ran something. So Problem 2 uses its own `upsert_result_row()` (in `Problem2/modules/results_io.py`) that replaces any existing row with the same `(problem, model, seed, seq_len)` key instead of blindly appending — verified with a small self-test (write two keys, re-write one with a different value, confirm the row count stays at 2 and the value actually changed) before trusting it on real runs.

## Results

- Trained-model metrics: `Results/results.csv` — one row per run, 66 rows total for Problem 1 (9 HAR + 36 adding-problem + 18 hidden-size sweep + 3 gradient-clipping ablation runs).
- Figures: `Problem1/figures/`
  - `class_balance/` — train/test/per-seed class distribution plots
  - `confusion_matrix/` — one per (model, seed) combo, 9 total; the one actually used in the report is the best-macro-F1 run (GRU, seed 999)
  - `mse_vs_T.png`, `learning_curves_T200.png` — adding-problem results
  - `gradient_norms_T100.png` — before/after training gradient analysis
  - `hidden_size_sweep.png` — LSTM/GRU across h = {16, 32, 64, 128}
  - `gradient_clipping_ablation_T200.png` — my individual ablation (see Assumptions)
- Checkpoints: `Problem1/checkpoints/` — every trained model's weights, kept locally for re-generating plots without retraining; **not** included in the final submission zip per the assignment's size/content rules.

- Problem 2 adds further rows to the same `Results/results.csv`: 9 HAR-encoder runs (P2.3, 3 positional-encoding arms × 3 seeds), 1 main char-LM run (P2.4), 3 heads-ablation runs (P2.5b), 2 mask-ablation runs (P2.5c), and 4 context-length timing rows (P2.5d) — 19 additional rows, all written through the dedup-safe `upsert_result_row()` rather than Problem 1's plain-append logger.
- `Results/generated_samples.txt` — the three required 500-character samples at temperature 0.2 / 0.8 / 1.5, top-k=40.
- `Results/no_mask_generation_demo.txt` — my own extra side-by-side comparison (causal vs no-mask generation) backing up the Step 6(c) ablation; not one of the required submission files, just evidence for that section of the report.
- Figures: `Problem2/figures/`
  - `attention_heatmaps/two_heads_sentence.png` — the two-head attention heatmap for Step 6(a)
  - `heads_ablation.png` — validation loss vs. heads (1/4/8)
  - `causal_vs_nomask_train_loss.png` — training-loss curves, causal vs no-mask
  - `context_length_time.png`, `context_length_memory.png` — measured time/memory vs. context length, each with an O(T²) reference curve overlaid
- Checkpoints: `Problem2/checkpoints/` — HAR-encoder checkpoints per (positional-encoding arm, seed), plus the main char-LM checkpoint; same as Problem 1, kept locally and not included in the submission zip.

## Runtime

**HAR:** Data source is [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones), auto-downloaded and extracted by the notebook on first run (no manual download needed). 9 training runs (3 models × 3 seeds), 30 epochs each — a few minutes total on a T4 GPU.

**tiny_shakespere:** <<need to fill>>

**Adding Problem:** 36 runs (3 models × 4 sequence lengths × 3 seeds). Epoch budget scales with T (60/80/120/150 epochs for T=50/100/200/400) rather than a flat number for every T — longer sequences seemed to need more updates to actually converge, especially the gated models. This took noticeably longer than the HAR runs.

**Hidden-size sweep + ablation:** 18 + 3 = 21 more runs, each individually small/fast since these are still tiny HAR/adding-problem models.

**Transformer (Problem 2):** All of Problem 2 ran on Colab's T4. The HAR-encoder ablation (9 runs, 30 epochs each at ~1.33–1.36s/epoch) took roughly 6–7 minutes total. The main character-LM run (5,000 iterations at ~17.6s per 500-iteration block) took about 3 minutes. The heads ablation (3 runs × 2,000 iterations) and the no-mask ablation (2 runs × 2,000 iterations) added roughly 3.5 and 2 minutes respectively. The context-length timing/memory sweep is just a handful of forward/backward passes per context length and finished in seconds. Rough total for Problem 2: somewhere around 15–20 minutes on a T4, noticeably faster than Problem 1's RNN/LSTM/GRU sweep despite the transformer models having far more parameters — short training budgets (30 epochs / a few thousand iterations) rather than long ones, and a lot of the Problem 2 runs are small ablations rather than full sweeps.

Rough total for everything done so far in Problem 1: somewhere around 45–60 minutes on a Colab T4, across all the steps combined.

## Assumptions

Things the assignment didn't pin down exactly, so I made a call and I'm noting it here so it's not a mystery later:

1. **Hidden size / model width / seeds:** chosen myself (h=64, d=128, seeds 42/123/999) per the Prof's instruction to ignore the variants table.
2. **Adding-problem epoch budget:** scaled by T instead of using one flat epoch count for all four lengths — `{50: 60, 100: 80, 200: 120, 400: 150}`. Reasoning: a budget sized for T=50 would badly undertrain T=400, and a budget sized for T=400 would waste a lot of time re-running T=50 way longer than it needs.
3. **Gradient-norm analysis batch size:** used a fixed batch of 256 test sequences at T=100 (not the full 1,000-sequence test set) for computing `||∂L/∂h_t||`. The rubric doesn't specify a sample size here — 256 felt like a reasonable middle ground between a smooth/stable averaged curve and not being wasteful about it.
4. **Individual ablation choice:** my student ID ends in 6, which is even, so per the (original, ID-based) ablation rule that's gradient clipping (norm 1.0 vs. none) for the vanilla RNN at T=200. I double-checked this still applies even after the Prof's instruction to self-choose h/d/seeds, since that instruction only explicitly mentioned ignoring the variants table and the h/width/seeds FAQ entry — not the ablation-selection rule, which lives in the same "Individualisation" section but wasn't named directly. Went with the narrower reading (keep the original ablation assignment) since that was the safer interpretation.
5. **"Best HAR model" selection metric:** used macro-F1 rather than raw accuracy to pick the "best" run for the confusion-matrix figure, since macro-F1 is the fairer metric across 6 somewhat-imbalanced activity classes. This is called out explicitly in the report.
6. **Positional-encoding ablation epoch budget:** kept at 30 epochs for all three variants (none/sinusoidal/learned), matching Problem 1's HAR budget exactly, so the Problem 1-vs-Problem 2 comparison stays apples-to-apples.
7. **Top-k for generation:** the assignment says "top-k sampling" without specifying k; I used top-k=40 for all three temperature samples.
8. **Heads (1/4/8) and no-mask ablations use a reduced, matched 2,000-iteration budget**, not the main P2.4 run's 5,000 — this keeps the ablation arms comparable to each other, but their absolute validation-loss numbers aren't directly comparable to the headline P2.4 result (which used the full 5,000-iteration budget).
9. **Tiny Shakespeare train/val split:** a contiguous 90/10 split by character position, not a random shuffle — shuffling would leak future characters into earlier training windows, which doesn't make sense for a next-character-prediction task.

## Known Limitations

- **Gradient clipping didn't actually help the vanilla RNN at T=200** — clipped-run MSE (0.173 ± 0.008) came out essentially the same as the no-clipping baseline (0.171 ± 0.005), both stuck near the 0.167 constant-prediction floor. My read on this: the RNN's problem here is genuine vanishing gradient (signal disappearing before it reaches the early marker), not gradient explosion, so clipping the gradient norm doesn't address the actual bottleneck.
- **All three architectures show essentially zero gradient at t=1 by the time you're 100 steps back** (both before and after training) — so even LSTM/GRU aren't literally preserving gradient all the way back; they just seem to still learn the task through some other mechanism (maybe shorter effective dependencies, or very well-tuned local dynamics), which is a bit different from the "gates solve vanishing gradients" story I expected going in.
- **LSTM starts to struggle at T=400** (only 1 of 3 seeds solved, vs. 3/3 for GRU at the same length) — so even among the gated models, GRU came out more consistently robust as sequence length grows, at least in this setup.
- **The HAR transformer encoder ties the best Problem 1 GRU on accuracy (0.9216 vs 0.9228) but costs 54× more parameters and ~5.2× more time per epoch** — for short, low-dimensional sensor windows like HAR, the small recurrent model is clearly the better accuracy-per-parameter trade; the transformer isn't "wrong," it's just the wrong tool for data this small and this short.
- **The heads ablation (1/4/8) used a short, matched 2,000-iteration budget for a fair comparison across arms**, so the "fewer heads did slightly better" result should be read as a short-budget observation, not a general claim — a longer run might change the ordering once each configuration has had time to actually specialize its heads.
- **The no-mask ablation's near-zero validation loss (0.0185 nats) is not evidence of a better language model** — it's the model exploiting a shortcut (attending directly to its own training target) that only exists because the mask is missing; the generation test confirms this cleanly, since the no-mask model produces degenerate output ("OOOO...") while the causal model doesn't.
- **Measured time/memory growth with context length is clearly super-linear but well below the naive O(T²) curve in the tested range (T=64–512)** — at this model size (d=128, 4 layers), linear-in-T costs still make up a large share of the total; the quadratic attention term would likely dominate only at much larger context lengths than were tested here.
- The trained models reside in the Gdrive as those are generated in the Colab and I have not downloaded.