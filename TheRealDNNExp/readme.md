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

Still working through this one. So far I've finished:
- Scaled dot-product attention from scratch, with unit tests (softmax rows sum to 1, matches `F.scaled_dot_product_attention` within 1e-5, causal mask blocks future positions) plus a hand-worked 3-token example (no instructor example was given for this, so I made up my own small Q/K/V with d_k=2 and checked the hand arithmetic against the code output).
- Multi-head attention, the pre-LayerNorm encoder block, and sinusoidal positional encoding are next — still in progress at time of writing this README.

Will update this section once Tasks 3 and 4 (HAR encoder classifier, character-level LM) are done.

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

## Runtime

**HAR:** Data source is [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones), auto-downloaded and extracted by the notebook on first run (no manual download needed). 9 training runs (3 models × 3 seeds), 30 epochs each — a few minutes total on a T4 GPU.

**Adding Problem:** 36 runs (3 models × 4 sequence lengths × 3 seeds). Epoch budget scales with T (60/80/120/150 epochs for T=50/100/200/400) rather than a flat number for every T — longer sequences seemed to need more updates to actually converge, especially the gated models. This took noticeably longer than the HAR runs.

**Hidden-size sweep + ablation:** 18 + 3 = 21 more runs, each individually small/fast since these are still tiny HAR/adding-problem models.

**Transformer (Problem 2):** Not run yet — will fill this in once Tasks 3/4 are done (those are the GPU-heavy ones: HAR encoder classifier and the character-level language model on Tiny Shakespeare).

Rough total for everything done so far in Problem 1: somewhere around 45–60 minutes on a Colab T4, across all the steps combined.

## Assumptions

Things the assignment didn't pin down exactly, so I made a call and I'm noting it here so it's not a mystery later:

1. **Hidden size / model width / seeds:** chosen myself (h=64, d=128, seeds 42/123/999) per the Prof's instruction to ignore the variants table.
2. **Adding-problem epoch budget:** scaled by T instead of using one flat epoch count for all four lengths — `{50: 60, 100: 80, 200: 120, 400: 150}`. Reasoning: a budget sized for T=50 would badly undertrain T=400, and a budget sized for T=400 would waste a lot of time re-running T=50 way longer than it needs.
3. **Gradient-norm analysis batch size:** used a fixed batch of 256 test sequences at T=100 (not the full 1,000-sequence test set) for computing `||∂L/∂h_t||`. The rubric doesn't specify a sample size here — 256 felt like a reasonable middle ground between a smooth/stable averaged curve and not being wasteful about it.
4. **Individual ablation choice:** my student ID ends in 6, which is even, so per the (original, ID-based) ablation rule that's gradient clipping (norm 1.0 vs. none) for the vanilla RNN at T=200. I double-checked this still applies even after the Prof's instruction to self-choose h/d/seeds, since that instruction only explicitly mentioned ignoring the variants table and the h/width/seeds FAQ entry — not the ablation-selection rule, which lives in the same "Individualisation" section but wasn't named directly. Went with the narrower reading (keep the original ablation assignment) since that was the safer interpretation.
5. **"Best HAR model" selection metric:** used macro-F1 rather than raw accuracy to pick the "best" run for the confusion-matrix figure, since macro-F1 is the fairer metric across 6 somewhat-imbalanced activity classes. This is called out explicitly in the report.

## Known Limitations

- **Gradient clipping didn't actually help the vanilla RNN at T=200** — clipped-run MSE (0.173 ± 0.008) came out essentially the same as the no-clipping baseline (0.171 ± 0.005), both stuck near the 0.167 constant-prediction floor. My read on this: the RNN's problem here is genuine vanishing gradient (signal disappearing before it reaches the early marker), not gradient explosion, so clipping the gradient norm doesn't address the actual bottleneck.
- **All three architectures show essentially zero gradient at t=1 by the time you're 100 steps back** (both before and after training) — so even LSTM/GRU aren't literally preserving gradient all the way back; they just seem to still learn the task through some other mechanism (maybe shorter effective dependencies, or very well-tuned local dynamics), which is a bit different from the "gates solve vanishing gradients" story I expected going in.
- **LSTM starts to struggle at T=400** (only 1 of 3 seeds solved, vs. 3/3 for GRU at the same length) — so even among the gated models, GRU came out more consistently robust as sequence length grows, at least in this setup.
- **Problem 2 is incomplete as of this writing** — only attention-from-scratch (P2.1) is fully done. Multi-head attention, the encoder block, the HAR encoder classifier, and the character-level language model are still in progress.
- The trained models reside in the Gdrive as those are generated in the Colab and I have not downloaded.