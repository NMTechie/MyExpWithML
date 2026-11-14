# DNN Assignment 2
- [DNN Assignment 2](#dnn-assignment-2)
  - [Student Details](#student-details)
  - [Configuration](#configuration)
    - [Problem 1](#problem-1)
    - [Problem 2](#problem-2)
  - [Environment](#environment)
  - [Execution Order](#execution-order)
    - [Run problem1.ipynb](#run-problem1ipynb)
        - [Project Structure](#project-structure)
        - [Parameter Counts and Bias Terms](#parameter-counts-and-bias-terms)
    - [Run problem2.ipynb](#run-problem2ipynb)
        - [Project Structure](#project-structure-1)
        - [Step 2 – Attention From Scratch (P2.1)](#step-2--attention-from-scratch-p21)
        - [Step 3 – Multi-Head Attention and Encoder Block (P2.2)](#step-3--multi-head-attention-and-encoder-block-p22)
        - [Step 4 – HAR Transformer Classifier and Positional-Encoding Ablation (P2.3)](#step-4--har-transformer-classifier-and-positional-encoding-ablation-p23)
        - [Step 5 – Character-Level Language Model on Tiny Shakespeare (P2.4)](#step-5--character-level-language-model-on-tiny-shakespeare-p24)
          - [Temperature = 0.2](#temperature--02)
          - [Temperature = 0.8](#temperature--08)
          - [Temperature = 1.5](#temperature--15)
        - [Step 6 – Analysis and Ablations (P2.5)](#step-6--analysis-and-ablations-p25)
          - [Attention Heatmaps](#attention-heatmaps)
          - [Number of Attention Heads](#number-of-attention-heads)
          - [Removing the Causal Mask](#removing-the-causal-mask)
          - [Context-Length Scaling](#context-length-scaling)
        - [Engineering Note](#engineering-note)
  - [Results](#results)
  - [Runtime](#runtime)
  - [Assumptions](#assumptions)
    - [1. Hidden Size, Model Dimension, and Random Seeds](#1-hidden-size-model-dimension-and-random-seeds)
    - [2. Adding-Problem Training Budget](#2-adding-problem-training-budget)
    - [3. Gradient-Norm Analysis Sample Size](#3-gradient-norm-analysis-sample-size)
    - [4. Individual Ablation Selection](#4-individual-ablation-selection)
    - [5. Selection of the Best HAR Model](#5-selection-of-the-best-har-model)
    - [6. Positional-Encoding Ablation Budget](#6-positional-encoding-ablation-budget)
    - [7. Top-k Sampling Configuration](#7-top-k-sampling-configuration)
    - [8. Ablation Training Budget for Transformer Experiments](#8-ablation-training-budget-for-transformer-experiments)
    - [9. Tiny Shakespeare Train/Validation Split](#9-tiny-shakespeare-trainvalidation-split)
  - [Known Limitations](#known-limitations)
    - [1. Gradient Clipping Had Limited Impact on RNN Performance](#1-gradient-clipping-had-limited-impact-on-rnn-performance)
    - [2. Long-Range Gradients Become Extremely Small for All Models](#2-long-range-gradients-become-extremely-small-for-all-models)
    - [3. LSTM Performance Declined at the Longest Sequence Length](#3-lstm-performance-declined-at-the-longest-sequence-length)
    - [4. Transformer Performance Came at a Much Higher Cost](#4-transformer-performance-came-at-a-much-higher-cost)
    - [5. Head-Count Ablation Used a Reduced Training Budget](#5-head-count-ablation-used-a-reduced-training-budget)
    - [6. Very Low Loss Without a Causal Mask Was Misleading](#6-very-low-loss-without-a-causal-mask-was-misleading)
    - [7. Context-Length Scaling Was Not Fully Quadratic](#7-context-length-scaling-was-not-fully-quadratic)
    - [8. Model Checkpoints Are Not Included](#8-model-checkpoints-are-not-included)
    - [9. Generation of requirements.txt](#9-generation-of-requirementstxt)

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

Python: 3.12.10 (this is what I have locally); While ran on Google Colab's T4 GPU runtime, the version is 3.13.16)

PyTorch: 2.14.1+cu130

NumPy: 2.4.6

Pandas: 2.2.3

Matplotlib: 3.10.0

Seaborn: 0.13.2

scikit-learn: 1.6.1

## Execution Order

### Run problem1.ipynb

##### Project Structure

Since I am still learning Deep Learning and working through multiple experiments, I found it easier to organize the code into smaller modules rather than putting everything into one large notebook. This made debugging and testing much more manageable.

The main notebook, `problem1.ipynb`, acts as the entry point for running the experiments. Most of the actual implementation is kept in separate Python files under `Problem1/modules/`, such as data loading, model definitions, training functions, evaluation logic, and plotting utilities. This helped me keep related functionality in one place and made the notebook easier to read.

To make execution straightforward, the notebook adds the module folder to `sys.path` at startup. I also included a small `IN_COLAB` check so that the same notebook can be run either locally or in Google Colab without requiring any manual code changes.

---

##### Parameter Counts and Bias Terms

One interesting thing I discovered while implementing the RNN, LSTM, and GRU cells was that my parameter counts did not initially match the values shown in the assignment. For a hidden size of `h = 64`, the assignment lists the following parameter counts:

| Model | Assignment Count |
|---------|---------:|
| RNN | 4,736 |
| LSTM | 18,944 |
| GRU | 14,208 |

However, when I calculated the parameters from my implementations and from PyTorch's built-in layers, I obtained:

| Model | PyTorch Count |
|---------|---------:|
| RNN | 4,800 |
| LSTM | 19,200 |
| GRU | 14,400 |

At first, I assumed I had made a mistake. After investigating further, I learned that PyTorch stores two bias vectors for each gate: one associated with the input transformation (`bias_ih`) and another associated with the hidden-state transformation (`bias_hh`).

Mathematically, these two bias terms can often be combined into a single bias because:

$$
xW_{ih} + b_{ih} + hW_{hh} + b_{hh}
=
xW_{ih} + hW_{hh} + (b_{ih}+b_{hh})
$$

This means the extra bias vector does not necessarily increase the expressive power of the model, but PyTorch keeps both bias terms as part of its implementation.

To make the verification task easier, I followed PyTorch's parameter layout exactly in my custom `MyRNNCell`, `MyLSTMCell`, and `MyGRUCell` implementations. This allowed me to directly copy `weight_ih`, `weight_hh`, `bias_ih`, and `bias_hh` from the corresponding PyTorch layers and compare outputs without introducing additional conversion logic.

As a result, my parameter counts match PyTorch's values exactly. The difference between the assignment's theoretical counts and PyTorch's reported counts comes from the additional bias vectors maintained internally by PyTorch for each gate.

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

##### Project Structure

For Problem 2, I followed the same approach that I used in Problem 1. The main notebook, `problem2.ipynb`, acts as the entry point for all experiments. The actual implementation is organised under `Problem2/modules/`, including attention layers, Transformer components, training utilities, evaluation functions, plotting helpers, and result management. This made the code easier to understand, test, and debug.As the language-modeling experiments became more computationally intensive, I eventually moved to a Colab T4 GPU runtime, similar to the training experiments in Problem 1.

---

##### Step 2 – Attention From Scratch (P2.1)

The first task was to implement scaled dot-product attention from scratch using only tensor and matrix operations. The implementation follows the standard equation:

$$
\text{Attention}(Q,K,V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

No use was made of `nn.MultiheadAttention` or any Transformer helper classes.

To verify correctness, I added several unit tests:

- Verified that each row of the attention matrix sums to 1.
- Compared the output against `torch.nn.functional.scaled_dot_product_attention`.
- Verified that the causal mask completely blocks access to future positions.
- Compared the masked output against PyTorch's own causal attention implementation.

The output matched PyTorch with a maximum error of approximately `1.19e-07`, which is significantly better than the assignment requirement of `1e-5`.

Since no instructor-provided example was supplied, I created a simple three-token example using small Q, K, and V matrices and manually calculated the attention output. The output from my implementation matched the hand calculations with a maximum difference of approximately `3.19e-05`, providing additional confidence that the implementation was correct.

---

##### Step 3 – Multi-Head Attention and Encoder Block (P2.2)

After completing scaled dot-product attention, I built a full multi-head attention module. The implementation projects the input through the query, key, value, and output matrices (`W_Q`, `W_K`, `W_V`, and `W_O`), splits the representation across multiple heads, applies attention independently within each head, concatenates the results, and then produces the final output projection.

The encoder block was implemented using a pre-LayerNorm architecture, where Layer Normalization is applied before both the attention and feed-forward sublayers. Residual connections and dropout were also included.

During this task I encountered a situation very similar to the parameter-count discrepancy observed in Problem 1.

The assignment provides the following weight-only formula:

$$
4d^2 + 2dd_{ff}
$$

For:

- \(d = 128\)
- \(d_{ff} = 512\)

the expected count is:

| Component | Parameter Count |
|------------|------------:|
| Weights only | 196,608 |

However, when I calculated the total number of trainable parameters using:

```python
sum(p.numel() for p in model.parameters())
```

the value was larger.

After reviewing the implementation, I realised that the assignment formula counts only the weight matrices and excludes:

- Attention biases
- Feed-forward biases
- LayerNorm scale parameters
- LayerNorm shift parameters

Accounting for all of these components produces the actual parameter count observed in code:

| Component | Count |
|------------|------------:|
| Weights only | 196,608 |
| Multi-head attention biases | 512 |
| Feed-forward biases | 640 |
| LayerNorm parameters | 512 |
| **Total per encoder block** | **198,272** |

This is kind of similar bias vector explanation of Problem 1: theoretical parameter formulas often focus on the main weight matrices, while framework-reported parameter counts include additional trainable components such as biases and normalization parameters.

Using the chosen configuration (`d=128`, `heads=4`, `d_ff=512`), the complete four-layer Transformer contains **793,344 trainable parameters**.

---

##### Step 4 – HAR Transformer Classifier and Positional-Encoding Ablation (P2.3)

For this task, I reused the same HAR preprocessing pipeline, data splits, normalization strategy, and training framework that were used in Problem 1. This ensured that any differences in performance were due to the model architecture rather than changes in data processing.

Rather than processing all 128 time steps individually, I followed the assignment's patch-based option. Each HAR window was divided into 16 patches of 8 time steps each. Every patch was projected into a 128-dimensional representation, and a learned CLS token was prepended to the sequence. The classifier then used the final CLS embedding for prediction.

To understand the contribution of positional information, I evaluated three configurations:

| Positional Encoding | Test Accuracy | Macro-F1 | Accuracy Std | Parameters | Sec/Epoch |
|---|---:|---:|---:|---:|---:|
| None | 0.9216 | 0.9211 | 0.0040 | 803,590 | 1.33 |
| Sinusoidal | 0.9102 | 0.9096 | 0.0057 | 803,590 | 1.34 |
| Learned | 0.9077 | 0.9086 | 0.0155 | 805,766 | 1.36 |
| **Best Problem 1 GRU** | **0.9226** | **0.9228** | — | **14,790** | **0.257** |

One result that genuinely surprised me was that the model without positional encoding performed best. My initial expectation was that sinusoidal encoding would improve performance. A possible explanation is that with only 17 total tokens (16 patches plus CLS), patch contents may already carry enough information for the model to identify relative positions.

The learned positional-encoding variant also displayed noticeably larger variation across random seeds. This suggests that learning positional embeddings from scratch within the available training budget may be less stable than relying directly on patch content.

The most interesting comparison was against the best GRU model from Problem 1. Both approaches achieved nearly identical classification performance, but the Transformer required approximately **54 times more parameters** and roughly **five times longer training per epoch**. For this particular sensor-classification task, the GRU appears to provide a significantly better accuracy-to-complexity trade-off.

---

##### Step 5 – Character-Level Language Model on Tiny Shakespeare (P2.4)

The Tiny Shakespeare dataset contains 1,115,394 characters and, as expected, exactly 65 unique characters when processed programmatically.

Following the assignment instructions, I performed a 90/10 split without shuffling:

- Training characters: 1,003,854
- Validation characters: 111,540

The final model configuration was:

- Context length = 128
- Layers = 4
- Heads = 4
- Model dimension = 128
- Sinusoidal positional encoding
- Total parameters = 810,049

Training was performed for 5,000 iterations using AdamW with a learning rate of `3e-4`.

The validation loss decreased steadily throughout training and converged to:

| Iteration | Validation Loss (nats) | Bits/Char |
|---|---:|---:|
| 500 | 2.1698 | 3.1303 |
| 1500 | 1.8497 | 2.6686 |
| 3000 | 1.6828 | 2.4278 |
| 5000 | 1.6024 | 2.3118 |

The final loss of **1.6024 nats** falls comfortably within the sanity range suggested in the assignment and is significantly better than the random baseline of:

$$
\ln(65) \approx 4.1744
$$

For text generation, I produced samples at three temperatures using top-k sampling (`k=40`).

###### Temperature = 0.2

The generated text was grammatically reasonable but often repetitive. The model tended to repeatedly select highly probable words and phrases.

###### Temperature = 0.8

This produced the most balanced output. The generated text maintained Shakespeare-like structure, character names, and dialogue formatting while remaining diverse.

###### Temperature = 1.5

The generated text became much more creative but significantly less coherent. Many invented words appeared, and grammatical consistency decreased noticeably.

---

##### Step 6 – Analysis and Ablations (P2.5)

###### Attention Heatmaps

Attention visualisations were generated from two attention heads in the first encoder layer using the sentence:

> "To be, or not to be: that is the question"

Both heads focused primarily on nearby characters and neighbouring positions. A strong attention pattern also appeared around the first token, indicating that many later positions repeatedly referenced the beginning of the sequence.

Overall, the visualisations suggest that lower Transformer layers concentrate mainly on local context before higher layers construct more abstract representations.

###### Number of Attention Heads

To study the effect of head count, I compared 1, 4, and 8 attention heads while keeping the overall model dimension fixed.

| Heads | Validation Loss (nats) |
|---|---:|
| 1 | 1.7592 |
| 4 | 1.7638 |
| 8 | 1.7837 |

One observation I found particularly interesting is that the total parameter count remains unchanged when the number of heads changes. Increasing the number of heads simply partitions the same representation space differently.

Under the fixed training budget used here, the smaller-head configurations trained slightly faster and achieved marginally lower losses.

###### Removing the Causal Mask

The impact of removing the causal mask was dramatic.

| Configuration | Validation Loss (nats) |
|---|---:|
| Causal Mask | 1.7638 |
| No Mask | 0.0185 |

Initially this looked like a huge improvement, but the result is misleading. Without the causal mask, the model can directly view future characters during training and effectively learns to copy the answer rather than predict it.

This became obvious when generating text. While the masked model produced readable Shakespeare-like output, the model trained without masking generated highly repetitive and unusable sequences.

This experiment demonstrated why the causal mask is essential in autoregressive language modeling.

###### Context-Length Scaling

The final study measured execution time and memory consumption across different context lengths.

| Context Length | Time/Step (ms) | Peak Memory (MB) |
|---|---:|---:|
| 64 | 16.1 | 110.6 |
| 128 | 15.3 | 160.5 |
| 256 | 20.8 | 297.7 |
| 512 | 58.2 | 751.9 |

Both memory usage and execution time increased as the sequence length grew. Although the observed growth was not yet fully quadratic, the trend clearly reflected the increasing cost of attention computation.

From this experiment, I gained a practical appreciation for one of the main limitations of Transformer models: while they are powerful and flexible, their computational cost increases rapidly as longer contexts are used.

---

##### Engineering Note

One practical issue encountered during experimentation was managing repeated runs on Google Colab.

In Problem 1 I used a simple function that always appended new rows to `results.csv`. However, Problem 2 required frequent reruns while fixing bugs and tuning training configurations. Blindly appending results would have created duplicate records and made analysis difficult.

To address this, I implemented an `upsert_result_row()` helper that updates an existing result if the same `(problem, model, seed, seq_len)` combination already exists. A small self-test was used to verify that existing rows were correctly replaced rather than duplicated.

Although this was a relatively small implementation detail, it significantly improved result tracking and helped keep the experiment logs clean and reproducible.

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

**HAR:** Data source is [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones)

**tiny_shakespere:** The data is download from [github source](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)

**Adding Problem:** 36 runs (3 models × 4 sequence lengths × 3 seeds). Epoch budget scales with T (60/80/120/150 epochs for T=50/100/200/400) rather than a flat number for every T — longer sequences seemed to need more updates to actually converge, especially the gated models. This took noticeably longer than the HAR runs.

**Hidden-size sweep + ablation:** 18 + 3 = 21 more runs, each individually small/fast since these are still tiny HAR/adding-problem models.

**Transformer (Problem 2):** All of Problem 2 ran on Colab's T4. The HAR-encoder ablation (9 runs, 30 epochs each at ~1.33–1.36s/epoch) took roughly 6–7 minutes total. The main character-LM run (5,000 iterations at ~17.6s per 500-iteration block) took about 3 minutes. The heads ablation (3 runs × 2,000 iterations) and the no-mask ablation (2 runs × 2,000 iterations) added roughly 3.5 and 2 minutes respectively. The context-length timing/memory sweep is just a handful of forward/backward passes per context length and finished in seconds. Rough total for Problem 2: somewhere around 15–20 minutes on a T4, noticeably faster than Problem 1's RNN/LSTM/GRU sweep despite the transformer models having far more parameters — short training budgets (30 epochs / a few thousand iterations) rather than long ones, and a lot of the Problem 2 runs are small ablations rather than full sweeps.

Rough total for everything done so far in Problem 1: somewhere around 55–90 minutes on a Colab T4, across all the steps combined.

## Assumptions
There were a few places where the assignment did not provide a fixed value or a detailed instruction. In those cases, I made reasonable decisions and documented them here for clarity and reproducibility.

### 1. Hidden Size, Model Dimension, and Random Seeds

Following the instructor's clarification to ignore the variants table, I selected my own experimental settings:

- Hidden size (`h`) = 64
- Transformer model dimension (`d`) = 128
- Random seeds = 42, 123, and 999

These values were used consistently throughout the assignment unless otherwise stated.

---

### 2. Adding-Problem Training Budget

The assignment does not specify the number of training epochs for each sequence length. Instead of using the same epoch count for all values of `T`, I adjusted the training budget based on sequence length:

| Sequence Length (T) | Epochs |
|---|---:|
| 50 | 60 |
| 100 | 80 |
| 200 | 120 |
| 400 | 150 |

My reasoning was that longer sequences generally require more training to converge. Using a very small budget for all values would likely under-train the larger sequence lengths, while using the largest budget everywhere would unnecessarily increase runtime for shorter sequences.

---

### 3. Gradient-Norm Analysis Sample Size

For the gradient analysis in Problem 1, I used a fixed batch of 256 test sequences at `T = 100` when computing:

$$
\left\|\frac{\partial L}{\partial h_t}\right\|
$$

The assignment does not specify how many samples should be used for this analysis. I chose 256 examples because it provided a stable average gradient profile while keeping the computation reasonably efficient.

---

### 4. Individual Ablation Selection

My student ID ends in **6**, which is an even number.

Based on the assignment instructions, I therefore performed the **gradient clipping ablation**:

- Gradient clipping (norm = 1.0)
- No gradient clipping

using the vanilla RNN on the adding problem with `T = 200`.

Although the instructor later removed the hidden-size and seed variants, the ablation rule was not explicitly changed. For that reason, I retained the original even/odd ablation assignment.

---

### 5. Selection of the Best HAR Model

Where the assignment required identifying the best HAR model, I used **Macro-F1 Score** as the primary selection metric rather than overall accuracy.

Since the HAR dataset contains six activity classes with some class imbalance, Macro-F1 provides a more balanced assessment of performance across all classes. The confusion matrix shown in the report therefore corresponds to the run with the highest Macro-F1 score.

---

### 6. Positional-Encoding Ablation Budget

For the Transformer classifier in Problem 2, all positional-encoding variants were trained using the same budget:

- No positional encoding
- Sinusoidal positional encoding
- Learned positional encoding

Each configuration was trained for **30 epochs**, matching the HAR training budget used in Problem 1. This was done to ensure a fair comparison between the different architectures.

---

### 7. Top-k Sampling Configuration

The assignment specifies the use of top-k sampling for text generation but does not provide a value for `k`.

I selected:

```text
top-k = 40
```

for all generated samples. The same value was used consistently for every temperature setting to ensure comparability.

---

### 8. Ablation Training Budget for Transformer Experiments

The Transformer head-count experiment (`1`, `4`, and `8` heads) and the causal-mask ablation were trained using a reduced budget of **2,000 iterations** instead of the full **5,000 iterations** used for the main language-model experiment.

This allowed all ablation configurations to be compared under identical conditions while keeping the total runtime manageable.

As a result, the validation losses reported in the ablation experiments should be compared only against each other and not directly against the final language-model result obtained after the full 5,000-iteration training run.

---

### 9. Tiny Shakespeare Train/Validation Split

The Tiny Shakespeare dataset was divided into:

- First 90% of characters → Training set
- Last 10% of characters → Validation set

No shuffling was performed.

This follows the assignment specification and is appropriate for next-character prediction tasks. Random shuffling would allow future text to appear in earlier training windows, which would not reflect the true language-modeling setting.

## Known Limitations

During the course of this assignment, I observed a few limitations and results that were different from my initial expectations. These observations are noted here to provide proper context when interpreting the reported results.

### 1. Gradient Clipping Had Limited Impact on RNN Performance

For the adding problem at `T = 200`, gradient clipping (norm = 1.0) did not lead to a noticeable improvement in the vanilla RNN.

| Configuration | Test MSE |
|---|---:|
| No Clipping | 0.171 ± 0.005 |
| Clipping (Norm = 1.0) | 0.173 ± 0.008 |

Both configurations remained close to the constant-prediction baseline of approximately 0.167. Based on these results, it appears that the main difficulty for the RNN is the loss of useful gradient information over long sequences rather than occasional gradient explosions. As a result, gradient clipping alone was not sufficient to improve performance.

---

### 2. Long-Range Gradients Become Extremely Small for All Models

The gradient-norm analysis showed that gradients become extremely small when propagated far back in time. This behaviour was visible in all three architectures, including LSTM and GRU.

Before starting the assignment, I expected the gated models to completely solve the vanishing-gradient problem. However, the results suggest a more nuanced picture. While LSTM and GRU still learn the task successfully, they do not appear to preserve strong gradients across the entire sequence. Instead, the gating mechanisms seem to help the models retain useful information more effectively during training, even when gradients become small at distant time steps.

---

### 3. LSTM Performance Declined at the Longest Sequence Length

For the adding problem with `T = 400`, the LSTM model showed reduced consistency across random seeds.

| Model | Solved Runs |
|---|---:|
| LSTM | 1 / 3 |
| GRU | 3 / 3 |

While both gated architectures performed significantly better than the vanilla RNN, GRU was the more reliable model at the longest sequence length evaluated in this assignment.

---

### 4. Transformer Performance Came at a Much Higher Cost

The Transformer encoder achieved classification performance very similar to the best GRU model on the HAR dataset.

| Model | Accuracy | Parameters |
|---|---:|---:|
| Best GRU | 0.9228 | 14,790 |
| Best Transformer | 0.9216 | 803,590 |

Although the accuracy difference was minimal, the Transformer required approximately 54 times more parameters and significantly more training time per epoch.

For short sensor sequences such as HAR, the recurrent models provided a much better balance between performance and computational cost.

---

### 5. Head-Count Ablation Used a Reduced Training Budget

The comparison between 1, 4, and 8 attention heads was performed using a matched training budget of 2,000 iterations for all configurations.

Under this budget, the smaller-head configurations achieved slightly lower validation losses. However, these results should be interpreted carefully. A longer training run may lead to different outcomes once each configuration has had more opportunity to converge.

---

### 6. Very Low Loss Without a Causal Mask Was Misleading

Removing the causal mask produced an extremely small validation loss.

| Configuration | Validation Loss (nats) |
|---|---:|
| Causal Mask | 1.7638 |
| No Mask | 0.0185 |

At first glance this appears to be a major improvement. However, the result is not representative of a genuine language model because the model can directly observe future characters during training.

This became clear during text generation. The masked model generated coherent Shakespeare-like text, whereas the no-mask model produced highly repetitive and unusable outputs. This experiment demonstrated the importance of evaluating models using both quantitative metrics and qualitative behaviour.

---

### 7. Context-Length Scaling Was Not Fully Quadratic

The time and memory measurements increased as context length grew, but the observed growth was lower than the simple theoretical expectation of pure quadratic scaling.

A likely explanation is that, for the tested model size (`d = 128`, 4 layers), several components of the network still contribute significant linear computation costs. The quadratic attention cost becomes increasingly important as the context grows, but it does not completely dominate within the range of context lengths evaluated in this assignment.

---

### 8. Model Checkpoints Are Not Included

The assignment instructions explicitly state that trained model checkpoints should not be included in the final submission.

Most training runs were executed using Google Colab, and the resulting model checkpoints were stored in Google Drive during experimentation. These checkpoint files were not downloaded or included in the repository or final submission package. All reported results can be reproduced by running the provided notebooks with the documented configurations and random seeds.

---

### 9. Generation of requirements.txt
As I have flipped from the local to colab and vice versa. Also my code is segregated between `.py` and `.ipynb` files. So I did not find a best way of generating requirements.txt which is full proof. Thus invented my own way of doing the same. Giving the code block below but removed from the actual files to make it clean.
```python
# IPython magic: captures stdout of `pip freeze` as a list, kept in memory only
freeze_output = !pip freeze

freeze_versions = {}
for line in freeze_output:
    if "==" in line:
        pkg, ver = line.split("==", 1)
        freeze_versions[pkg.lower()] = ver

# import name -> actual PyPI distribution name (differs for sklearn and colab)
IMPORT_TO_DIST = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "google.colab": "google-colab",
}

print("Pip-installed packages:")
for import_name, dist_name in IMPORT_TO_DIST.items():
    version = freeze_versions.get(dist_name.lower(), "NOT FOUND")
    print(f"  {import_name:15s} -> {dist_name:15s} {version}")

import sys

print("\nNot separately pip-installed (no version to look up this way):")
print(f"  torch.nn.functional -> submodule of torch, same version as torch above")
print(f"  math, time, urllib, zipfile -> Python standard library, version = Python {sys.version.split()[0]}")
```