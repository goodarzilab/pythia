# Pythia

Pythia is an RNA sequence analysis toolkit for predicting RNA-binding protein (RBP) binding sites and structural properties. It provides models for both sequence-based (RBP) and structure-aware (BEACON benchmark) tasks.

**Maintainer:** Mehran Karimzadeh ([mehran.karimzade@gmail.com](mailto:mehran.karimzade@gmail.com))

**License:** [MIT License](LICENSE)

---

## Tasks

| Task | Description | Model | Metric |
|------|-------------|-------|--------|
| **RBP** | Binary classification: does this sequence bind an RBP? | `PythiaRBP` | AUPRC, AUROC |
| **SSP** | Secondary structure prediction (L×L base-pairing matrix) | `PythiaSSP` | F1, Precision, Recall |
| **CMP** | Contact map prediction (long-range pairs ≥23 nt) | `PythiaCMP` | Top-L precision |
| **DMP** | Distance map prediction (pairwise spatial distances in [0,1]) | `PythiaDMP` | R², MSE |
| **SSI** | Structural score imputation (per-position regression) | `PythiaSSI` | R² |

---

## Environment Setup

```bash
git clone <this-repo-url> pythia
cd pythia
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Data Formats

### RBP Task

Tab-separated files (`.tsv` or `.tsv.gz`) with header:

```
Input          Response  SeqNames                 MFEs
UCACAUC...     Unbound   Shuffled:chr6:126146...  0
GCUUAGC...     Bound     chr1:1234-1491(+)        -5.2
```

| Column | Description |
|--------|-------------|
| `Input` | RNA sequence (ACGU or ACGT; T→U is applied automatically) |
| `Response` | `Bound` / `Unbound` or `1` / `0` |
| `SeqNames` | Sequence identifier (optional) |
| `MFEs` | Minimum free energy (optional) |

**Directory layout:**
```
dataSplits/
└── ELAVL1/
    ├── ELAVL1_trainingSet.tsv.gz
    ├── ELAVL1_tuningSet.tsv.gz
    └── ELAVL1_validationSet.tsv.gz
```

The synthetic RBP benchmark tasks (`pseudoknot`, `sequence_motif`, `structural_motif`) used in the published RBP benchmark are not real RBP datasets — they are generated locally; see [Reproducing published results](#reproducing-published-results) below. Real RBP CLIP-seq datasets are not redistributed with this repository; bring your own in the TSV format above (obtained, e.g., from ENCODE eCLIP or your own CLIP-seq processing pipeline).

### SSP / CMP Tasks (bpRNA-style CSV)

```
data_name,file_name,seq,dot_string
TR0,d.16.b.E.coli.1,GCGGAUUUAGCU...,((((....))))...
VL0,d.16.b.H.sapiens.3,AGUUCGCUAUCC...,..((.))..
TS0,d.16.b.T.thermophilus.1,GCGGAUUU...,((((...))))
```

| Column | Description |
|--------|-------------|
| `data_name` | Split: `TR0` (train), `VL0` (val), `TS0` (test) |
| `file_name` | Sequence identifier |
| `seq` | RNA sequence |
| `dot_string` | Dot-bracket secondary structure notation |

For **CMP**, only base pairs with `|i - j| ≥ 23` are retained. The `bpRNA-1m` dataset used for the published SSP results is publicly available from the [bpRNA project](https://bprna.cgrb.oregonstate.edu/). CMP and DMP in the deployed configuration instead use the pre-computed `.npy` contact-map / distance-map format from the [BEACON benchmark](https://github.com/terry-r123/BEACON) (see below).

### CMP / DMP Tasks (BEACON `.npy` format)

```
{data_dir}/
├── train.csv          # columns: id, input
├── val.csv
├── contact_map/       # (CMP) or distance_map/ (DMP)
│   └── {id}.npy       # pre-computed L x L contact/distance matrix
├── RFAM19.csv
├── DIRECT.csv
└── test.csv
```

This is the format used by the [BEACON benchmark](https://github.com/terry-r123/BEACON)'s ContactMap and DistanceMap tasks. BEACON `.npy` distance maps are already normalized to `[0, 1]`.

### SSI Task

```
sequence,struct,struct_true
GCUGAUC...,0.5 -1.0 0.3 0.7 ...,0.5 0.45 0.3 0.72 ...
```

| Column | Description |
|--------|-------------|
| `sequence` | RNA sequence |
| `struct` | Observed structural scores (space-separated floats; `-1` = missing) |
| `struct_true` | Ground-truth structural scores (space-separated floats) |

This is the StructuralScoreImputation task format from the [BEACON benchmark](https://github.com/terry-r123/BEACON).

---

## Architecture

### Core Component: FixedDilatedConv

Pythia's key innovation is a frozen convolutional layer that encodes RNA base-pairing biochemistry. Weights detect Watson-Crick (A-U, C-G) and wobble (G-U) pairs at various stem lengths and with bulge variants.

**Critical:** `FixedDilatedConv` weights are **never trained** (`trainable=False`).

### Dual-Arm Backbone (SSP / CMP / DMP / SSI)

```
one-hot (B, 4, L)
    ├─→ FixedDilatedConv (frozen) → arm1 [K16, K10, K5] → (B, H1, L)
    └─→ arm2 [K12, K7, K5] ─────────────────────────────→ (B, H2, L)
                concat → project → pairwise features → 2D ResNet → output
```

### Structure-Specific Heads

| Task | Head | Extra Features |
|------|------|---------------|
| SSP | 2D ResNet (outer_prod + outer_sum + outer_diff + dist_enc) | — |
| CMP | 2D ResNet (same) | long-range mask (|i-j| ≥ 23) |
| DMP | 2D ResNet (+ weighted_diff, weighted_sum, seq_dist, decay×3) | sigmoid output |
| SSI | Squeeze-and-excite + 1D ResBlock stack + linear regression head | observed-score conditioning |

---

## Model Architectures

### FixedDilatedConv

The foundation of every Pythia model is a hand-crafted, permanently frozen convolutional layer that encodes RNA secondary-structure biochemistry directly into the network.

**Mechanism.** For each dilation `d` in `[dil_start, dil_start+1, …, dil_end]`, two dilated 1-D convolutions with kernel size `2·bulge_size+1` detect the complementary pair at positions `(i, i+d)`:

- Watson-Crick pairs: A-U (`A=1, U=4`) and C-G (`C=2, G=3`)
- Wobble pair: G-U

Weights are pre-filled from biochemical pairing scores and are never updated. Two filter variants (exact + bulge-tolerant) are instantiated per dilation, giving `2 × (dil_end − dil_start + 1)` output channels.

**Output.** `(B, C_fd, L)` where `C_fd = 2 × (dil_end − dil_start + 1)`.

**Default parameters.**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `dil_start` | 5 | minimum stem span |
| `dil_end` | 24 (SSP/CMP/DMP), 36 (SSI) | maximum stem span |
| `bulge_size` | 2 | half-width of bulge kernel |
| `trainable` | `False` | always frozen |

---

### PythiaRBP

**Task.** Binary classification — does a given RNA sequence bind the target RBP?

**Architecture.**

```
one-hot (B, 4, L)
    ├─→ FixedDilatedConv → AdaptiveMaxPool1d(inputsize) → arm1
    │       [Conv1d K16 → Conv1d K10 → Conv1d K5]   (B, 32, inputsize)
    └─→ arm2
            [Conv1d K12 → Conv1d K7  → Conv1d K5]   (B, 32, inputsize)

concat → flatten → Linear(64·inputsize → 64) → GELU → Dropout
       → Linear(64 → 32) → GELU → Dropout
       → Linear(32 → 2)   [logits for Bound / Unbound]
```

Each arm is a stack of Conv1d + BatchNorm + GELU blocks with decreasing kernel sizes. The FD features are globally pooled to a fixed length (`inputsize=256`) before entering arm1; arm2 operates on the raw one-hot input at the same pooled length.

**Training.** PyTorch-Lightning; cross-entropy loss with configurable per-class weighting (`pos_weight` for the negative class); cosine-warmup LR schedule.

**Deployed hyperparameters.**

| Parameter | Value |
|-----------|-------|
| `inputsize` | 256 |
| `dil_start` / `dil_end` | 2 / 48 |
| `bulge_size` | 2 |
| `binarize_fd` | True |
| `dp` | 0.5 |
| `lr` | 0.004 |
| `batch_size` | 256 |
| `max_epochs` | 60 |
| `patience` | 20 |

---

### PythiaSSP — Secondary Structure Prediction

**Task.** Predict the full L×L base-pairing contact matrix for an RNA sequence.

**Architecture.**

```
one-hot (B, 4, L)
    ├─→ FixedDilatedConv → arm1 [K16, K10, K5] → (B, 128, L)
    └─→                    arm2 [K12,  K7, K5] → (B, 128, L)

concat (B, 256, L) → 1×1 Conv → projection (B, hidden_dim, L)

Pairwise features  (B, L, L, F):
    outer_product(i,j)      [hidden_dim channels]
    outer_sum(i,j)          [hidden_dim channels]
    outer_diff(i,j)         [hidden_dim channels]
    learned_dist_enc(|i−j|) [dist_enc_dim channels]

→ 2-D ResNet (num_res_layers blocks, dilation pattern [1,2,4,1,2,4,1,1])
→ Conv2d head → (B, 1, L, L) → sigmoid → symmetrize
```

The 2-D ResNet blocks use GroupNorm + GELU. The final prediction matrix is symmetrized by averaging with its transpose and thresholded at 0.5 for evaluation.

**Loss.** `BCEWithLogitsLoss`. **Metric.** Global micro-averaged F1, precision, recall.

**Deployed hyperparameters.**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 384 |
| `num_res_layers` | 8 |
| `arm1_widths` | (128, 128, 128) |
| `arm2_widths` | (256, 128, 128) |
| `dist_enc_dim` | 64 |
| `dropout` | 0.15 |
| `lr` | 3e-4 |
| `warmup_epochs` | 3 |
| `max_len` | 512 |

---

### PythiaCMP — Contact Map Prediction

**Task.** Predict long-range RNA tertiary contacts (nucleotide pairs with `|i−j| ≥ 23`).

**Architecture.** Identical dual-arm + 2-D ResNet backbone to `PythiaSSP`, with two differences:

1. **Tapered arm widths.** Arm channels narrow across layers to reduce computation on long sequences: arm1 `(128 → 96 → 64)`, arm2 `(192 → 128 → 64)`.
2. **Long-range mask.** All pairs with `|i−j| < min_separation` (default 23) are excluded from the loss and evaluation metric (Top-L precision at threshold 0.5).

The 2-D ResNet uses a `(1, 2, 4, 8)` cyclic dilation pattern.

**Deployed hyperparameters.**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 320 |
| `num_res_layers` | 7 |
| `arm1_widths` | (128, 96, 64) |
| `arm2_widths` | (192, 128, 64) |
| `min_separation` | 23 |
| `lr` | 5e-5 |
| `weight_decay` | 0.01 |
| `batch_size` | 1 (grad_accum=8) |
| `max_len` | 1024 |

---

### PythiaDMP — Distance Map Prediction

**Task.** Predict a pairwise normalized spatial distance matrix in `[0, 1]` for all nucleotide pairs.

**Architecture.** Same dual-arm + 2-D ResNet backbone, but with a substantially richer set of pairwise input features:

| Feature group | Channels | Description |
|---------------|----------|-------------|
| `outer_product` | hidden_dim | element-wise product of 1-D embeddings |
| `outer_sum` | hidden_dim | element-wise sum |
| `outer_diff` | hidden_dim | element-wise difference |
| `weighted_diff` | hidden_dim | `outer_diff × seq_dist_norm(i,j)` |
| `weighted_sum` | hidden_dim | `outer_sum × inv_seq_dist(i,j)` |
| `seq_dist` | 1 | raw `|i−j| / L` |
| `decay` | 3 | `exp(−|i−j| / τ)` for τ ∈ {10, 50, 100} |
| `learned_dist_enc` | dist_enc_dim | binned learnable embeddings |

The output is passed through sigmoid to enforce the `[0, 1]` range and symmetrized. Loss is `HuberLoss`. Metric: global Pearson R² and MSE.

**Deployed hyperparameters.**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_res_layers` | 9 |
| `arm1_widths` | (192, 192, 128) |
| `arm2_widths` | (256, 256, 128) |
| `dist_enc_dim` | 128 |
| `huber_delta` | 0.3 |
| `lr` | 1e-4 |
| `weight_decay` | 1e-3 |
| `max_len` | 1024 |

---

### PythiaSSI — Structural Score Imputation

**Task.** Per-position regression: given partially observed DMS/SHAPE reactivity scores, impute the missing positions.

**Architecture.** Unlike the 2-D tasks, SSI uses a **1-D refinement backbone**, and conditions on the partially-observed scores.

```
one-hot (B, 4, L)
    ├─→ FixedDilatedConv (dil_end=36) → squeeze-and-excite → arm1 [K16, K10, K5] → (B, 128, L)
    └─→ one-hot + obs_score + obs_known ─────────────────────→ arm2 [K12, K7, K5] → (B, 128, L)

concat → 1×1 Conv → projection (B, hidden_dim, L)

→ num_res_layers × ResBlock1D(hidden_dim)
      [Conv1d → BatchNorm → GELU → Conv1d → BatchNorm + residual]

→ LayerNorm
→ Linear(hidden_dim → hidden_dim//2) → GELU
→ Linear(hidden_dim//2 → 1)
→ per-position scalar output (B, L)
```

**Loss.** Only computed on positions where `struct_true` is masked as missing in `struct` (masked imputation). Loss function is configurable: `mse`, `huber`, or `l1` (deployed: `l1`). **Metric:** Pearson R².

**Dual learning-rate optimizer.** Feature extractor and regression head use separate learning rates (`lr_feat`, `lr_head`) via two AdamW parameter groups.

**Deployed hyperparameters.**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 384 |
| `num_res_layers` | 10 |
| `dil_start` / `dil_end` | 2 / 36 |
| `bulge_size` | 4 |
| `dropout` | 0.307 |
| `loss_fn` | `l1` |
| `lr_feat` | 7.071e-4 |
| `lr_head` | 2.598e-4 |
| `weight_decay` | 8.864e-3 |
| `warmup_epochs` | 20 |
| `grad_clip` | 1.152 |
| `max_len` | 440 |

---

## Usage

### Training

**RBP task:**
```bash
python train_rbp.py \
    --train-tsv ELAVL1_trainingSet.tsv.gz \
    --val-tsv   ELAVL1_tuningSet.tsv.gz \
    --test-tsv  ELAVL1_validationSet.tsv.gz \
    --output-dir results/ELAVL1 \
    --inputsize 256 --max-epochs 50 --batch-size 64
```

**BEACON tasks (SSP / CMP / DMP / SSI):**
```bash
python train_beacon.py \
    --task ssp \
    --csv /data/bpRNA.csv \
    --output-dir results/ssp \
    --max-epochs 80 --batch-size 1 --lr 3e-4

python train_beacon.py \
    --task ssi \
    --train-csv /data/ssi_train.csv \
    --val-csv /data/ssi_val.csv \
    --test-csv /data/ssi_test.csv \
    --output-dir results/ssi \
    --dil-end 36 --hidden-dim 384 --num-res-layers 10
```

### Inference

```bash
python infer.py \
    --task rbp \
    --checkpoint results/ELAVL1/checkpoints/best.ckpt \
    --input-csv  ELAVL1_validationSet.tsv.gz \
    --output-csv results/ELAVL1/predictions.csv \
    --inputsize 256
```

### Plotting

```bash
python plot.py \
    --task rbp \
    --predictions results/ELAVL1/predictions.csv \
    --output-dir  results/ELAVL1/figures \
    --log-csv     results/ELAVL1/lightning_logs/version_0/metrics.csv
```

---

## Published Performance

| Task | Metric | Pythia |
|------|--------|--------|
| SSP | Test F1 | **0.714** |
| CMP | Top-L precision | **62.45%** |
| DMP | R² | **46.42%** |
| SSI | R² | **39.4%** |

---

## Package Structure

```
pythia/
├── pyproject.toml
├── LICENSE
├── README.md
├── src/pythia/
│   ├── configs.py           # Pydantic v2 config classes
│   ├── train_rbp.py         # RBP training script (Lightning)
│   ├── train_beacon.py      # BEACON training script (PyTorch native)
│   ├── infer.py             # Inference script
│   ├── plot.py              # Performance plotting
│   ├── models/
│   │   ├── fixed_dilated_conv.py   # Canonical FixedDilatedConv
│   │   ├── common.py               # Shared: ResBlock1D, ResBlock2D, PositionalEncoding2D
│   │   ├── rbp.py                  # PythiaRBP
│   │   ├── ssp.py                  # PythiaSSP
│   │   ├── cmp.py                  # PythiaCMP
│   │   ├── dmp.py                  # PythiaDMP
│   │   └── ssi.py                  # PythiaSSI
│   ├── data/
│   │   ├── rbp_dataset.py          # RBPDataset, RBPDataModule
│   │   └── beacon_dataset.py       # SSP/CMP/DMP/SSI datasets + collators
│   └── notebooks/
│       ├── 01_rbp_binary_classification.ipynb
│       └── 02_structural_tasks.ipynb
├── data_generation/
│   ├── generate_sequence_motif_dataset.py
│   ├── generate_structural_motif_dataset.py
│   └── pseudoknot_joblib_to_tsv.py
├── experiments/
│   └── compute_deeplift_contributions.py
├── plots/
│   ├── benchmark_comparison_plot.R
│   └── sequence_structure_contribution_plot.R
└── examples/
    ├── run_train_rbp.sh    # RBP end-to-end example
    └── beacon/
        ├── run_ssp.sh      # SSP end-to-end example
        ├── run_cmp.sh      # CMP end-to-end example
        ├── run_dmp.sh      # DMP end-to-end example
        ├── run_ssi.sh      # SSI end-to-end example
        ├── pythia_beacon_config.yaml
        ├── pythia_beacon_structural_tasks.py
        ├── run_pythia_beacon_structural_tasks.sh
        ├── beacon_leaderboard.yaml
        └── beacon_benchmark_plot.R
```

---

## Reproducing published results

All items below are runnable from a clean `uv sync` environment plus your own copy of the input data in the documented format. None of the training/evaluation data used for the published results is redistributed with this repository.

| # | Reproduces | Command |
|---|---|---|
| 1 | Training a Pythia RBP classifier | `bash examples/run_train_rbp.sh <data-dir> <rbp-name> <output-dir>` (wraps `train_rbp.py` + `infer.py` + `plot.py`) |
| 2 | Generating the synthetic RBP benchmark datasets (pseudoknot / sequence_motif / structural_motif) | `python data_generation/generate_sequence_motif_dataset.py`, `python data_generation/generate_structural_motif_dataset.py --mode structure` (self-contained, no external data), then `python data_generation/pseudoknot_joblib_to_tsv.py` to produce training TSVs for use with (1) |
| 3 | Training BEACON benchmarks (SSP/CMP/DMP/SSI) | `bash examples/beacon/run_ssp.sh`, `run_cmp.sh`, `run_dmp.sh`, `run_ssi.sh`, or drive all four via `python examples/beacon/pythia_beacon_structural_tasks.py --task <ssp\|cmp\|dmp\|ssi>` after editing `pythia_beacon_config.yaml` for your data paths |
| 4 | Computing sequence-vs-structure DeepLIFT contribution | `python experiments/compute_deeplift_contributions.py --data-dir ... --splits-dir ... --outdir ...` (zero-baseline attribution, computed on checkpoints from (1)) |
| 5 | Plotting the DeepLIFT contribution output | `Rscript plots/sequence_structure_contribution_plot.R <data-dir> <outdir>` (uses the output of (4)) |
| 6 | Generating the RBP benchmark comparison plots | `Rscript plots/benchmark_comparison_plot.R <indir> <outdir> <rnafm-dir>` (compares against third-party baseline model outputs -- see the script's header for the expected directory layout; those baseline outputs are not redistributed here) |
| 7 | Generating the BEACON leaderboard comparison plot | `Rscript examples/beacon/beacon_benchmark_plot.R <performance-dir> [outdir]` (uses `beacon_leaderboard.yaml` plus the `*_pythia_performance.json` files written by item 3) |

For the RBP benchmark comparison plot (item 6), organize each RBP's Pythia predictions at `{indir}/{rbp}/Pythia/{rbp}_validationPredictions.tsv` by passing that path as `<output-dir>` to `examples/run_train_rbp.sh`.

---

## Hyperparameter Reference

### SSP (secondary structure prediction)
- `hidden_dim=384`, `num_res_layers=8`, `dropout=0.15`
- `arm1_widths=(128,128,128)`, `arm2_widths=(256,128,128)`
- Adam `lr=3e-4`, cosine schedule, warmup=3 epochs, patience=10

### CMP (contact map prediction)
- `hidden_dim=320`, `num_res_layers=7`, `dropout=0.15`
- `arm1_widths=(128,96,64)`, `arm2_widths=(192,128,64)`
- `min_separation=23` (only long-range contacts)
- AdamW `lr=5e-5`, `wd=0.01`, warmup=50 epochs, patience=5

### DMP (distance map prediction)
- `hidden_dim=256`, `num_res_layers=9`, `dropout=0.1`
- `arm1_widths=(192,192,128)`, `arm2_widths=(256,256,128)`
- AdamW `lr=1e-4`, `wd=1e-3`, warmup=20 epochs, patience=10

### SSI (structural score imputation)
- `hidden_dim=384`, `num_res_layers=10`, `dropout=0.307`, `dil_start=2`, `dil_end=36`, `bulge_size=4`
- `lr_feat=7.071e-4`, `lr_head=2.598e-4`, `loss_fn=l1`, warmup=20 epochs, patience=12
