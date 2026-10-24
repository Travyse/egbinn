# EGBINN

EGBINN: Ensembled Gated Biological Interpretable Neural Network for DNA methylation-based age prediction.

## Overview

EGBINN is a DNA methylation-based aging clock that uses a multi-stage sparse biological hierarchy (CpG → gene → Reactome pathway → ...) with gated sparse linear layers and TabM-style parameter-efficient ensembling. The model maps CpG beta values through a biologically-informed sparse hierarchy, followed by per-stage ensemble prediction heads.

Three models are provided:

- **EGBINN** (default, recommended): Gated sparse biological hierarchy + TabM ensemble with `k=8` members and multi-stage prediction heads.
- **BINN**: Non-ensemble variant with all-sparse stages and per-stage linear heads. Simpler and more interpretable.
- **MLP**: Standard flat MLP baseline operating on the full CpG beta vector.

## Quick Start

### Making age predictions (command-line version)

```bash
egbinn-evaluate \
    --checkpoint resources/pretrained/egbinn/checkpoint.pth \
    --data /path/to/methylation.memmap \
    --metadata /path/to/metadata.csv \
    --output_file predictions.csv
```

### Making age predictions (python version)

```python
from egbinn.evaluation import predict

results_df = predict(
    checkpoint_path="resources/pretrained/egbinn/checkpoint.pth",
    data_path="/path/to/methylation.memmap",
    metadata_path="/path/to/metadata.csv",
)
print(results_df.head())
```

### Output format

Returns a pandas DataFrame with:

- `sample`: Sample ID (from the metadata index)
- `chron_age`: Chronological age (true age from metadata)
- `pred_age`: Predicted age in years

If an output file is specified, a companion `_metrics.json` file is also written with MAE, MSE, MedAE, Pearson, and Spearman correlation. When the metadata contains a dataset column (default `Accession_Code`), the CLI additionally writes a `_per_dataset.csv` with per-dataset metrics.

## Checkpoint format

Checkpoints are **self-contained bundles** — a single `.pth` file carries everything needed for inference:

- model weights (class-agnostic state dict; `_orig_mod.` prefixes from `torch.compile` are stripped on load)
- the **training** label scaler (predictions are always inverse-transformed with it, never a scaler fit on evaluation data)
- the resolved training config, including the inlined biological hierarchy masks (BINN / EGBINN)
- provenance: input dimension, package version, git commit, best epoch, training timestamp

`config_path` / `label_scaler_path` are only needed for **legacy** plain state-dict checkpoints; bundle checkpoints work with no extra files. A human-readable `config.yml` and `label_scaler.json` are shipped alongside each pretrained checkpoint for reference.

## Installation

### Install with conda (recommended)

```bash
git clone https://github.com/<your-org>/egbinn.git
cd egbinn
conda env create -f environment.yml
conda activate egbinn
```

### Install from source (pip only)

```bash
conda create -n egbinn python=3.12
conda activate egbinn
git clone https://github.com/<your-org>/egbinn.git
cd egbinn
pip install -e .
```

### Check installation

```bash
egbinn-evaluate --help
```

```
usage: egbinn-evaluate [-h] --checkpoint CHECKPOINT [--config CONFIG_PATH]
                       --data DATA_PATH --metadata METADATA_PATH
                       [--imputation_path IMPUTATION_PATH]
                       [--data_format {memmap,csv}]
                       [--label_scaler_path LABEL_SCALER_PATH]
                       [--label_column LABEL_COLUMN]
                       [--dataset_col DATASET_COL] [--batch_size BATCH_SIZE]
                       [--num_workers NUM_WORKERS]
                       [--device {auto,cuda,mps,cpu}]
                       [--output_dir OUTPUT_DIR] [--output_file OUTPUT_FILE]
```

The `--device` flag defaults to `auto` (CUDA when available, then Apple MPS, then CPU). An explicit request that is unavailable on the machine fails immediately instead of silently falling back.

## Supported Models

| Model | Class | Description |
|-------|-------|-------------|
| EGBINN | `egbinn.models.EGBINN` | Gated sparse hierarchy + TabM ensemble (k=8, multi-stage heads) |
| BINN | `egbinn.models.BINNRegression` | Non-ensemble sparse hierarchy with per-stage heads |
| MLP | `egbinn.models.MLPRegression` | Standard flat MLP baseline |

## Input Data Format

### Data format options

Two data formats are supported:

- **Memmap** (`.memmap`): For full-scale CpG data (180K+ features). A sidecar `_avg_beta_values.npy` file next to the memmap provides the feature count and is used for NaN imputation.
- **CSV** (`.csv`): For smaller datasets. Samples as rows, CpG sites as columns.

### Metadata format

A CSV file with at least:

- An index column with sample IDs
- An `age` column (chronological age in years)
- An `Accession_Code` column (dataset identifier, used for per-dataset metrics)

The metadata rows must correspond one-to-one, in order, with the rows of the methylation data file — mismatches fail fast with a clear error.

Example:

```csv
sample_id,age,Accession_Code
GSM123456,45.2,GSE40279
GSM123457,62.1,GSE40279
...
```

### Biological hierarchy masks (BINN and EGBINN only)

BINN and EGBINN follow a biological hierarchy of sparse edge-mask artifacts (`mask_indices.npy` + `shapes.json`) for each stage (cpg_to_gene, gene_to_l2, l2_to_l1, l1_to_l0), shipped in `resources/masks/`. They are required when **training from a config**; bundle checkpoints inline the masks, so inference needs nothing from `resources/masks/`. The SHAP tooling also uses `output_index.csv` / `edges.tsv` from this directory for pathway display names. Masks can be regenerated from Reactome pathway annotations.

## Model Training

### Training with CLI

```bash
egbinn-train --config configs/egbinn.yml
```

### Training with python

```python
from egbinn.training import train_model, build_model
from egbinn.adapters import get_adapter
from egbinn.data import load_data
from egbinn.config import load_config
from egbinn.utils import seed_everything
import torch

config = load_config("configs/egbinn.yml")
seed_everything(42)

# Build model
model = build_model(config, input_dim=189291, mask_dir="resources/masks")
adapter = get_adapter(config.model)

# Load data
train_loader, label_scaler = load_data(
    methylation_path=config.data.train_methylation_path,
    metadata_path=config.data.metadata_path,
    label_column="age",
    batch_size=256,
    shuffle=True,
)
val_loader, _ = load_data(
    methylation_path=config.data.val_methylation_path,
    metadata_path=config.data.metadata_path,
    label_column="age",
    label_scaler=label_scaler,
    batch_size=256,
    is_validation=True,
)

# Train
summary = train_model(
    model=model,
    adapter=adapter,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=torch.device("cuda"),
    label_scaler=label_scaler,
)
print(f"Best epoch: {summary['best_epoch']}, MAE: {summary['best_val_mae']:.4f}")
```

### Configuration

Training is controlled by a YAML config file. See `configs/` for examples:

- `configs/egbinn.yml` — EGBINN with gated stages, k=8, multi-stage heads
- `configs/binn.yml` — BINN with all-sparse stages, stage_activation=relu
- `configs/mlp.yml` — Standard MLP baseline

Key config fields:

```yaml
model: EGBINN  # or BINN or MLP

general:
  lr: 0.0001
  weight_decay: 0.0001
  batch_size: 256
  train_epochs: 101
  patience: 30
  huber_delta: 0.36

model_params:
  k: 8                      # ensemble members (EGBINN only)
  regression_approach: multi_stage  # or "final"
  stages:                   # BINN/EGBINN only
    - name: cpg_to_gene
      layer_type: gated     # or "sparse"
      ...

data:
  train_methylation_path: /path/to/train.memmap
  val_methylation_path: /path/to/val.memmap
  metadata_path: /path/to/metadata.csv
  mask_dir: resources/masks  # BINN/EGBINN only

output:
  save_dir: ./outputs/egbinn
  checkpoint_name: egbinn_checkpoint.pth
```

## Age Prediction

### CLI

```bash
# EGBINN
egbinn-evaluate \
    --checkpoint resources/pretrained/egbinn/checkpoint.pth \
    --data /path/to/data.memmap \
    --metadata /path/to/metadata.csv \
    --output_file egbinn_predictions.csv

# BINN
egbinn-evaluate \
    --checkpoint resources/pretrained/binn/checkpoint.pth \
    --data /path/to/data.memmap \
    --metadata /path/to/metadata.csv \
    --output_file binn_predictions.csv

# MLP
egbinn-evaluate \
    --checkpoint resources/pretrained/mlp/checkpoint.pth \
    --data /path/to/data.memmap \
    --metadata /path/to/metadata.csv \
    --output_file mlp_predictions.csv
```

### Python

```python
from egbinn.evaluation import predict

# EGBINN
results = predict(
    checkpoint_path="resources/pretrained/egbinn/checkpoint.pth",
    data_path="/path/to/data.memmap",
    metadata_path="/path/to/metadata.csv",
    output_file="egbinn_predictions.csv",
)
```

## SHAP Feature Importance

EGBINN supports SHAP GradientExplainer analysis at the hidden-layer level (genes, L2/L1/L0 Reactome pathways), attributing age predictions to biological features rather than raw CpG inputs.

### CLI

```bash
egbinn-shap \
    --checkpoint resources/pretrained/egbinn/checkpoint.pth \
    --config resources/pretrained/egbinn/config.yml \
    --background_data /path/to/normal.memmap \
    --test_data /path/to/tumor.memmap \
    --mask_dir resources/masks/ \
    --output_dir shap_results/
```

### Python

```python
from egbinn.shap_analysis import run_shap_analysis

results = run_shap_analysis(
    checkpoint_path="resources/pretrained/egbinn/checkpoint.pth",
    config_path="resources/pretrained/egbinn/config.yml",
    background_betas=normal_betas,  # numpy array (N_bg, n_features)
    test_betas=tumor_betas,        # numpy array (N_test, n_features)
    mask_dir="resources/masks/",
    output_dir="shap_results/",
)
```

Output: per-layer importance parquet files (`gene_importance.parquet`, `l2_importance.parquet`, etc.) in the output directory.

## Reproducing Manuscript Figures

The `notebooks/` directory contains Jupyter notebooks to reproduce the figures shown in the manuscript:

| Notebook | Figure | Required inputs |
|----------|--------|-----------------|
| `pathway_importance.ipynb` | Cross-cohort Reactome pathway recurrence dot plots (direct L1 and L2 SHAP; main + supplementary variants, within-cohort rank color) | Per-cohort SHAP parquets under `outputs/shap/per_cohort/{cohort}/` |
| `gating_analysis.ipynb` | Gate ~ Age + Tissue regression analysis (Panel A: overall age-associated representations; Panel B: tissue-dependent interaction test) | Gate activations under `outputs/gates/` |

Each notebook has a configuration cell at the top with file paths anchored to the repo root; the input layouts are documented there and produced by the SHAP CLI and the gate-activation export tooling, respectively.

## Pretrained Models

`resources/pretrained/{egbinn,binn,mlp}/` contain the manuscript's production checkpoints, each as a self-contained bundle (weights + training scaler + resolved config + provenance).

## Citation

If you use this code or the pretrained models, please cite:

Edwards TA, Long Q, Shen L. EG-BINN: Parameter-Efficient Ensembling of Gated Biologically Informed Neural Networks for Epigenetic Age Estimation. bioRxiv 2026.10.17.759869; doi: [10.64898/2026.10.17.759869](https://doi.org/10.64898/2026.10.17.759869)

```bibtex
@article {Edwards2026.10.17.759869,
	author = {Edwards, Travyse Anthony and Long, Qi and Shen, Li},
	title = {EG-BINN: Parameter-Efficient Ensembling of Gated Biologically Informed Neural Networks for Epigenetic Age Estimation},
	elocation-id = {2026.10.17.759869},
	year = {2026},
	doi = {10.64898/2026.10.17.759869},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/10/22/2026.10.17.759869},
	eprint = {https://www.biorxiv.org/content/early/2026/10/22/2026.10.17.759869.full.pdf},
	journal = {bioRxiv}
}
```