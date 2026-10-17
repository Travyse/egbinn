"""SHAP GradientExplainer analysis for EGBINN aging clocks.

Attributes the model's age predictions to hidden-layer features (genes,
L2/L1/L0 Reactome pathways) rather than the raw CpG inputs. Uses
``shap.GradientExplainer`` with a "cut-model" strategy: the model is split
at a target layer, post-activation hidden activations of the background
and test samples are pre-computed from the front-end, and a
:class:`LayerWrapper` (representing the back-end + downstream heads) is
the function being explained.
"""

__all__ = [
    "LayerWrapper",
    "compute_activations",
    "load_feature_names",
    "run_shap_analysis",
    "run_shap_for_layer",
]

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import load_config
from .evaluation import load_checkpoint
from .training import build_model
from .utils import resolve_device

logger = logging.getLogger(__name__)

LAYER_SPECS: dict[str, dict[str, Any]] = {
    "gene": {"cut_idx": 0, "stage_name": "cpg_to_gene", "wide_output": True, "name_col": "gene"},
    "l2": {"cut_idx": 1, "stage_name": "gene_to_l2", "wide_output": False, "name_col": "pathway"},
    "l1": {"cut_idx": 2, "stage_name": "l2_to_l1", "wide_output": False, "name_col": "pathway"},
    "l0": {"cut_idx": 3, "stage_name": "l1_to_l0", "wide_output": False, "name_col": "pathway"},
}


class LayerWrapper(nn.Module):
    """Back-end sub-model from a cut point to the downstream prediction.

    Takes a flattened ``(B, K * dim)`` post-activation hidden representation
    at hierarchy level ``l = cut_idx`` and runs the remaining sparse stages,
    optional dense blocks, and trained ``stage_heads`` to produce the
    downstream component of the multi-stage EGBINN prediction.
    """

    def __init__(self, model, cut_idx: int, feature_dim: int):
        super().__init__()
        if model.n_heads == 0:
            raise ValueError("LayerWrapper requires regression_approach='multi_stage'.")
        self.cut_idx = cut_idx
        self.feature_dim = feature_dim
        self.k = model.k
        self.H = model.n_heads
        self.n_dense_blocks = model.n_dense_blocks

        self.remaining_stages = nn.ModuleList(list(model.sparse_stages[cut_idx + 1 :]))
        self.dense_blocks = model.dense_blocks
        self.stage_heads = model.stage_heads
        self.activation = model.activation
        self.stage_dropout = model.stage_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], self.k, self.feature_dim)
        preds = [self.stage_heads[self.cut_idx](x)]

        for i, stage in enumerate(self.remaining_stages):
            x = stage(x)
            x = self.activation(x)
            x = self.stage_dropout(x)
            head_idx = self.cut_idx + 1 + i
            preds.append(self.stage_heads[head_idx](x))

        if self.n_dense_blocks > 0:
            for block in self.dense_blocks:
                x = block(x)
            preds.append(self.stage_heads[-1](x))

        all_preds = torch.stack(preds, dim=1)
        return all_preds.sum(dim=(1, 2)) / (self.H * self.k)


def compute_activations(model, betas: torch.Tensor, cut_idx: int, device: torch.device) -> torch.Tensor:
    """Run front-end (stages 0..cut_idx) and return post-activation hidden activations.

    Returns ``(N, K * dim)`` flattened per-member activations.
    """
    model.eval()
    x = betas.to(device)

    with torch.no_grad():
        if model.input_dropout_layer is not None:
            x = model.input_dropout_layer(x)
        x = x[:, None].expand(-1, model.k, -1)

        for i in range(cut_idx + 1):
            x = model.sparse_stages[i](x)
            x = model.activation(x)
            x = model.stage_dropout(x)

        n, k, dim = x.shape
        return x.reshape(n, k * dim).cpu()


def load_feature_names(mask_dir: str, stage_name: str, edges_path: str | None = None) -> list[str]:
    """Load output feature names for a sparse stage.

    For ``cpg_to_gene``: ``output_index.csv`` has ``gene, gene_idx``.
    For pathway stages: ``output_index.csv`` has ``pathway_id, pathway_id_idx``
    and ``edges.tsv`` maps Reactome IDs to display names.
    """
    output_index_path = os.path.join(mask_dir, "output_index.csv")
    df = pd.read_csv(output_index_path)

    if stage_name == "cpg_to_gene":
        df = df.sort_values("gene_idx")
        return df["gene"].tolist()

    if edges_path is None or not os.path.exists(edges_path):
        df = df.sort_values("pathway_id_idx")
        return df["pathway_id"].tolist()

    edges = pd.read_csv(edges_path, sep="\t")
    stage_edges = edges[edges["hidden_layer"] == stage_name][["target", "target_name"]].drop_duplicates()
    name_map = dict(zip(stage_edges["target"], stage_edges["target_name"], strict=False))
    df = df.sort_values("pathway_id_idx")
    return [name_map.get(pid, pid) for pid in df["pathway_id"].tolist()]


def run_shap_for_layer(
    model,
    background_betas: torch.Tensor,
    test_betas: torch.Tensor,
    layer_key: str,
    spec: dict[str, Any],
    bg_cap: int,
    batch_size: int,
    device: torch.device,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run GradientExplainer for one layer and return importance DataFrame.

    For the gene layer (wide_output=True): wide format (one row per sample,
    one column per gene).
    For pathway layers: long format (one row per sample-pathway pair).
    """
    import shap

    cut_idx = spec["cut_idx"]
    wide_output = spec.get("wide_output", False)
    name_col = spec["name_col"]

    background = compute_activations(model, background_betas, cut_idx, device)
    test = compute_activations(model, test_betas, cut_idx, device)

    n_bg, feat_dim = background.shape
    n_test = test.shape[0]

    if n_bg > bg_cap:
        rng = np.random.default_rng(42)
        bg_indices = rng.choice(n_bg, size=bg_cap, replace=False)
        background = background[bg_indices]
        n_bg = bg_cap

    wrapper_feature_dim = feat_dim // model.k
    wrapper = LayerWrapper(model=model, cut_idx=cut_idx, feature_dim=wrapper_feature_dim).to(device)
    wrapper.eval()

    background = background.to(device)
    test = test.to(device)

    explainer = shap.GradientExplainer(wrapper, background)  # ty: ignore[invalid-argument-type]

    all_shap_values = []
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch = test[start:end]
        sv = explainer.shap_values(batch)
        if isinstance(sv, (list, tuple)):
            sv = sv[0]
        if isinstance(sv, torch.Tensor):
            sv = sv.detach().cpu().numpy()
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv[:, :, 0]
        all_shap_values.append(sv)

    shap_values = np.concatenate(all_shap_values, axis=0)

    per_member_dim = feat_dim // model.k
    per_feature = shap_values.reshape(n_test, model.k, per_member_dim)
    per_feature = per_feature.sum(axis=1)

    with torch.no_grad():
        f_bg = wrapper(background).cpu().numpy().mean()
        f_test = wrapper(test).cpu().numpy().reshape(-1)
    shap_sums = per_feature.sum(axis=1)
    residuals = shap_sums - (f_test - f_bg)
    max_residual = float(np.max(np.abs(residuals)))
    mean_residual = float(np.mean(np.abs(residuals)))

    if wide_output:
        df = pd.DataFrame(per_feature, columns=pd.Index(feature_names))
        df.insert(0, "sample", pd.Series([f"sample_{i}" for i in range(n_test)]))
    else:
        rows = []
        for i in range(n_test):
            for j in range(per_feature.shape[1]):
                rows.append(
                    {
                        "sample": f"sample_{i}",
                        name_col: feature_names[j],
                        "importance": float(per_feature[i, j]),
                    }
                )
        df = pd.DataFrame(rows)

    return df, {"max_residual": max_residual, "mean_residual": mean_residual}


def run_shap_analysis(
    checkpoint_path,
    config_path,
    background_betas,
    test_betas,
    mask_dir,
    output_dir,
    target_layers=None,
    edges_path=None,
    bg_cap=200,
    batch_size=32,
    device="auto",
):
    """Run SHAP analysis at hidden-layer level for an EGBINN model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the trained .pth checkpoint.
    config_path : str
        Path to the YAML config used to train the model.
    background_betas : torch.Tensor or np.ndarray
        Background beta values, shape ``(N_bg, input_dim)``.
    test_betas : torch.Tensor or np.ndarray
        Test beta values, shape ``(N_test, input_dim)``.
    mask_dir : str
        Base directory containing per-stage mask + feature name artifacts.
    output_dir : str
        Directory to save importance parquets.
    target_layers : list[str] or None
        Layer keys to explain (default: all of gene, l2, l1, l0).
    edges_path : str or None
        Path to edges.tsv for pathway display names.
    bg_cap : int
        Maximum number of background samples for GradientExplainer.
    batch_size : int
        Batch size for SHAP computation.
    device : str
        "auto", "cuda", "mps", or "cpu".

    Returns
    -------
    dict
        Mapping from layer key to output file path.
    """
    import shap  # noqa: F401

    config = load_config(config_path)
    device = resolve_device(device)

    input_dim = background_betas.shape[-1]
    model = build_model(config, input_dim, mask_dir)
    model = load_checkpoint(model, checkpoint_path, device)

    if not hasattr(model, "stage_heads") or model.n_heads == 0:
        raise ValueError("SHAP analysis requires EGBINN with regression_approach='multi_stage'.")

    if target_layers is None:
        target_layers = ["gene", "l2", "l1", "l0"]

    if isinstance(background_betas, np.ndarray):
        background_betas = torch.from_numpy(background_betas).float()
    if isinstance(test_betas, np.ndarray):
        test_betas = torch.from_numpy(test_betas).float()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for layer_key in target_layers:
        if layer_key not in LAYER_SPECS:
            logger.warning("Unknown layer: %s, skipping.", layer_key)
            continue

        spec = LAYER_SPECS[layer_key]
        stage_name = spec["stage_name"]
        stage_mask_dir = os.path.join(mask_dir, stage_name)

        try:
            feature_names = load_feature_names(stage_mask_dir, stage_name, edges_path)
        except FileNotFoundError:
            logger.warning("Feature names not found for %s, using indices.", layer_key)
            out_features = model.sparse_stages[spec["cut_idx"]].out_features
            feature_names = [f"feature_{i}" for i in range(out_features)]

        logger.info("Running SHAP for layer: %s...", layer_key)
        df, residual_info = run_shap_for_layer(
            model=model,
            background_betas=background_betas,
            test_betas=test_betas,
            layer_key=layer_key,
            spec=spec,
            bg_cap=bg_cap,
            batch_size=batch_size,
            device=device,
            feature_names=feature_names,
        )

        output_file = os.path.join(output_dir, f"{layer_key}_importance.parquet")
        df.to_parquet(output_file, index=False)
        results[layer_key] = output_file
        logger.info("Saved %s rows to %s", len(df), output_file)
        logger.info("Additivity: max|residual|=%.6f", residual_info["max_residual"])

    summary_path = os.path.join(output_dir, "shap_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"layers": list(results.keys())}, f, indent=2)

    return results
