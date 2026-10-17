"""Tests for the SHAP analysis module: LayerWrapper, activations, per-layer SHAP, feature names."""

import json
import os

import numpy as np
import pandas as pd
import torch

from egbinn.shap_analysis import (
    LAYER_SPECS,
    LayerWrapper,
    compute_activations,
    load_feature_names,
    run_shap_analysis,
    run_shap_for_layer,
)

DEVICE = torch.device("cpu")


def _front_end(model, betas, cut_idx):
    """Manual replication of compute_activations' front-end pass."""
    x = betas.to(DEVICE)
    with torch.no_grad():
        if model.input_dropout_layer is not None:
            x = model.input_dropout_layer(x)
        x = x[:, None].expand(-1, model.k, -1)
        for i in range(cut_idx + 1):
            x = model.sparse_stages[i](x)
            x = model.activation(x)
            x = model.stage_dropout(x)
    n, k, dim = x.shape
    return x.reshape(n, k * dim)


def test_compute_activations_matches_manual(tiny_shap_setup):
    model = tiny_shap_setup.model
    model.eval()
    betas = torch.from_numpy(tiny_shap_setup.background)
    for cut_idx in (0, 1):
        activations = compute_activations(model, betas, cut_idx, DEVICE)
        expected = _front_end(model, betas, cut_idx)
        assert activations.shape == expected.shape
        assert torch.allclose(activations, expected, atol=1e-5)


def test_layer_wrapper_forward_value(tiny_shap_setup):
    model = tiny_shap_setup.model
    model.eval()
    dim = model.sparse_stages[0].out_features
    z = torch.randn(5, model.k * dim)
    wrapper = LayerWrapper(model=model, cut_idx=0, feature_dim=dim).to(DEVICE)
    wrapper.eval()

    with torch.no_grad():
        out = wrapper(z)
    assert out.shape == (5, 1)

    x = z.reshape(5, model.k, dim)
    with torch.no_grad():
        preds = [model.stage_heads[0](x)]
        for i, stage in enumerate(model.sparse_stages[1:]):
            x = stage(x)
            x = model.activation(x)
            x = model.stage_dropout(x)
            preds.append(model.stage_heads[1 + i](x))
        manual = torch.stack(preds, dim=1).sum(dim=(1, 2)) / (model.n_heads * model.k)
    assert torch.allclose(out, manual.reshape(5, 1), atol=1e-5)


def test_run_shap_for_layer_gene_wide(tiny_shap_setup):
    model = tiny_shap_setup.model
    model.eval()
    n_genes = tiny_shap_setup.n_genes
    n_test = tiny_shap_setup.test.shape[0]

    df, info = run_shap_for_layer(
        model=model,
        background_betas=torch.from_numpy(tiny_shap_setup.background),
        test_betas=torch.from_numpy(tiny_shap_setup.test),
        layer_key="gene",
        spec=LAYER_SPECS["gene"],
        bg_cap=8,
        batch_size=4,
        device=DEVICE,
        feature_names=[f"GENE{i}" for i in range(n_genes)],
    )
    assert df.shape[0] == n_test
    assert df["sample"].iloc[0] == "sample_0"
    assert [c for c in df.columns if c != "sample"] == [f"GENE{i}" for i in range(n_genes)]
    assert np.isfinite(info["max_residual"])
    assert np.isfinite(info["mean_residual"])


def test_run_shap_for_layer_l2_long(tiny_shap_setup):
    model = tiny_shap_setup.model
    model.eval()
    n_l2 = tiny_shap_setup.n_l2
    n_test = tiny_shap_setup.test.shape[0]

    df, _ = run_shap_for_layer(
        model=model,
        background_betas=torch.from_numpy(tiny_shap_setup.background),
        test_betas=torch.from_numpy(tiny_shap_setup.test),
        layer_key="l2",
        spec=LAYER_SPECS["l2"],
        bg_cap=8,
        batch_size=4,
        device=DEVICE,
        feature_names=[f"R-HSA-{i}" for i in range(n_l2)],
    )
    assert len(df) == n_test * n_l2
    assert list(df.columns) == ["sample", "pathway", "importance"]
    assert set(df["pathway"]) == {f"R-HSA-{i}" for i in range(n_l2)}


def test_run_shap_analysis_end_to_end(tiny_shap_setup, tmp_path):
    out_dir = tmp_path / "shap_out"
    results = run_shap_analysis(
        checkpoint_path=tiny_shap_setup.bundle_path,
        config_path=tiny_shap_setup.config_path,
        background_betas=tiny_shap_setup.background,
        test_betas=tiny_shap_setup.test,
        mask_dir=tiny_shap_setup.mask_dir,
        output_dir=str(out_dir),
        target_layers=["gene", "l2"],
        edges_path=None,
        bg_cap=8,
        batch_size=4,
        device="cpu",
    )
    assert set(results) == {"gene", "l2"}
    assert (out_dir / "gene_importance.parquet").exists()
    assert (out_dir / "l2_importance.parquet").exists()

    gene_df = pd.read_parquet(out_dir / "gene_importance.parquet")
    assert "GENE0" in gene_df.columns
    assert "sample" in gene_df.columns

    summary = json.loads((out_dir / "shap_summary.json").read_text())
    assert summary["layers"] == ["gene", "l2"]


def test_load_feature_names(tiny_shap_setup, tmp_path):
    mask_dir = tiny_shap_setup.mask_dir

    gene_names = load_feature_names(os.path.join(mask_dir, "cpg_to_gene"), "cpg_to_gene")
    assert gene_names == [f"GENE{i}" for i in range(tiny_shap_setup.n_genes)]

    l2_ids = load_feature_names(os.path.join(mask_dir, "gene_to_l2"), "gene_to_l2", edges_path=None)
    assert l2_ids == [f"R-HSA-{i}" for i in range(tiny_shap_setup.n_l2)]

    edges = pd.DataFrame(
        {
            "hidden_layer": ["gene_to_l2"] * tiny_shap_setup.n_l2,
            "target": [f"R-HSA-{i}" for i in range(tiny_shap_setup.n_l2)],
            "target_name": [f"Pathway {i}" for i in range(tiny_shap_setup.n_l2)],
        }
    )
    edges_path = tmp_path / "edges.tsv"
    edges.to_csv(edges_path, sep="\t", index=False)
    mapped = load_feature_names(os.path.join(mask_dir, "gene_to_l2"), "gene_to_l2", edges_path=str(edges_path))
    assert mapped == [f"Pathway {i}" for i in range(tiny_shap_setup.n_l2)]
