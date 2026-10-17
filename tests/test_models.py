"""Smoke tests: instantiate each model with synthetic masks and verify forward/backward."""

import numpy as np
import torch

from egbinn.adapters import get_adapter


def _make_stages(masks):
    stages = []
    for m in masks:
        stages.append(
            {
                "mask_indices": torch.as_tensor(m["mask"], dtype=torch.long),
                "out_features": m["out"],
                "layer_type": "sparse",
                "match_dense_init": True,
                "pos_weights": False,
                "bias": True,
            }
        )
    return stages


def _make_gated_stages(masks):
    stages = []
    for i, m in enumerate(masks):
        stages.append(
            {
                "mask_indices": torch.as_tensor(m["mask"], dtype=torch.long),
                "out_features": m["out"],
                "layer_type": "gated",
                "gate_activation": "silu",
                "match_dense_init": True,
                "pos_weights": False,
                "value_bias": True,
                "gate_bias": True,
                "gate_bias_init": 1.2785,
                "gate_weight_init": "scaled",
                "ensemblify": i > 0,
            }
        )
    return stages


def test_mlp_forward():
    from egbinn.models import MLPRegression

    model = MLPRegression(input_dim=200, hidden_size=32, num_layers=3, dropout=0.1)
    batch = {"betas": torch.randn(8, 200)}
    latent, y_pred = model(batch)
    assert latent.shape == (8, 32)
    assert y_pred.shape == (8, 1)


def test_binn_forward(synthetic_masks):
    from egbinn.models import BINNRegression

    stages = _make_stages(synthetic_masks)
    model = BINNRegression(input_dim=200, stages=stages, stage_activation="relu", dropout=0.0)
    batch = {"betas": torch.randn(8, 200)}
    preds = model(batch)
    assert preds.shape == (8, len(stages), 1)


def test_egbinn_forward_final(synthetic_masks):
    from egbinn.models import EGBINN

    stages = _make_gated_stages(synthetic_masks)
    model = EGBINN(
        input_dim=200, stages=stages, k=4, n_dense_blocks=0, regression_approach="final", stage_activation="none"
    )
    batch = {"betas": torch.randn(8, 200)}
    preds = model(batch)
    assert preds.shape == (8, 4, 1)


def test_egbinn_forward_multi_stage(synthetic_masks):
    from egbinn.models import EGBINN

    stages = _make_gated_stages(synthetic_masks)
    model = EGBINN(
        input_dim=200, stages=stages, k=4, n_dense_blocks=0, regression_approach="multi_stage", stage_activation="none"
    )
    batch = {"betas": torch.randn(8, 200)}
    preds = model(batch)
    assert preds.ndim == 4
    assert preds.shape[0] == 8
    assert preds.shape[2] == 4
    assert preds.shape[3] == 1


def test_egbinn_backward(synthetic_masks):
    from egbinn.models import EGBINN
    from egbinn.losses import ensemble_huber_loss

    stages = _make_gated_stages(synthetic_masks)
    model = EGBINN(input_dim=200, stages=stages, k=4, n_dense_blocks=0, regression_approach="multi_stage")
    batch = {"betas": torch.randn(8, 200), "label": torch.randn(8)}
    preds = model(batch)
    loss = ensemble_huber_loss(preds[:, 0, :, :], batch["label"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_invalid_model_name():
    try:
        get_adapter("InvalidModel")
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def _unique_mask(rng, n_in, n_out, n_edges):
    pairs = set()
    while len(pairs) < n_edges:
        pairs.add((int(rng.integers(0, n_in)), int(rng.integers(0, n_out))))
    pairs = sorted(pairs)
    return np.array([[p[0] for p in pairs], [p[1] for p in pairs]], dtype=np.int64)


def test_sparse_masked_linear_matches_dense():
    from egbinn.models.sparse import SparseMaskedLinear

    mask = _unique_mask(np.random.default_rng(0), n_in=200, n_out=50, n_edges=300)

    layer = SparseMaskedLinear(mask_indices=mask, in_features=200, out_features=50)
    x = torch.randn(4, 200)
    out = layer(x)
    dense = x @ layer.weight.T + layer.bias
    assert torch.allclose(out, dense, atol=1e-5)


def test_neutral_gate_init_is_identity():
    """With gate_weight_init='zero' and gate_bias_init=1.2785, silu(gate) ~= 1 at init."""
    from egbinn.models.sparse import GatedSparseLinearLayer

    mask = _unique_mask(np.random.default_rng(0), n_in=50, n_out=20, n_edges=200)

    layer = GatedSparseLinearLayer(
        mask_indices=mask,
        in_features=50,
        out_features=20,
        value_bias=True,
        gate_bias=True,
        gate_bias_init=1.2785,
        gate_weight_init="zero",
    )
    x = torch.randn(4, 50)
    _ = layer(x)
    assert torch.allclose(layer._last_gate, torch.ones_like(layer._last_gate), atol=1e-4)


def test_binn_gated_stages_forward(synthetic_masks):
    """BINNRegression's gated-stage branch (non-ensemble GatedSparseLinearLayer)."""
    from egbinn.models import BINNRegression

    stages = _make_gated_stages(synthetic_masks)
    model = BINNRegression(input_dim=200, stages=stages, stage_activation="relu")
    preds = model({"betas": torch.randn(4, 200)})
    assert preds.shape == (4, len(stages), 1)


def test_egbinn_final_with_dense_blocks(synthetic_masks):
    from egbinn.models import EGBINN

    stages = _make_gated_stages(synthetic_masks)
    model = EGBINN(
        input_dim=200,
        stages=stages,
        k=4,
        n_dense_blocks=1,
        regression_approach="final",
        stage_activation="none",
    )
    preds = model({"betas": torch.randn(4, 200)})
    assert preds.shape == (4, 4, 1)
    assert len(model.dense_blocks) == 1


def test_egbinn_random_signs_fallback_to_dense_blocks(synthetic_masks):
    """With no ensemblified sparse stage, symmetry breaking falls to the first dense block."""
    from egbinn.models import EGBINN

    stages = []
    for m in synthetic_masks:
        stages.append(
            {
                "mask_indices": torch.as_tensor(m["mask"], dtype=torch.long),
                "out_features": m["out"],
                "layer_type": "gated",
                "gate_activation": "silu",
                "match_dense_init": True,
                "pos_weights": False,
                "value_bias": True,
                "gate_bias": True,
                "gate_bias_init": 1.2785,
                "ensemblify": False,
            }
        )
    model = EGBINN(
        input_dim=200,
        stages=stages,
        k=4,
        n_dense_blocks=1,
        regression_approach="final",
        stage_activation="none",
    )
    r = model.dense_blocks[0][0].r
    assert r is not None
    assert torch.all((r == 1) | (r == -1))


def test_sparse_weight_view():
    from egbinn.models.sparse import SparseMaskedLinear

    mask = _unique_mask(np.random.default_rng(0), n_in=200, n_out=50, n_edges=300)
    layer = SparseMaskedLinear(mask_indices=mask, in_features=200, out_features=50)
    w = layer.weight_sparse()
    assert tuple(w.shape) == (50, 200)
    assert w._nnz() == 300


def test_gate_weight_sparse_view():
    from egbinn.models.sparse import GatedSparseLinearLayer

    mask = _unique_mask(np.random.default_rng(1), n_in=50, n_out=20, n_edges=200)
    layer = GatedSparseLinearLayer(mask_indices=mask, in_features=50, out_features=20)
    w = layer.gate_weight_sparse()
    assert tuple(w.shape) == (20, 50)
    assert w._nnz() == 200


def test_pos_weights_init_nonnegative():
    from egbinn.models.sparse import SparseMaskedLinear

    mask = _unique_mask(np.random.default_rng(2), n_in=40, n_out=12, n_edges=30)
    layer = SparseMaskedLinear(mask_indices=mask, in_features=40, out_features=12, pos_weights=True)
    assert float(layer._weight_flat.min()) >= 0.0
    assert float(layer.bias.min()) >= 0.0
