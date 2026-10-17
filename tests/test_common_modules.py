"""Unit tests for the TabM ensemble primitives in egbinn.models.common."""

import torch
from torch import nn

from egbinn.models.common import (
    LinearEfficientEnsemble,
    NLinear,
    ScaleEnsemble,
    TabMMLP,
    init_random_signs_,
    make_efficient_ensemble,
)


def test_nlinear_matches_independent_linears():
    layer = NLinear(n=3, in_features=5, out_features=4)
    x = torch.randn(6, 3, 5)
    out = layer(x)
    assert out.shape == (6, 3, 4)
    manual = torch.stack([x[:, i] @ layer.weight[i] + layer.bias[i] for i in range(3)], dim=1)
    assert torch.allclose(out, manual, atol=1e-6)


def test_linear_efficient_ensemble_forward():
    k, b, in_f, out_f = 3, 4, 5, 6
    layer = LinearEfficientEnsemble(
        in_f,
        out_f,
        bias=True,
        k=k,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=True,
        scaling_init="ones",
    )
    assert layer.r.shape == (k, in_f)
    assert layer.s.shape == (k, out_f)
    assert layer.bias.shape == (k, out_f)
    x = torch.randn(b, k, in_f)
    out = layer(x)
    assert out.shape == (b, k, out_f)
    manual = ((x * layer.r) @ layer.weight.T) * layer.s + layer.bias
    assert torch.allclose(out, manual, atol=1e-6)


def test_linear_efficient_ensemble_shared_weight():
    """No ensemble adapters: one shared weight + shared per-feature bias across members."""
    layer = LinearEfficientEnsemble(
        4,
        4,
        bias=True,
        k=2,
        ensemble_scaling_in=False,
        ensemble_scaling_out=False,
        ensemble_bias=False,
        scaling_init="ones",
    )
    assert layer.r is None and layer.s is None
    assert layer.bias is not None and layer.bias.shape == (4,)
    x = torch.randn(3, 2, 4)
    manual = torch.stack([x[:, i] @ layer.weight.T + layer.bias for i in range(2)], dim=1)
    assert torch.allclose(layer(x), manual, atol=1e-6)


def test_linear_efficient_ensemble_random_signs_init():
    layer = LinearEfficientEnsemble(
        4,
        4,
        bias=True,
        k=3,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=True,
        scaling_init="random-signs",
    )
    assert torch.all((layer.r == 1) | (layer.r == -1))
    assert torch.all((layer.s == 1) | (layer.s == -1))


def test_scale_ensemble_broadcast():
    x = torch.randn(2, 3, 5)
    ones_layer = ScaleEnsemble(3, 5, init="ones")
    assert torch.allclose(ones_layer(x), x)
    normal_layer = ScaleEnsemble(3, 5, init="normal")
    assert not torch.allclose(normal_layer(x), x)


def test_make_efficient_ensemble_replaces_linears():
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    make_efficient_ensemble(
        module,
        LinearEfficientEnsemble,
        k=2,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=True,
        scaling_init="ones",
    )
    children = list(module.children())
    assert isinstance(children[0], LinearEfficientEnsemble)
    assert isinstance(children[1], nn.ReLU)
    assert isinstance(children[2], LinearEfficientEnsemble)


def test_tabm_mlp_2d_and_3d_equivalent():
    mlp = TabMMLP(d_in=6, d_block=4, n_blocks=3, dropout=0.0)
    mlp.eval()
    x2 = torch.randn(5, 6)
    x3 = x2.unsqueeze(1).expand(5, 2, 6)
    with torch.no_grad():
        y2 = mlp(x2)
        y3 = mlp(x3)
    assert y2.shape == (5, 4)
    assert y3.shape == (5, 2, 4)
    assert torch.allclose(y3, y2.unsqueeze(1).expand(5, 2, 4), atol=1e-6)


def test_tabm_mlp_first_block_injection(synthetic_masks):
    from egbinn.models.sparse import SparseMaskedLinear

    m = synthetic_masks[0]
    first = SparseMaskedLinear(mask_indices=m["mask"], in_features=200, out_features=m["out"])
    mlp = TabMMLP(d_block=8, n_blocks=3, dropout=0.0, first_block=first)
    mlp.eval()
    x = torch.randn(4, 200)
    with torch.no_grad():
        y = mlp(x)
    assert y.shape == (4, 8)


def test_init_random_signs():
    x = torch.ones(1000)
    init_random_signs_(x)
    assert set(torch.unique(x).tolist()) <= {1.0, -1.0}
