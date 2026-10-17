"""Numerical correctness tests for the Huber losses."""

import torch
import torch.nn.functional as F

from egbinn.losses import ensemble_huber_loss, standard_huber_loss


def test_standard_huber_matches_reference():
    y_pred = torch.randn(32, 1)
    y = torch.randn(32)
    delta = 0.36
    ours = standard_huber_loss(y_pred, y, delta=delta)
    ref = F.huber_loss(y_pred.squeeze(), y, reduction="sum", delta=delta) / y.size(0)
    assert torch.allclose(ours, ref)


def test_ensemble_huber_equals_member_mean():
    """The core TabM invariant: per-member loss mean, not loss of the mean."""
    n, k = 16, 4
    delta = 0.7
    y_pred = torch.randn(n, k, 1)
    y = torch.randn(n)

    ours = ensemble_huber_loss(y_pred, y, delta=delta)
    member_losses = [standard_huber_loss(y_pred[:, k], y, delta=delta) for k in range(k)]
    ref = torch.stack(member_losses).mean()
    assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-6)


def test_ensemble_huber_differs_from_loss_of_mean():
    n, k = 16, 4
    y_pred = torch.randn(n, k, 1)
    y = torch.randn(n)
    ours = ensemble_huber_loss(y_pred, y, delta=1.0)
    loss_of_mean = standard_huber_loss(y_pred.mean(dim=1, keepdim=True), y, delta=1.0)
    assert not torch.allclose(ours, loss_of_mean)


def test_ensemble_huber_handles_2d_preds():
    y_pred = torch.randn(8, 4)
    y = torch.randn(8)
    ours_2d = ensemble_huber_loss(y_pred, y)
    ours_3d = ensemble_huber_loss(y_pred.unsqueeze(-1), y)
    assert torch.allclose(ours_2d, ours_3d)
