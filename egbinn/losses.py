__all__ = ["ensemble_huber_loss", "standard_huber_loss"]

import torch.nn.functional as F


def standard_huber_loss(y_pred, y, delta=1.0):
    loss = F.huber_loss(y_pred.squeeze(), y.squeeze(), reduction="sum", delta=delta) / y.size(0)
    return loss


def ensemble_huber_loss(y_pred, y, delta=1.0):
    """Mean-of-losses for ensemble predictions (TabM-style).

    Computes Huber loss independently for each of the ``K`` ensemble members
    and averages (mean of per-member losses), not the loss of the mean
    prediction.

    Parameters
    ----------
    y_pred : Tensor (N, K, 1) or (N, K)
        Per-member predictions from the ensemble.
    y : Tensor (N,) or (N, 1)
        Target values (one per sample, shared across all K members).
    delta : float, default 1.0
        Huber delta parameter.

    Returns
    -------
    Tensor
        Scalar loss = ``(1 / (N*K)) * sum_{n,k} Huber(pred[n,k] - y[n], delta)``.
    """
    K = y_pred.shape[1]
    y_pred_flat = y_pred.squeeze(-1).reshape(-1)  # (N*K,)
    y_expanded = y.squeeze().repeat_interleave(K)  # (N*K,)
    loss = F.huber_loss(y_pred_flat, y_expanded, reduction="sum", delta=delta) / (y.size(0) * K)
    return loss
