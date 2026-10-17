"""Model adapters: bridge between model forward outputs and the training loop.

Each adapter handles the model-specific forward pass and loss computation so
the training loop and inference stay model-agnostic.
"""

__all__ = ["BaseAdapter", "MLPAdapter", "BINNAdapter", "EGBINNAdapter", "get_adapter"]

import torch

from .losses import ensemble_huber_loss, standard_huber_loss


class BaseAdapter:
    """Minimal interface for model/task-specific behavior."""

    def forward_train(self, model, batch):
        raise NotImplementedError

    def forward_val(self, model, batch):
        return self.forward_train(model, batch)

    def forward_eval(self, model, batch):
        return self.forward_train(model, batch)

    def compute_loss(self, model, batch, outputs, loss_func_params=None):
        raise NotImplementedError


class MLPAdapter(BaseAdapter):
    """Adapter for MLPRegression.

    Model forward returns ``(latent, y_pred)`` where ``y_pred`` is ``(B, 1)``.
    """

    def forward_train(self, model, batch):
        _, y_pred = model(batch)
        return {"y_pred": y_pred}

    def compute_loss(self, model, batch, outputs, loss_func_params=None):
        params = loss_func_params or {}
        delta = params.get("huber_delta", 1.0)
        loss = standard_huber_loss(outputs["y_pred"], batch["label"], delta=delta)
        return loss, {"total": loss.item()}


class BINNAdapter(BaseAdapter):
    """Adapter for BINNRegression.

    Model forward returns ``(B, H, 1)`` — one prediction per stage head.
    The adapter averages over H for prediction and computes per-stage Huber
    loss (mean of the H losses, so every stage head gets gradients).
    """

    def forward_train(self, model, batch):
        preds = model(batch)
        y_pred = preds.mean(dim=1)
        return {"y_pred": y_pred, "raw_preds": preds}

    def compute_loss(self, model, batch, outputs, loss_func_params=None):
        params = loss_func_params or {}
        delta = params.get("huber_delta", 1.0)
        loss = ensemble_huber_loss(outputs["raw_preds"], batch["label"], delta=delta)
        return loss, {"total": loss.item()}


class EGBINNAdapter(BaseAdapter):
    """Adapter for EGBINN (TabM ensemble).

    Model forward returns ``(B, K, 1)`` (final) or ``(B, H, K, 1)``
    (multi_stage). The adapter computes a per-member ensemble loss for
    training and averages over K (or H and K) for reporting.
    """

    def forward_train(self, model, batch):
        preds = model(batch)
        if preds.ndim == 4:
            y_pred = preds.mean(dim=(1, 2))
        else:
            y_pred = preds.mean(dim=1)
        return {"y_pred": y_pred, "raw_preds": preds}

    def compute_loss(self, model, batch, outputs, loss_func_params=None):
        params = loss_func_params or {}
        delta = params.get("huber_delta", 1.0)
        raw_preds = outputs["raw_preds"]
        if raw_preds.ndim == 4:
            n_stages = raw_preds.shape[1]
            stage_losses = [
                ensemble_huber_loss(raw_preds[:, h, :, :], batch["label"], delta=delta) for h in range(n_stages)
            ]
            loss = torch.stack(stage_losses).mean()
        else:
            loss = ensemble_huber_loss(raw_preds, batch["label"], delta=delta)
        return loss, {"total": loss.item()}


_ADAPTER_MAP = {
    "MLP": MLPAdapter,
    "BINN": BINNAdapter,
    "EGBINN": EGBINNAdapter,
}


def get_adapter(model_name):
    """Return an adapter instance by model name."""
    cls = _ADAPTER_MAP.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model name: {model_name}. Expected one of {list(_ADAPTER_MAP)}.")
    return cls()
