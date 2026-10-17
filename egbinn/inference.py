"""Shared inference utilities: model inference over a DataLoader and metrics.

A single implementation of the inference path used by both the training loop
(epoch-end validation) and final evaluation, plus the label-scaler
serialization helpers shared by training (save) and evaluation (load).
"""

__all__ = [
    "run_inference",
    "compute_metrics",
    "per_dataset_metrics",
    "extract_label_scaler",
    "rebuild_label_scaler",
]

import logging

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def extract_label_scaler(scaler):
    """Serialize a fitted 1-D StandardScaler into a plain, pickle/JSON-safe dict."""
    if scaler is None:
        return None
    mean_ = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
    scale_ = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
    var_ = np.asarray(getattr(scaler, "var_", [0.0]), dtype=np.float64).reshape(-1)
    return {
        "mean_": float(mean_[0]),
        "scale_": float(scale_[0]),
        "var_": float(var_[0]),
        "n_samples_seen_": int(getattr(scaler, "n_samples_seen_", 0) or 0),
    }


def rebuild_label_scaler(d):
    """Reconstruct a fitted StandardScaler from :func:`extract_label_scaler` output."""
    if d is None:
        return None
    scaler = StandardScaler()
    scaler.mean_ = np.asarray([d["mean_"]], dtype=np.float64)
    scaler.var_ = np.asarray([d.get("var_", 0.0)], dtype=np.float64)
    scaler.scale_ = np.asarray([d["scale_"]], dtype=np.float64)
    setattr(scaler, "n_features_in_", 1)
    scaler.n_samples_seen_ = int(d.get("n_samples_seen_", 0))
    return scaler


def compute_metrics(y_true, y_pred):
    """Compute MAE, MSE, MedAE, Pearson R, and Spearman R on the raw age scale.

    Samples with NaN or inf in either ``y_true`` or ``y_pred`` are dropped
    before metric computation (e.g. missing chronological ages in TCGA
    metadata). The returned dict includes ``n_valid`` and ``n_dropped`` for
    transparency.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {}
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int(y_true.size - valid.sum())
    if n_dropped:
        logger.warning(
            "Dropping %d/%d samples with NaN or inf age/prediction from metrics.",
            n_dropped,
            y_true.size,
        )
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {"n_valid": 0, "n_dropped": n_dropped}
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "medae": float(np.median(np.abs(y_true - y_pred))),
        "n_valid": int(valid.sum()),
        "n_dropped": n_dropped,
    }
    if y_true.size > 2:
        r, p = pearsonr(y_true, y_pred)
        metrics["pearson_r"] = float(r)
        metrics["pearson_p"] = float(p)
        rho, _ = spearmanr(y_true, y_pred)
        metrics["spearman_r"] = float(rho)
    return metrics


def per_dataset_metrics(predictions_df, dataset_col="Accession_Code"):
    """Compute per-dataset metrics from a predictions DataFrame.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain ``chron_age``, ``pred_age``, and a dataset column
        (default ``Accession_Code``).
    dataset_col : str
        Column identifying the dataset of each sample.

    Returns
    -------
    pd.DataFrame
        One row per dataset with metrics, the dataset code, and sample count.
    """
    if predictions_df.empty or dataset_col not in predictions_df.columns:
        return pd.DataFrame()
    results = []
    for dataset, group in predictions_df.groupby(dataset_col):
        if group.empty:
            continue
        m = compute_metrics(group["chron_age"], group["pred_age"])
        if m.get("n_valid", 0) == 0:
            logger.warning("Skipping dataset %s: 0 valid samples (all NaN/inf).", dataset)
            continue
        m[dataset_col] = dataset
        m["n_samples"] = len(group)
        results.append(m)
    return pd.DataFrame(results)


@torch.no_grad()
def run_inference(model, adapter, data_loader, label_scaler, device):
    """Run inference on a data loader.

    Parameters
    ----------
    model : nn.Module
    adapter : BaseAdapter
    data_loader : DataLoader
        Yields batches with ``label_raw`` (and optionally ``sample``).
    label_scaler : StandardScaler or None
        Training-time scaler; predictions are inverse-transformed to raw
        years with it. When None, predictions are returned on the model's
        (scaled) output scale.
    device : torch.device

    Returns
    -------
    pd.DataFrame
        Columns ``sample``, ``chron_age``, ``pred_age``. Sample IDs come
        from the loader when provided, else positional indices.
    """
    model.eval()
    samples, ages, preds = [], [], []

    for batch in data_loader:
        batch_gpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = adapter.forward_eval(model, batch_gpu)
        y_pred_scaled = outputs["y_pred"].squeeze(-1).cpu().numpy()

        if label_scaler is not None:
            y_pred_raw = label_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred_raw = y_pred_scaled

        y_true_raw = batch["label_raw"].cpu().numpy()
        ages.extend(y_true_raw.tolist())
        preds.extend(y_pred_raw.tolist())

        if "sample" in batch:
            samples.extend(list(batch["sample"]))

    if samples and len(samples) == len(ages):
        df = pd.DataFrame({"sample": samples, "chron_age": ages, "pred_age": preds})
    else:
        df = pd.DataFrame(
            {
                "sample": [str(i) for i in range(len(ages))],
                "chron_age": ages,
                "pred_age": preds,
            }
        )
    n_nan_pred = int(df["pred_age"].isna().sum())
    if n_nan_pred:
        logger.warning(
            "%d/%d predictions are NaN — check that imputation_path was provided "
            "if the input data contains NaN beta values.",
            n_nan_pred,
            len(df),
        )
    return df
