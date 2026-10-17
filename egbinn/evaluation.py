"""Evaluation: load checkpoints, run inference, compute metrics.

Checkpoints are *bundles*: weights + the training label scaler + the
resolved config + provenance. ``predict`` is fully self-contained — given a
bundle checkpoint and user data, no other files are needed. Legacy plain
state-dict checkpoints are supported with an explicit config and a label
scaler JSON.
"""

__all__ = [
    "compute_metrics",
    "evaluate_model",
    "load_checkpoint",
    "load_checkpoint_artifacts",
    "per_dataset_metrics",
    "predict",
]

import json
import logging
import os

import torch

from . import __version__ as PACKAGE_VERSION
from .adapters import get_adapter
from .config import AttrDict, config_from_dict, load_config
from .data import infer_input_dim, load_data
from .inference import (
    compute_metrics,
    per_dataset_metrics,
    rebuild_label_scaler,
    run_inference,
)
from .training import build_model
from .utils import resolve_device

logger = logging.getLogger(__name__)


def _load_state(checkpoint_path, device):
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state, state["state_dict"]
    return None, state


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model weights from a bundle or legacy plain state-dict checkpoint.

    ``_orig_mod.`` prefixes (from ``torch.compile``) are stripped. The label
    scaler / config are not touched here — use
    :func:`load_checkpoint_artifacts` for those.
    """
    if isinstance(device, str):
        device = torch.device(device)
    _, state_dict = _load_state(checkpoint_path, device)
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def load_checkpoint_artifacts(checkpoint_path):
    """Load provenance artifacts from a checkpoint.

    Returns
    -------
    dict
        Keys: ``is_bundle``, ``label_scaler`` (a fitted StandardScaler or
        None), ``input_dim``, ``model_name``, ``resolved_config`` (plain
        dict), ``package_version``, ``git_commit``, ``best_epoch``,
        ``trained_at``. For legacy plain state dicts every field except
        ``is_bundle`` is None.
    """
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if not (isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict)):
        return {
            "is_bundle": False,
            "label_scaler": None,
            "input_dim": None,
            "model_name": None,
            "resolved_config": None,
            "package_version": None,
            "git_commit": None,
            "best_epoch": None,
            "trained_at": None,
        }
    return {
        "is_bundle": True,
        "label_scaler": rebuild_label_scaler(state.get("label_scaler")),
        "input_dim": state.get("input_dim"),
        "model_name": state.get("model_name"),
        "resolved_config": state.get("resolved_config"),
        "package_version": state.get("package_version"),
        "git_commit": state.get("git_commit"),
        "best_epoch": state.get("best_epoch"),
        "trained_at": state.get("trained_at"),
    }


def evaluate_model(model, adapter, data_loader, label_scaler, device):
    """Run inference and return ``(predictions_df, metrics_dict)``.

    ``predictions_df`` has columns ``sample``, ``chron_age``, ``pred_age``.
    """
    df = run_inference(model, adapter, data_loader, label_scaler, device)
    metrics = compute_metrics(df["chron_age"], df["pred_age"]) if len(df) else {}
    return df, metrics


def predict(
    checkpoint_path,
    data_path,
    metadata_path,
    config_path=None,
    imputation_path=None,
    data_format=None,
    label_scaler_path=None,
    label_column="age",
    batch_size=64,
    num_workers=0,
    device="auto",
    output_file=None,
):
    """Load a trained model and predict ages on user-provided data.

    The bundle checkpoint is self-contained: weights, the training label
    scaler, the resolved training config, and (for BINN/EGBINN) the
    biological hierarchy masks are all embedded, so ``config_path`` is only
    needed for legacy plain state-dict checkpoints. Predictions are
    inverse-transformed with the *training* scaler — never a scaler fit on
    the evaluation data.

    Parameters
    ----------
    checkpoint_path : str
        Path to a bundle checkpoint (``.pth``).
    data_path : str
        Methylation data (``.memmap`` or ``.csv``).
    metadata_path : str
        Metadata CSV with a ``label_column`` column; its index provides
        sample IDs.
    config_path : str or None
        YAML training config. Required only for legacy checkpoints; ignored
        for bundles (which embed their own resolved config).
    imputation_path : str or None
        ``.npy`` of per-feature mean beta values for NaN imputation.
    data_format : str or None
        ``"memmap"`` or ``"csv"``; auto-detected when None.
    label_scaler_path : str or None
        JSON file (``mean_``, ``scale_``, ``var_``, ``n_samples_seen_``)
        with the training scaler — required only for legacy checkpoints
        without an embedded scaler.
    label_column : str
        Age column name in the metadata.
    batch_size : int
    num_workers : int
    device : str
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    output_file : str or None
        When given, predictions are saved here and metrics to a companion
        ``*_metrics.json``.

    Returns
    -------
    pd.DataFrame
        Columns ``sample``, ``chron_age``, ``pred_age``.
    """
    device = resolve_device(device)

    artifacts = load_checkpoint_artifacts(checkpoint_path)

    scaler = artifacts["label_scaler"]
    if scaler is None and label_scaler_path is not None:
        with open(label_scaler_path) as f:
            scaler = rebuild_label_scaler(json.load(f))

    if artifacts["is_bundle"] and artifacts["resolved_config"]:
        config = config_from_dict(artifacts["resolved_config"])
        if config_path is not None:
            logger.info("Ignoring config_path: the bundle checkpoint embeds its own resolved config.")
    else:
        if config_path is None:
            raise ValueError(
                f"{checkpoint_path} is a legacy plain state dict with no embedded config. "
                "Provide config_path (the YAML config used at training time)."
            )
        config = load_config(config_path)
    if "model" not in config:
        raise ValueError("Config has no 'model' key; cannot determine the model to build.")

    if scaler is None:
        raise ValueError(
            f"No label scaler available for {checkpoint_path}: the checkpoint is a legacy "
            "state dict and no label_scaler_path was provided. Without the training scaler, "
            "predictions would be inverse-transformed with the wrong (evaluation-set) age "
            "statistics. Re-export the checkpoint as a bundle or pass label_scaler_path."
        )

    if artifacts["package_version"] is not None and artifacts["package_version"] != PACKAGE_VERSION:
        logger.warning(
            "Checkpoint was saved with egbinn %s but the installed version is %s; results may differ.",
            artifacts["package_version"],
            PACKAGE_VERSION,
        )
    logger.info(
        "Loaded checkpoint: model=%s input_dim=%s best_epoch=%s package=%s git=%s trained_at=%s",
        artifacts["model_name"] or config.model,
        artifacts["input_dim"],
        artifacts["best_epoch"],
        artifacts["package_version"],
        artifacts["git_commit"],
        artifacts["trained_at"],
    )

    if artifacts["input_dim"] is not None:
        input_dim = int(artifacts["input_dim"])
    else:
        input_dim = infer_input_dim(data_path, imputation_path, data_format)

    mask_dir = config.get("data", AttrDict()).get("mask_dir", None)
    model = build_model(config, input_dim, mask_dir)
    model = load_checkpoint(model, checkpoint_path, device)

    adapter = get_adapter(config.model)

    loader, _ = load_data(
        methylation_path=data_path,
        metadata_path=metadata_path,
        label_column=label_column,
        label_scaler=scaler,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        imputation_path=imputation_path,
        is_validation=True,
        data_format=data_format,
    )

    predictions_df, metrics = evaluate_model(model, adapter, loader, scaler, device)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        predictions_df.to_csv(output_file, index=False)
        metrics_path = str(output_file).replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s predictions to %s; metrics to %s", len(predictions_df), output_file, metrics_path)

    return predictions_df
