"""Core training loop for EGBINN aging clocks.

A clean PyTorch training loop with early stopping. ``train_model`` saves a
*self-contained bundle checkpoint*: weights, the training label scaler, the
resolved config (with the biological hierarchy masks inlined for
BINN/EGBINN), and provenance (package version, git commit, timestamps) —
so evaluation needs nothing but the checkpoint file itself.
"""

__all__ = [
    "build_model",
    "create_optimizer_and_scheduler",
    "train_model",
]

import json
import os
from datetime import datetime, timezone
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import __version__ as PACKAGE_VERSION
from .config import to_plain_dict
from .inference import compute_metrics, extract_label_scaler, run_inference
from .utils import get_git_commit


def _load_mask_indices(mask_path):
    if str(mask_path).endswith(".pt"):
        return torch.load(mask_path, map_location="cpu", weights_only=True)
    return torch.as_tensor(np.load(mask_path), dtype=torch.long)


def _load_stage_out_features(mask_path):
    shapes_path = os.path.join(os.path.dirname(str(mask_path)), "shapes.json")
    if os.path.exists(shapes_path):
        with open(shapes_path) as f:
            return int(json.load(f)["out_features"])
    idx = _load_mask_indices(mask_path)
    return int(idx[1].max().item()) + 1


def _resolve_stages(stage_configs, mask_dir, input_dim):
    """Build fully-resolved stage dicts: mask indices loaded, out_features fixed.

    Stages that already carry ``mask_indices`` (e.g. from a bundle
    checkpoint's embedded config) are used as-is; otherwise masks are
    loaded from ``mask_path`` or ``{mask_dir}/{stage name}/mask_indices.npy``.

    Raises
    ------
    ValueError
        When a stage has no reachable mask, or the mask indices are
        inconsistent with the previous stage's output dimension (a
        mismatched-hierarchy export).
    """
    loaded_stages = []
    running_in = input_dim
    for i, stage in enumerate(stage_configs):
        stage_copy = dict(stage)
        stage_name = stage_copy.get("name", f"stage_{i}")

        if "mask_indices" in stage_copy:
            if "out_features" not in stage_copy:
                raise ValueError(f"Stage '{stage_name}' provides mask_indices but not out_features.")
            mask_indices = torch.as_tensor(stage_copy["mask_indices"], dtype=torch.long)
            out_features = int(stage_copy["out_features"])
        else:
            mask_path = stage_copy.pop("mask_path", None)
            if mask_path is None and mask_dir is not None:
                mask_path = os.path.join(mask_dir, stage_name, "mask_indices.npy")
            if mask_path is None:
                raise ValueError(f"Stage '{stage_name}' has no mask_path and no mask_dir provided.")
            mask_indices = _load_mask_indices(mask_path)
            out_features = _load_stage_out_features(mask_path)

        if mask_indices.numel() > 0:
            max_in = int(mask_indices[0].max().item()) + 1
            if max_in > running_in:
                raise ValueError(
                    f"Stage '{stage_name}' input indices reach {max_in - 1} but the previous "
                    f"layer only produces {running_in} outputs. The mask artifacts appear "
                    f"to come from a different hierarchy export."
                )
            if int(mask_indices[1].max().item()) >= out_features:
                raise ValueError(
                    f"Stage '{stage_name}' output indices exceed out_features={out_features} declared in shapes.json."
                )

        stage_copy["mask_indices"] = mask_indices
        stage_copy["out_features"] = out_features
        loaded_stages.append(stage_copy)
        running_in = out_features
    return loaded_stages


def build_model(config, input_dim, mask_dir=None):
    """Construct a model from config.

    Parameters
    ----------
    config : AttrDict
        Loaded YAML config (or a bundle's embedded resolved config) with
        ``model`` (name) and ``model_params``.
    input_dim : int
        Number of input features (CpG beta values).
    mask_dir : str or None
        Base directory for biological hierarchy masks. Stage configs may
        reference masks via ``mask_path`` (explicit) or ``name`` (resolved
        as ``{mask_dir}/{name}/mask_indices.npy``). Ignored when stages
        already carry inlined ``mask_indices``.

    Returns
    -------
    nn.Module
    """
    model_name = config.model
    params = config.model_params

    if model_name == "MLP":
        from .models import MLPRegression

        return MLPRegression(
            input_dim=input_dim,
            hidden_size=params.get("hidden_size", 32),
            num_layers=params.get("num_layers", 5),
            input_dropout=params.get("input_dropout", None),
            dropout=params.get("dropout", 0.0),
        )

    from .models import BINNRegression, EGBINN

    stage_configs = params.get("stages", [])
    if not stage_configs:
        raise ValueError(f"{model_name} requires 'stages' in model_params.")
    loaded_stages = _resolve_stages(stage_configs, mask_dir, input_dim)

    if model_name == "BINN":
        return BINNRegression(
            input_dim=input_dim,
            stages=loaded_stages,
            stage_activation=params.get("stage_activation", "relu"),
            dropout=params.get("dropout", 0.0),
            input_dropout=params.get("input_dropout", None),
            gate_weight_init=params.get("gate_weight_init", "default"),
            gate_weight_scale=params.get("gate_weight_scale", 0.01),
        )

    if model_name == "EGBINN":
        return EGBINN(
            input_dim=input_dim,
            stages=loaded_stages,
            k=params.get("k", 16),
            d_block_mult=params.get("d_block_mult", 1.0),
            n_dense_blocks=params.get("n_dense_blocks", 1),
            dropout=params.get("dropout", 0.0),
            input_dropout=params.get("input_dropout", None),
            share_training_batches=params.get("share_training_batches", True),
            stage_activation=params.get("stage_activation", "none"),
            regression_approach=params.get("regression_approach", "final"),
            gate_weight_init=params.get("gate_weight_init", "default"),
            gate_weight_scale=params.get("gate_weight_scale", 0.01),
        )

    raise ValueError(f"Unknown model: {model_name}. Expected 'MLP', 'BINN', or 'EGBINN'.")


def create_optimizer_and_scheduler(model, lr, weight_decay, warmup_epochs, train_epochs, first_layer_weight_decay=None):
    """Create AdamW with a cosine schedule (optional linear warmup).

    Applies no weight decay to biases and 1-D parameters, and a separate
    weight decay to the first sparse stage's flat weights (if present).
    """
    first_layer_param_ids = set()
    sparse_stages = getattr(model, "sparse_stages", None)
    if sparse_stages is not None and len(sparse_stages) > 0:
        from .models.sparse import GatedSparseLinearLayer, SparseMaskedLinear

        stage0 = sparse_stages[0]
        if isinstance(stage0, SparseMaskedLinear):
            first_layer_param_ids.add(id(stage0._weight_flat))
        elif isinstance(stage0, GatedSparseLinearLayer):
            first_layer_param_ids.add(id(stage0._value_weight_flat))
            first_layer_param_ids.add(id(stage0._gate_weight_flat))
        for adapter_name in ("r", "s"):
            adapter_param = getattr(stage0, adapter_name, None)
            if adapter_param is not None:
                first_layer_param_ids.add(id(adapter_param))

    if first_layer_weight_decay is None:
        first_layer_weight_decay = weight_decay

    decay_params = []
    first_layer_decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in first_layer_param_ids:
            first_layer_decay_params.append(param)
        elif name.endswith("_weight_flat"):
            decay_params.append(param)
        elif param.ndim == 1 or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if first_layer_decay_params:
        param_groups.insert(
            1,
            {
                "params": first_layer_decay_params,
                "weight_decay": first_layer_weight_decay,
            },
        )

    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, train_epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, train_epochs))

    return optimizer, scheduler


def _snapshot_state_dict(model):
    return {
        k: v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v.clone() for k, v in model.state_dict().items()
    }


def _recover_input_dim_from_model(model):
    sparse = getattr(model, "sparse_stages", None)
    if sparse is not None and len(sparse) > 0:
        return int(sparse[0].in_features)
    mlp = getattr(model, "mlp", None)
    if mlp is not None:
        first = mlp.layers[0][0]
        if isinstance(first, nn.Linear):
            return int(first.in_features)
        if hasattr(first, "down"):
            return int(first.down.in_features)
    raise ValueError("Cannot recover input_dim from the model; pass input_dim explicitly.")


def _resolved_stages_from_model(model, stage_cfgs):
    """Rebuild fully-resolved stage dicts (with inlined mask tensors) from the model itself."""
    sparse = getattr(model, "sparse_stages", None)
    if sparse is None or len(sparse) == 0:
        return None
    if len(sparse) != len(stage_cfgs):
        raise ValueError(f"Model has {len(sparse)} sparse stages but the config lists {len(stage_cfgs)}.")
    out = []
    for layer, cfg in zip(sparse, stage_cfgs, strict=True):
        s = dict(cfg)
        if hasattr(layer, "cpg_idx"):
            s["mask_indices"] = torch.stack([layer.cpg_idx, layer.gene_idx])
        else:
            s["mask_indices"] = torch.stack([layer.in_idx, layer.out_idx])
        s["out_features"] = int(layer.out_features)
        out.append(s)
    return out


def train_model(
    model, adapter, train_loader, val_loader, config, device, logger=None, label_scaler=None, input_dim=None
):
    """Train a model with early stopping on validation MAE.

    Parameters
    ----------
    model : nn.Module
    adapter : BaseAdapter
    train_loader : DataLoader
    val_loader : DataLoader or None
        If None, no validation runs and the final-epoch checkpoint is saved.
    config : AttrDict
        Must contain ``general`` (lr, weight_decay, train_epochs, patience,
        huber_delta, ...) and ``output`` (save_dir, checkpoint_name).
    device : torch.device or str
    logger : logging.Logger or None
    label_scaler : StandardScaler or None
        The training-fit scaler — embedded in the saved checkpoint so
        evaluation can inverse-transform predictions correctly.
    input_dim : int or None
        Number of input features; recovered from the model when None.

    Returns
    -------
    dict
        Training summary with best epoch, metrics, checkpoint path, and provenance.
    """
    general = config.general
    lr = float(general.get("lr", 0.0001))
    weight_decay = float(general.get("weight_decay", 0.0))
    first_layer_wd = general.get("first_layer_weight_decay", weight_decay)
    warmup_epochs = int(general.get("warmup_epochs", 0))
    train_epochs = int(general.get("train_epochs", 100))
    patience = int(general.get("patience", 30))
    huber_delta = float(general.get("huber_delta", 1.0))
    save_dir = config.output.get("save_dir", "./outputs")
    checkpoint_name = config.output.get("checkpoint_name", "checkpoint.pth")

    os.makedirs(save_dir, exist_ok=True)
    if isinstance(device, str):
        device = torch.device(device)

    resolved_input_dim = int(input_dim) if input_dim is not None else _recover_input_dim_from_model(model)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info("Total parameters: %s, Trainable: %s", total_params, trainable_params)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, lr, weight_decay, warmup_epochs, train_epochs, first_layer_wd
    )

    loss_func_params = {"huber_delta": huber_delta}

    best_val_mae = float("inf")
    best_epoch = 0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    start_time = time()

    for epoch in range(train_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = adapter.forward_train(model, batch)
            loss, _ = adapter.compute_loss(model, batch, outputs, loss_func_params)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if val_loader is not None:
            val_df = run_inference(model, adapter, val_loader, label_scaler, device)
            val_metrics = compute_metrics(val_df["chron_age"], val_df["pred_age"])
            val_mae = val_metrics.get("mae", float("inf"))

            if logger:
                logger.info(
                    "Epoch %s/%s - train_loss=%.4f, val_mae=%.4f, val_mse=%.4f, val_pearson_r=%.4f",
                    epoch + 1,
                    train_epochs,
                    avg_loss,
                    val_mae,
                    val_metrics.get("mse", 0.0),
                    val_metrics.get("pearson_r", 0.0),
                )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                best_state = _snapshot_state_dict(model)
                best_metrics = dict(val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                if logger:
                    logger.info("Early stopping at epoch %s (patience=%s)", epoch + 1, patience)
                break
        elif logger:
            logger.info("Epoch %s/%s - train_loss=%.4f", epoch + 1, train_epochs, avg_loss)

    if best_state is None:
        best_state = _snapshot_state_dict(model)
        best_epoch = train_epochs

    has_val = best_val_mae != float("inf")
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    resolved_stages = _resolved_stages_from_model(model, config.model_params.get("stages", []))
    bundle_config = to_plain_dict(config)
    if resolved_stages is not None:
        bundle_config["model_params"]["stages"] = resolved_stages

    bundle = {
        "state_dict": best_state,
        "label_scaler": extract_label_scaler(label_scaler),
        "input_dim": resolved_input_dim,
        "model_name": str(config.model),
        "resolved_config": bundle_config,
        "package_version": PACKAGE_VERSION,
        "git_commit": get_git_commit(),
        "best_epoch": int(best_epoch),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(bundle, checkpoint_path)
    if logger:
        logger.info("Saved checkpoint (best epoch %s) to %s", best_epoch, checkpoint_path)

    summary = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae if has_val else None,
        "best_metrics": best_metrics,
        "checkpoint_path": checkpoint_path,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_dim": resolved_input_dim,
        "resolved_config": to_plain_dict(config),
        "label_scaler": bundle["label_scaler"],
        "package_version": PACKAGE_VERSION,
        "git_commit": bundle["git_commit"],
        "torch_version": torch.__version__,
        "device": str(device),
        "runtime_sec": round(time() - start_time, 2),
    }
    with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
