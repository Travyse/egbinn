"""End-to-end training tests: train tiny models, verify bundle checkpoints."""

import numpy as np
import pandas as pd
import pytest
import torch

from egbinn.training import build_model, train_model


def _make_stages(masks, layer_type="sparse"):
    stages = []
    for i, m in enumerate(masks):
        stage = {
            "mask_indices": torch.as_tensor(m["mask"], dtype=torch.long),
            "out_features": m["out"],
            "layer_type": layer_type,
            "match_dense_init": True,
            "pos_weights": False,
        }
        if layer_type == "sparse":
            stage["bias"] = True
        else:
            stage.update(
                {
                    "gate_activation": "silu",
                    "value_bias": True,
                    "gate_bias": True,
                    "gate_bias_init": 1.2785,
                    "gate_weight_init": "scaled",
                    "ensemblify": i > 0,
                }
            )
        stages.append(stage)
    return stages


def _train(model, adapter, train_loader, val_loader, config, device):
    return train_model(
        model=model,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        label_scaler=None,
        input_dim=None,
    )


def test_mlp_training(make_config, make_dataloader, tmp_path):
    from egbinn.adapters import MLPAdapter
    from egbinn.models import MLPRegression

    model = MLPRegression(input_dim=200, hidden_size=16, num_layers=2)
    adapter = MLPAdapter()
    loader, _ = make_dataloader()
    config = make_config(output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"})

    summary = _train(model, adapter, loader, None, config, torch.device("cpu"))
    assert summary["best_epoch"] == 2
    assert summary["best_val_mae"] is None
    assert summary["checkpoint_path"].endswith("test.pth")
    assert (tmp_path / "training_summary.json").exists()


def test_binn_training(synthetic_masks, make_config, make_dataloader, tmp_path):
    from egbinn.adapters import BINNAdapter
    from egbinn.models import BINNRegression

    stages = _make_stages(synthetic_masks, "sparse")
    model = BINNRegression(input_dim=200, stages=stages, stage_activation="relu")
    adapter = BINNAdapter()
    loader, _ = make_dataloader()
    config = make_config(
        model="BINN",
        model_params={"stages": stages, "stage_activation": "relu"},
        output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"},
    )

    summary = _train(model, adapter, loader, None, config, torch.device("cpu"))
    assert summary["best_epoch"] > 0
    assert summary["resolved_config"]["model_params"]["stages"][0]["out_features"] == 50


def test_egbinn_training(synthetic_masks, make_config, make_dataloader, tmp_path):
    from egbinn.adapters import EGBINNAdapter
    from egbinn.models import EGBINN

    stages = _make_stages(synthetic_masks, "gated")
    model = EGBINN(input_dim=200, stages=stages, k=4, n_dense_blocks=0, regression_approach="multi_stage")
    adapter = EGBINNAdapter()
    loader, _ = make_dataloader()
    config = make_config(
        model="EGBINN",
        model_params={"stages": stages, "k": 4, "regression_approach": "multi_stage"},
        output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"},
    )

    summary = _train(model, adapter, loader, None, config, torch.device("cpu"))
    assert summary["best_epoch"] > 0


def test_validation_early_stopping_and_best_metrics(make_config, make_dataloader, tmp_path):
    from egbinn.adapters import MLPAdapter
    from egbinn.models import MLPRegression

    model = MLPRegression(input_dim=200, hidden_size=16, num_layers=2)
    adapter = MLPAdapter()
    train_loader, scaler = make_dataloader()
    val_loader, _ = make_dataloader(n_samples=16, batch_size=16, shuffle=False, scaler=scaler)
    config = make_config(
        general={"train_epochs": 3, "patience": 1},
        output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"},
    )

    summary = _train(model, adapter, train_loader, val_loader, config, torch.device("cpu"))
    assert summary["best_val_mae"] is not None
    assert "mae" in summary["best_metrics"]


def test_bundle_checkpoint_round_trip(make_config, make_dataloader, tmp_path):
    """Train, then reload everything (weights + scaler + config) from the bundle alone."""
    from egbinn.adapters import MLPAdapter
    from egbinn.evaluation import load_checkpoint, load_checkpoint_artifacts
    from egbinn.models import MLPRegression

    model = MLPRegression(input_dim=200, hidden_size=16, num_layers=2)
    adapter = MLPAdapter()
    train_loader, scaler = make_dataloader()
    config = make_config(output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"})

    train_model(
        model=model,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        device=torch.device("cpu"),
        label_scaler=scaler,
        input_dim=200,
    )

    checkpoint_path = str(tmp_path / "test.pth")
    artifacts = load_checkpoint_artifacts(checkpoint_path)
    assert artifacts["is_bundle"]
    assert artifacts["model_name"] == "MLP"
    assert artifacts["input_dim"] == 200
    assert artifacts["label_scaler"] is not None
    assert artifacts["label_scaler"].mean_[0] == pytest.approx(scaler.mean_[0])

    from egbinn.config import config_from_dict

    rebuilt_config = config_from_dict(artifacts["resolved_config"])
    model2 = build_model(rebuilt_config, artifacts["input_dim"])
    load_checkpoint(model2, checkpoint_path, "cpu")

    betas = torch.randn(4, 200)
    model.eval()
    model2.eval()
    with torch.no_grad():
        _, y1 = model({"betas": betas})
        _, y2 = model2({"betas": betas})
    assert torch.allclose(y1, y2, atol=1e-6)


def test_pred_age_invariant_to_eval_metadata_ages(make_config, make_dataloader, tmp_path):
    """THE regression test for the inference scaler bug.

    Predictions must depend only on betas + the checkpoint's training
    scaler — never on the age distribution of the evaluation metadata.
    """
    from egbinn.adapters import MLPAdapter
    from egbinn.evaluation import predict
    from egbinn.models import MLPRegression

    n_samples, n_features = 24, 50
    model = MLPRegression(input_dim=n_features, hidden_size=16, num_layers=2)
    adapter = MLPAdapter()
    train_loader, scaler = make_dataloader(n_samples=n_samples, n_features=n_features)
    config = make_config(output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"})
    train_model(
        model=model,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        device=torch.device("cpu"),
        label_scaler=scaler,
        input_dim=n_features,
    )

    betas = np.random.default_rng(5).uniform(0, 1, size=(8, n_features)).astype(np.float32)
    index = [f"e{i}" for i in range(8)]
    eval_a = tmp_path / "eval_a.csv"
    eval_b = tmp_path / "eval_b.csv"
    data_a = tmp_path / "betas_a.csv"
    data_b = tmp_path / "betas_b.csv"
    imputation = tmp_path / "avg.npy"
    np.save(imputation, np.full(n_features, 0.5, dtype=np.float32))

    pd.DataFrame(betas, index=index, columns=[f"c{i}" for i in range(n_features)]).to_csv(data_a)
    pd.DataFrame(betas, index=index, columns=[f"c{i}" for i in range(n_features)]).to_csv(data_b)
    pd.DataFrame({"age": np.linspace(20, 80, 8)}, index=index).to_csv(eval_a)
    pd.DataFrame({"age": np.linspace(40, 90, 8)}, index=index).to_csv(eval_b)

    checkpoint_path = str(tmp_path / "test.pth")
    df_a = predict(
        checkpoint_path=checkpoint_path,
        data_path=str(data_a),
        metadata_path=str(eval_a),
        imputation_path=str(imputation),
        device="cpu",
    )
    df_b = predict(
        checkpoint_path=checkpoint_path,
        data_path=str(data_b),
        metadata_path=str(eval_b),
        imputation_path=str(imputation),
        device="cpu",
    )

    assert np.allclose(df_a["pred_age"].to_numpy(), df_b["pred_age"].to_numpy(), atol=1e-4)
    assert list(df_a["sample"]) == list(df_b["sample"])


def test_build_model_stage_chain_mismatch_raises(synthetic_masks):
    """A stage whose input indices exceed the previous stage's output dim is caught."""
    from egbinn.config import config_from_dict

    stages = _make_stages(synthetic_masks)
    broken = dict(stages[1])
    broken["mask_indices"] = torch.as_tensor(np.stack([np.full(10, 190), np.arange(10)]).astype(np.int64))
    config = config_from_dict(
        {
            "model": "BINN",
            "model_params": {"stages": [stages[0], broken], "stage_activation": "relu"},
        }
    )
    with pytest.raises(ValueError, match="input indices reach .* but the previous"):
        build_model(config, input_dim=200, mask_dir=None)


def test_train_model_warmup_schedule(make_config, make_dataloader, tmp_path):
    """train_model completes with the linear-warmup + cosine SequentialLR branch."""
    from egbinn.adapters import MLPAdapter
    from egbinn.models import MLPRegression

    model = MLPRegression(input_dim=200, hidden_size=16, num_layers=2)
    adapter = MLPAdapter()
    loader, _ = make_dataloader()
    config = make_config(
        general={"warmup_epochs": 1, "train_epochs": 2},
        output={"save_dir": str(tmp_path), "checkpoint_name": "test.pth"},
    )
    summary = _train(model, adapter, loader, None, config, torch.device("cpu"))
    assert summary["best_epoch"] == 2
    assert (tmp_path / "test.pth").exists()


def test_optimizer_param_groups_first_layer_decay(synthetic_masks):
    """A sparse first stage gets its own weight-decay group with the first-layer value."""
    from egbinn.models import BINNRegression
    from egbinn.training import create_optimizer_and_scheduler

    stages = _make_stages(synthetic_masks, "sparse")
    model = BINNRegression(input_dim=200, stages=stages, stage_activation="relu")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=1e-3,
        weight_decay=0.1,
        warmup_epochs=2,
        train_epochs=10,
        first_layer_weight_decay=0.5,
    )
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.5)
    assert optimizer.param_groups[2]["weight_decay"] == pytest.approx(0.0)
    assert len(optimizer.param_groups[1]["params"]) == 1
    assert optimizer.param_groups[1]["params"][0] is model.sparse_stages[0]._weight_flat
    from torch.optim.lr_scheduler import SequentialLR

    assert isinstance(scheduler, SequentialLR)


def test_optimizer_two_groups_when_no_sparse_layer():
    """Dense models get the plain decay/no-decay split and a plain cosine schedule."""
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from egbinn.models import MLPRegression
    from egbinn.training import create_optimizer_and_scheduler

    model = MLPRegression(input_dim=200, hidden_size=8, num_layers=2)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=1e-3,
        weight_decay=0.1,
        warmup_epochs=0,
        train_epochs=5,
        first_layer_weight_decay=0.9,
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)
    assert isinstance(scheduler, CosineAnnealingLR)
