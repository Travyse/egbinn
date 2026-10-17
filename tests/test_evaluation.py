"""Tests for predict(): bundle path, legacy path, and error remediation."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from egbinn.evaluation import predict
from egbinn.models import MLPRegression


def _write_eval_data(tmp_path, ages, n_features=50, n_samples=8):
    betas = np.random.default_rng(5).uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
    index = [f"e{i}" for i in range(n_samples)]
    data_csv = tmp_path / "betas.csv"
    meta_csv = tmp_path / "metadata.csv"
    imp_npy = tmp_path / "avg.npy"
    pd.DataFrame(betas, index=index, columns=[f"c{i}" for i in range(n_features)]).to_csv(data_csv)
    pd.DataFrame({"age": ages}, index=index).to_csv(meta_csv)
    np.save(imp_npy, np.full(n_features, 0.5, dtype=np.float32))
    return str(data_csv), str(meta_csv), str(imp_npy)


def _make_bundle(tmp_path, scaler=None, config=None, package_version="0.1.0"):
    model = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    if config is None:
        from egbinn.config import config_from_dict, to_plain_dict

        config = to_plain_dict(
            config_from_dict(
                {
                    "model": "MLP",
                    "model_params": {"hidden_size": 16, "num_layers": 2, "input_dropout": None, "dropout": 0.0},
                }
            )
        )
    if scaler is None:
        scaler = StandardScaler().fit(np.linspace(20, 80, 32).reshape(-1, 1))
    from egbinn.inference import extract_label_scaler

    bundle = {
        "state_dict": model.state_dict(),
        "label_scaler": extract_label_scaler(scaler),
        "input_dim": 50,
        "model_name": "MLP",
        "resolved_config": config,
        "package_version": package_version,
        "git_commit": None,
        "best_epoch": 5,
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path / "bundle.pth"
    torch.save(bundle, path)
    return str(path), scaler


def test_predict_with_bundle(tmp_path):
    checkpoint, _scaler = _make_bundle(tmp_path)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    df = predict(
        checkpoint_path=checkpoint, data_path=data_csv, metadata_path=meta_csv, imputation_path=imp_npy, device="cpu"
    )
    assert list(df.columns) == ["sample", "chron_age", "pred_age"]
    assert len(df) == 8
    assert not df["pred_age"].isna().any()


def test_predict_writes_outputs(tmp_path):
    checkpoint, _ = _make_bundle(tmp_path)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))
    out = tmp_path / "out" / "predictions.csv"

    predict(
        checkpoint_path=checkpoint,
        data_path=data_csv,
        metadata_path=meta_csv,
        imputation_path=imp_npy,
        device="cpu",
        output_file=str(out),
    )
    assert out.exists()
    metrics_file = tmp_path / "out" / "predictions_metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert "mae" in metrics


def test_predict_bundle_ignores_config(tmp_path, caplog):
    checkpoint, _ = _make_bundle(tmp_path)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    with caplog.at_level("INFO", logger="egbinn.evaluation"):
        predict(
            checkpoint_path=checkpoint,
            data_path=data_csv,
            metadata_path=meta_csv,
            config_path="whatever.yml",
            imputation_path=imp_npy,
            device="cpu",
        )
    assert any("embeds its own resolved config" in r.message for r in caplog.records)


def test_predict_version_mismatch_warns(tmp_path, caplog):
    checkpoint, _ = _make_bundle(tmp_path, package_version="0.0.0-test")
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    with caplog.at_level("WARNING", logger="egbinn.evaluation"):
        predict(
            checkpoint_path=checkpoint,
            data_path=data_csv,
            metadata_path=meta_csv,
            imputation_path=imp_npy,
            device="cpu",
        )
    assert any("results may differ" in r.message for r in caplog.records)


def test_legacy_checkpoint_requires_config(tmp_path):
    model = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    legacy = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), legacy)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    with pytest.raises(ValueError, match="legacy plain state dict with no embedded config"):
        predict(
            checkpoint_path=str(legacy),
            data_path=data_csv,
            metadata_path=meta_csv,
            imputation_path=imp_npy,
            device="cpu",
        )


def test_legacy_checkpoint_requires_scaler(tmp_path):
    model = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    legacy = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), legacy)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    config_yaml = tmp_path / "config.yml"
    config_yaml.write_text(
        "model: MLP\nmodel_params:\n  hidden_size: 16\n  num_layers: 2\n  input_dropout: null\n  dropout: 0.0\n"
    )

    with pytest.raises(ValueError, match="No label scaler available"):
        predict(
            checkpoint_path=str(legacy),
            data_path=data_csv,
            metadata_path=meta_csv,
            config_path=str(config_yaml),
            imputation_path=imp_npy,
            device="cpu",
        )


def test_legacy_checkpoint_with_config_and_scaler_json(tmp_path):
    model = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    legacy = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), legacy)
    data_csv, meta_csv, imp_npy = _write_eval_data(tmp_path, ages=np.linspace(20, 80, 8))

    config_yaml = tmp_path / "config.yml"
    config_yaml.write_text(
        "model: MLP\nmodel_params:\n  hidden_size: 16\n  num_layers: 2\n  input_dropout: null\n  dropout: 0.0\n"
    )
    scaler = StandardScaler().fit(np.linspace(20, 80, 32).reshape(-1, 1))
    from egbinn.inference import extract_label_scaler

    scaler_json = tmp_path / "label_scaler.json"
    scaler_json.write_text(json.dumps(extract_label_scaler(scaler)))

    df = predict(
        checkpoint_path=str(legacy),
        data_path=data_csv,
        metadata_path=meta_csv,
        config_path=str(config_yaml),
        label_scaler_path=str(scaler_json),
        imputation_path=imp_npy,
        device="cpu",
    )
    assert len(df) == 8


def test_orig_mod_prefix_is_stripped(tmp_path):
    model = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    legacy = tmp_path / "compiled.pth"
    torch.save(state, legacy)

    from egbinn.evaluation import load_checkpoint

    model2 = MLPRegression(input_dim=50, hidden_size=16, num_layers=2)
    load_checkpoint(model2, str(legacy), "cpu")
    model.eval()
    model2.eval()
    betas = torch.randn(4, 50)
    with torch.no_grad():
        _, y1 = model({"betas": betas})
        _, y2 = model2({"betas": betas})
    assert torch.allclose(y1, y2)
