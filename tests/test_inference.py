"""Tests for the shared inference path and label-scaler serialization."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from egbinn.adapters import MLPAdapter
from egbinn.inference import (
    compute_metrics,
    extract_label_scaler,
    per_dataset_metrics,
    rebuild_label_scaler,
    run_inference,
)
from egbinn.models import MLPRegression


def _make_model_and_loader(n_samples=32, n_features=50, batch_size=16):
    from torch.utils.data import DataLoader

    from egbinn.data import MethylationDataset

    betas = np.random.default_rng(3).uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
    ages = np.random.default_rng(3).uniform(20, 80, size=n_samples)
    meta = pd.DataFrame({"age": ages}, index=[f"s{i}" for i in range(n_samples)])
    dataset = MethylationDataset(betas, meta)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = MLPRegression(input_dim=n_features, hidden_size=8, num_layers=2)
    return model, loader, dataset.label_scaler


def test_compute_metrics_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(20, 90, size=100)
    y_pred = y_true + rng.normal(0, 5, size=100)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert metrics["mse"] == pytest.approx(float(np.mean((y_true - y_pred) ** 2)))
    assert "pearson_r" in metrics and "spearman_r" in metrics


def test_compute_metrics_empty():
    assert compute_metrics([], []) == {}


def test_run_inference_schema():
    model, loader, scaler = _make_model_and_loader()
    adapter = MLPAdapter()
    df = run_inference(model, adapter, loader, scaler, torch.device("cpu"))
    assert list(df.columns) == ["sample", "chron_age", "pred_age"]
    assert len(df) == 32
    assert df["sample"].iloc[0] == "s0"


def test_run_inference_predictions_on_raw_age_scale():
    model, loader, scaler = _make_model_and_loader()
    adapter = MLPAdapter()
    df = run_inference(model, adapter, loader, scaler, torch.device("cpu"))
    scaled_df = run_inference(model, adapter, loader, None, torch.device("cpu"))
    manual = scaler.inverse_transform(scaled_df["pred_age"].to_numpy().reshape(-1, 1)).flatten()
    assert np.allclose(df["pred_age"].to_numpy(), manual)


def test_label_scaler_round_trip():
    ages = np.random.default_rng(1).uniform(10, 90, size=500).reshape(-1, 1)
    scaler = StandardScaler().fit(ages)

    d = extract_label_scaler(scaler)
    rebuilt = rebuild_label_scaler(d)

    probe = np.array([[42.0]])
    assert rebuilt.transform(probe)[0, 0] == pytest.approx(scaler.transform(probe)[0, 0])
    assert rebuilt.inverse_transform(np.array([[0.5]]))[0, 0] == pytest.approx(
        scaler.inverse_transform(np.array([[0.5]]))[0, 0]
    )


def test_label_scaler_extract_rebuild_none():
    assert extract_label_scaler(None) is None
    assert rebuild_label_scaler(None) is None


def test_per_dataset_metrics():
    df = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(6)],
            "chron_age": [10, 20, 30, 40, 50, 60],
            "pred_age": [11, 19, 32, 38, 52, 58],
            "Accession_Code": ["A", "A", "A", "B", "B", "B"],
        }
    )
    per = per_dataset_metrics(df)
    assert set(per["Accession_Code"]) == {"A", "B"}
    a_row = per[per["Accession_Code"] == "A"].iloc[0]
    assert a_row["mae"] == pytest.approx((1 + 1 + 2) / 3)
    assert a_row["n_samples"] == 3


def test_per_dataset_metrics_missing_column_returns_empty():
    df = pd.DataFrame({"chron_age": [1], "pred_age": [2]})
    assert per_dataset_metrics(df).empty


def test_compute_metrics_drops_nan(caplog):
    y_true = np.array([50.0, np.nan, 60.0, 70.0])
    y_pred = np.array([52.0, 55.0, np.nan, 68.0])
    with caplog.at_level("WARNING", logger="egbinn.inference"):
        metrics = compute_metrics(y_true, y_pred)
    assert metrics["n_valid"] == 2
    assert metrics["n_dropped"] == 2
    assert "mae" in metrics
    assert any("Dropping 2/4" in r.message for r in caplog.records)


def test_compute_metrics_all_nan():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([1.0, 2.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["n_valid"] == 0
    assert metrics["n_dropped"] == 2
    assert "mae" not in metrics


def test_compute_metrics_drops_inf():
    y_true = np.array([50.0, np.inf, 60.0])
    y_pred = np.array([52.0, 55.0, 61.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["n_valid"] == 2
    assert metrics["n_dropped"] == 1


def test_per_dataset_metrics_skips_all_nan_group():
    df = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(5)],
            "chron_age": [50.0, np.nan, np.nan, 60.0, 70.0],
            "pred_age": [52.0, 55.0, 58.0, 61.0, 68.0],
            "Accession_Code": ["A", "B", "B", "A", "A"],
        }
    )
    per = per_dataset_metrics(df)
    assert set(per["Accession_Code"]) == {"A"}
    assert per[per["Accession_Code"] == "A"].iloc[0]["n_valid"] == 3
