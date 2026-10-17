"""Tests for data loading: datasets, loaders, validation, and array helpers."""

import numpy as np
import pandas as pd
import pytest
import torch

from egbinn.data import MethylationDataset, infer_input_dim, load_betas_array, load_data


def test_csv_dataset(synthetic_betas, synthetic_metadata):
    dataset = MethylationDataset(methylation_array=synthetic_betas, metadata=synthetic_metadata)
    assert len(dataset) == 50
    item = dataset[0]
    assert item["betas"].shape == (200,)
    assert "label" in item and "label_raw" in item
    assert item["sample"] == "sample_0"


def test_load_data_csv(synthetic_betas, synthetic_metadata, tmp_path):
    csv_path = tmp_path / "methylation.csv"
    df = pd.DataFrame(synthetic_betas, index=synthetic_metadata.index, columns=[f"cpg_{i}" for i in range(200)])
    df.to_csv(csv_path)
    meta_path = tmp_path / "metadata.csv"
    synthetic_metadata.to_csv(meta_path)

    loader, _scaler = load_data(
        methylation_path=str(csv_path), metadata_path=str(meta_path), batch_size=16, data_format="csv"
    )
    batch = next(iter(loader))
    assert batch["betas"].shape == (16, 200)
    assert batch["label"].shape == (16,)
    assert len(batch["sample"]) == 16


def test_memmap_dataset(synthetic_betas, synthetic_metadata, tmp_path):
    n_samples, n_features = synthetic_betas.shape
    memmap_path = tmp_path / "data.memmap"
    mm = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=(n_samples, n_features))
    mm[:] = synthetic_betas
    mm.flush()
    del mm

    sidecar = str(memmap_path).replace(".memmap", "_avg_beta_values.npy")
    np.save(sidecar, np.mean(synthetic_betas, axis=0))

    meta_path = tmp_path / "metadata.csv"
    synthetic_metadata.to_csv(meta_path)

    loader, _scaler = load_data(
        methylation_path=str(memmap_path), metadata_path=str(meta_path), batch_size=16, data_format="memmap"
    )
    batch = next(iter(loader))
    assert batch["betas"].shape == (16, n_features)


def test_memmap_csv_loaders_equivalent(synthetic_betas, synthetic_metadata, tmp_path):
    csv_path = tmp_path / "methylation.csv"
    pd.DataFrame(synthetic_betas, index=synthetic_metadata.index, columns=[f"cpg_{i}" for i in range(200)]).to_csv(
        csv_path
    )

    memmap_path = tmp_path / "data.memmap"
    mm = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=synthetic_betas.shape)
    mm[:] = synthetic_betas
    mm.flush()
    del mm
    np.save(str(memmap_path).replace(".memmap", "_avg_beta_values.npy"), np.mean(synthetic_betas, axis=0))

    meta_path = tmp_path / "metadata.csv"
    synthetic_metadata.to_csv(meta_path)

    csv_loader, csv_scaler = load_data(
        methylation_path=str(csv_path), metadata_path=str(meta_path), batch_size=50, shuffle=False, data_format="csv"
    )
    mm_loader, mm_scaler = load_data(
        methylation_path=str(memmap_path),
        metadata_path=str(meta_path),
        batch_size=50,
        shuffle=False,
        data_format="memmap",
    )
    csv_batch = next(iter(csv_loader))
    mm_batch = next(iter(mm_loader))
    assert torch.allclose(csv_batch["betas"], mm_batch["betas"])
    assert csv_scaler.mean_[0] == pytest.approx(mm_scaler.mean_[0])


def test_imputation():
    betas = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    meta = pd.DataFrame({"age": [30, 50]}, index=["s1", "s2"])
    imp = np.array([0.5, 2.0, 4.0], dtype=np.float32)

    dataset = MethylationDataset(betas, meta, imputation_values=imp)
    item = dataset[0]
    assert not torch.isnan(item["betas"]).any()
    assert item["betas"][1].item() == pytest.approx(2.0)


def test_missing_label_column_raises(synthetic_betas):
    meta = pd.DataFrame({"not_age": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing the label column 'age'"):
        MethylationDataset(np.zeros((3, 200), dtype=np.float32), meta)


def test_empty_metadata_raises():
    with pytest.raises(ValueError, match="Metadata is empty"):
        MethylationDataset(np.zeros((0, 200), dtype=np.float32), pd.DataFrame({"age": []}))


def test_row_count_mismatch_raises(synthetic_metadata):
    betas = np.zeros((10, 200), dtype=np.float32)
    with pytest.raises(ValueError, match="10 rows but metadata has 50"):
        MethylationDataset(betas, synthetic_metadata)


def test_imputation_length_mismatch_raises(synthetic_metadata):
    betas = np.zeros((50, 200), dtype=np.float32)
    imp = np.zeros(199, dtype=np.float32)
    with pytest.raises(ValueError, match="length 199 but the methylation array has 200"):
        MethylationDataset(betas, synthetic_metadata, imputation_values=imp)


def test_infer_input_dim_csv(tmp_path):
    csv_path = tmp_path / "methylation.csv"
    pd.DataFrame(np.zeros((3, 42), dtype=np.float32), columns=[f"c{i}" for i in range(42)]).to_csv(csv_path)
    assert infer_input_dim(str(csv_path)) == 42


def test_infer_input_imputation_priority(tmp_path):
    imp_path = tmp_path / "avg.npy"
    np.save(imp_path, np.zeros(7))
    csv_path = tmp_path / "methylation.csv"
    pd.DataFrame(np.zeros((3, 42), dtype=np.float32), columns=[f"c{i}" for i in range(42)]).to_csv(csv_path)
    assert infer_input_dim(str(csv_path), imputation_path=str(imp_path)) == 7


def test_infer_input_dim_memmap_sidecar(tmp_path):
    memmap_path = tmp_path / "data.memmap"
    mm = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=(4, 99))
    mm.flush()
    del mm
    np.save(str(memmap_path).replace(".memmap", "_avg_beta_values.npy"), np.zeros(99))
    assert infer_input_dim(str(memmap_path)) == 99


def test_infer_input_dim_unresolvable_raises(tmp_path):
    memmap_path = tmp_path / "data.memmap"
    memmap_path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Cannot infer the number of features"):
        infer_input_dim(str(memmap_path))


def test_load_betas_array_memmap_imputes(tmp_path):
    betas = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    memmap_path = tmp_path / "data.memmap"
    mm = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=betas.shape)
    mm[:] = betas
    mm.flush()
    del mm
    imp_path = tmp_path / "avg.npy"
    np.save(imp_path, np.array([0.5, 2.5], dtype=np.float32))

    arr = load_betas_array(str(memmap_path), imputation_path=str(imp_path))
    assert not np.isnan(arr).any()
    assert arr[0, 1] == pytest.approx(2.5)
    assert arr.dtype == np.float32


def test_load_betas_array_csv(tmp_path):
    csv_path = tmp_path / "m.csv"
    pd.DataFrame(np.ones((3, 4), dtype=np.float32), columns=[f"c{i}" for i in range(4)]).to_csv(csv_path)
    arr = load_betas_array(str(csv_path))
    assert arr.shape == (3, 4)
