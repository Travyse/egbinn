"""End-to-end CLI tests: train, evaluate, and shap entry points."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from egbinn.evaluation import load_checkpoint_artifacts


def _invoke(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)


@pytest.fixture
def cli_train_setup(tmp_path):
    """Synthetic CSV dataset + full training config for the train CLI."""
    rng = np.random.default_rng(11)
    n, d = 24, 50
    cols = [f"c{i}" for i in range(d)]

    train = pd.DataFrame(
        rng.uniform(0, 1, (n, d)).astype(np.float32), index=[f"tr_{i}" for i in range(n)], columns=cols
    )
    val = pd.DataFrame(rng.uniform(0, 1, (n, d)).astype(np.float32), index=[f"va_{i}" for i in range(n)], columns=cols)
    meta = pd.DataFrame(
        {
            "age": rng.uniform(20, 80, n),
            "Accession_Code": ["GSE1"] * 12 + ["GSE2"] * 12,
        },
        index=train.index,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    meta_path = data_dir / "metadata.csv"
    avg_path = data_dir / "avg.npy"
    train.to_csv(train_path)
    val.to_csv(val_path)
    meta.to_csv(meta_path)
    np.save(avg_path, np.full(d, 0.5, dtype=np.float32))

    save_dir = tmp_path / "train_out"
    config = {
        "model": "MLP",
        "general": {
            "lr": 0.01,
            "weight_decay": 0.0,
            "batch_size": 16,
            "warmup_epochs": 0,
            "train_epochs": 2,
            "patience": 0,
            "label_column": "age",
            "huber_delta": 1.0,
            "num_workers": 0,
        },
        "model_params": {"hidden_size": 8, "num_layers": 2, "input_dropout": None, "dropout": 0.0},
        "data": {
            "data_format": "csv",
            "metadata_path": str(meta_path),
            "train_methylation_path": str(train_path),
            "val_methylation_path": str(val_path),
            "imputation_path": str(avg_path),
        },
        "output": {"save_dir": str(save_dir), "checkpoint_name": "bundle.pth"},
    }
    config_path = tmp_path / "train_config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return SimpleNamespace(
        config_path=str(config_path),
        save_dir=save_dir,
        checkpoint=str(save_dir / "bundle.pth"),
        avg_path=str(avg_path),
        d=d,
    )


def test_train_cli_end_to_end(cli_train_setup, monkeypatch):
    from egbinn.scripts.train import main

    _invoke(
        monkeypatch,
        ["egbinn-train", "--config", cli_train_setup.config_path, "--device", "cpu", "--seed", 0],
    )
    main()

    assert (cli_train_setup.save_dir / "bundle.pth").exists()
    assert (cli_train_setup.save_dir / "training_summary.json").exists()
    assert (cli_train_setup.save_dir / "train.log").exists()

    artifacts = load_checkpoint_artifacts(cli_train_setup.checkpoint)
    assert artifacts["is_bundle"]
    assert artifacts["model_name"] == "MLP"
    assert artifacts["input_dim"] == cli_train_setup.d
    assert artifacts["label_scaler"] is not None


def test_evaluate_cli_end_to_end(cli_train_setup, monkeypatch, tmp_path):
    from egbinn.scripts.evaluate import main as evaluate_main
    from egbinn.scripts.train import main as train_main

    _invoke(monkeypatch, ["egbinn-train", "--config", cli_train_setup.config_path, "--device", "cpu"])
    train_main()

    rng = np.random.default_rng(12)
    n, d = 16, cli_train_setup.d
    eval_df = pd.DataFrame(
        rng.uniform(0, 1, (n, d)).astype(np.float32),
        index=[f"ev_{i}" for i in range(n)],
        columns=[f"c{i}" for i in range(d)],
    )
    eval_meta = pd.DataFrame(
        {
            "age": rng.uniform(20, 80, n),
            "Accession_Code": ["GSE1"] * 8 + ["GSE2"] * 8,
        },
        index=eval_df.index,
    )
    eval_data = tmp_path / "eval.csv"
    eval_meta_path = tmp_path / "eval_metadata.csv"
    eval_df.to_csv(eval_data)
    eval_meta.to_csv(eval_meta_path)
    results_dir = tmp_path / "results"

    _invoke(
        monkeypatch,
        [
            "egbinn-evaluate",
            "--checkpoint",
            cli_train_setup.checkpoint,
            "--data",
            str(eval_data),
            "--metadata",
            str(eval_meta_path),
            "--imputation_path",
            cli_train_setup.avg_path,
            "--data_format",
            "csv",
            "--device",
            "cpu",
            "--output_dir",
            str(results_dir),
        ],
    )
    evaluate_main()

    df = pd.read_csv(results_dir / "predictions.csv")
    assert list(df.columns) == ["sample", "chron_age", "pred_age"]
    assert len(df) == 16
    assert not df["pred_age"].isna().any()

    assert (results_dir / "predictions_metrics.json").exists()
    assert (results_dir / "evaluate.log").exists()

    per_dataset = pd.read_csv(results_dir / "predictions_per_dataset.csv")
    assert set(per_dataset["Accession_Code"]) == {"GSE1", "GSE2"}


def test_shap_cli_end_to_end(tiny_shap_setup, monkeypatch, tmp_path):
    from egbinn.scripts.shap_analysis import main as shap_main

    def _write_betas(arr, name):
        df = pd.DataFrame(
            arr,
            index=[f"{name}_{i}" for i in range(arr.shape[0])],
            columns=[f"c{i}" for i in range(arr.shape[1])],
        )
        path = tmp_path / f"{name}.csv"
        df.to_csv(path)
        return str(path)

    bg_csv = _write_betas(tiny_shap_setup.background, "bg")
    test_csv = _write_betas(tiny_shap_setup.test, "te")
    out_dir = tmp_path / "shap_cli_out"

    _invoke(
        monkeypatch,
        [
            "egbinn-shap",
            "--checkpoint",
            tiny_shap_setup.bundle_path,
            "--config",
            tiny_shap_setup.config_path,
            "--background_data",
            bg_csv,
            "--test_data",
            test_csv,
            "--mask_dir",
            tiny_shap_setup.mask_dir,
            "--output_dir",
            str(out_dir),
            "--target_layers",
            "gene",
            "l2",
            "--bg_cap",
            "8",
            "--batch_size",
            "4",
            "--device",
            "cpu",
        ],
    )
    shap_main()

    assert (out_dir / "gene_importance.parquet").exists()
    assert (out_dir / "l2_importance.parquet").exists()
    assert (out_dir / "shap_summary.json").exists()
    assert (out_dir / "shap.log").exists()


def test_train_cli_missing_required_args_exits(monkeypatch):
    from egbinn.scripts.train import main

    _invoke(monkeypatch, ["egbinn-train"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


def test_evaluate_cli_help_exits_zero(monkeypatch, capsys):
    from egbinn.scripts.evaluate import main

    _invoke(monkeypatch, ["egbinn-evaluate", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this machine")
def test_train_cli_explicit_cuda_unavailable_raises(cli_train_setup, monkeypatch):
    from egbinn.scripts.train import main

    _invoke(
        monkeypatch,
        ["egbinn-train", "--config", cli_train_setup.config_path, "--device", "cuda"],
    )
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        main()
