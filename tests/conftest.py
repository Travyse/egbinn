import numpy as np
import pandas as pd
import pytest

from egbinn.utils import seed_everything


@pytest.fixture(autouse=True)
def _seed_everything():
    seed_everything(42)


@pytest.fixture
def synthetic_masks():
    """Create 4-stage synthetic biological hierarchy masks."""
    rng = np.random.default_rng(42)
    n_cpgs, n_genes, n_l2, n_l1, n_l0 = 200, 50, 20, 10, 5

    def _make_mask(n_in, n_out, density=0.3):
        n_edges = max(1, int(n_in * n_out * density))
        in_idx = rng.integers(0, n_in, size=n_edges)
        out_idx = rng.integers(0, n_out, size=n_edges)
        return np.stack([in_idx, out_idx]).astype(np.int64)

    return [
        {"name": "cpg_to_gene", "mask": _make_mask(n_cpgs, n_genes), "out": n_genes},
        {"name": "gene_to_l2", "mask": _make_mask(n_genes, n_l2), "out": n_l2},
        {"name": "l2_to_l1", "mask": _make_mask(n_l2, n_l1), "out": n_l1},
        {"name": "l1_to_l0", "mask": _make_mask(n_l1, n_l0), "out": n_l0},
    ]


@pytest.fixture
def synthetic_metadata():
    """Small metadata DataFrame with sample-ID index, age, and dataset code."""
    return pd.DataFrame(
        {
            "age": np.random.default_rng(42).uniform(20, 80, size=50),
            "Accession_Code": ["GSE00001"] * 50,
        },
        index=[f"sample_{i}" for i in range(50)],
    )


@pytest.fixture
def synthetic_betas():
    return np.random.default_rng(42).uniform(0, 1, size=(50, 200)).astype(np.float32)


@pytest.fixture
def make_config():
    """Factory for in-memory training configs (AttrDict)."""
    from egbinn.config import config_from_dict

    def _make(model="MLP", model_params=None, general=None, data=None, output=None):
        base_general = {
            "lr": 0.001,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "train_epochs": 2,
            "patience": 0,
            "huber_delta": 1.0,
            "label_column": "age",
        }
        base_general.update(general or {})
        return config_from_dict(
            {
                "model": model,
                "general": base_general,
                "model_params": model_params or {"hidden_size": 16, "num_layers": 2, "dropout": 0.0},
                "data": data or {},
                "output": output or {},
            }
        )

    return _make


@pytest.fixture
def make_dataloader():
    """Factory for synthetic DataLoaders, (loader, scaler)."""
    from torch.utils.data import DataLoader

    from egbinn.data import MethylationDataset

    def _make(n_samples=32, n_features=200, ages=None, batch_size=16, shuffle=True, scaler=None):
        betas = np.random.default_rng(7).uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
        if ages is None:
            ages = np.random.default_rng(7).uniform(20, 80, size=n_samples)
        meta = pd.DataFrame({"age": ages}, index=[f"sample_{i}" for i in range(n_samples)])
        dataset = MethylationDataset(betas, meta, label_column="age", label_scaler=scaler)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset.label_scaler

    return _make


@pytest.fixture
def tiny_shap_setup(tmp_path, make_dataloader):
    """Tiny trained multi_stage EGBINN + mask artifacts for the SHAP tests.

    Builds a 2-stage gated hierarchy (40 CpGs -> 12 genes -> 5 L2 pathways),
    trains the model for 2 epochs on synthetic data (producing a bundle
    checkpoint), and writes the mask/feature-name artifacts plus a config
    YAML the SHAP tooling can consume.
    """
    import json
    from types import SimpleNamespace

    import torch
    import yaml

    from egbinn.adapters import EGBINNAdapter
    from egbinn.config import config_from_dict
    from egbinn.models import EGBINN
    from egbinn.training import train_model

    rng = np.random.default_rng(3)
    n_cpgs, n_genes, n_l2 = 40, 12, 5
    k = 2

    def _unique_edges(n_in, n_out, n_edges):
        pairs = set()
        while len(pairs) < n_edges:
            pairs.add((int(rng.integers(0, n_in)), int(rng.integers(0, n_out))))
        return np.array(sorted(pairs), dtype=np.int64).T

    mask_root = tmp_path / "masks"
    stage_dims = [("cpg_to_gene", n_cpgs, n_genes, 40), ("gene_to_l2", n_genes, n_l2, 30)]
    stage_dicts = []
    for name, n_in, n_out, n_edges in stage_dims:
        stage_dir = mask_root / name
        stage_dir.mkdir(parents=True)
        mask = _unique_edges(n_in, n_out, n_edges)
        np.save(stage_dir / "mask_indices.npy", mask)
        with open(stage_dir / "shapes.json", "w") as f:
            json.dump({"out_features": n_out}, f)
        if name == "cpg_to_gene":
            names_df = pd.DataFrame({"gene": [f"GENE{i}" for i in range(n_out)], "gene_idx": list(range(n_out))})
        else:
            names_df = pd.DataFrame(
                {"pathway_id": [f"R-HSA-{i}" for i in range(n_out)], "pathway_id_idx": list(range(n_out))}
            )
        names_df.to_csv(stage_dir / "output_index.csv", index=False)
        stage_dicts.append(
            {
                "name": name,
                "layer_type": "gated",
                "gate_activation": "silu",
                "match_dense_init": True,
                "pos_weights": False,
                "value_bias": True,
                "gate_bias": True,
                "gate_weight_init": "scaled",
                "gate_bias_init": 1.2785,
                "ensemblify": True,
                "mask_indices": torch.as_tensor(mask, dtype=torch.long),
                "out_features": n_out,
            }
        )

    model = EGBINN(
        input_dim=n_cpgs,
        stages=stage_dicts,
        k=k,
        d_block_mult=1.0,
        n_dense_blocks=0,
        dropout=0.0,
        input_dropout=None,
        share_training_batches=True,
        stage_activation="none",
        regression_approach="multi_stage",
    )

    loader, scaler = make_dataloader(n_samples=32, n_features=n_cpgs, batch_size=16)
    config = config_from_dict(
        {
            "model": "EGBINN",
            "general": {
                "lr": 0.01,
                "weight_decay": 0.0,
                "warmup_epochs": 0,
                "train_epochs": 2,
                "patience": 0,
                "huber_delta": 1.0,
                "label_column": "age",
            },
            "model_params": {
                "k": k,
                "n_dense_blocks": 0,
                "stage_activation": "none",
                "regression_approach": "multi_stage",
                "stages": stage_dicts,
            },
            "output": {"save_dir": str(tmp_path / "train_out"), "checkpoint_name": "bundle.pth"},
        }
    )
    train_model(
        model=model,
        adapter=EGBINNAdapter(),
        train_loader=loader,
        val_loader=None,
        config=config,
        device=torch.device("cpu"),
        label_scaler=scaler,
        input_dim=n_cpgs,
    )

    yaml_config = {
        "model": "EGBINN",
        "model_params": {
            "k": k,
            "n_dense_blocks": 0,
            "stage_activation": "none",
            "regression_approach": "multi_stage",
            "stages": [
                {
                    **{kk: vv for kk, vv in s.items() if kk not in ("mask_indices", "out_features")},
                    "mask_path": str(mask_root / s["name"] / "mask_indices.npy"),
                }
                for s in stage_dicts
            ],
        },
        "data": {"mask_dir": str(mask_root)},
        "output": {},
    }
    config_path = tmp_path / "egbinn_config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_config, f, sort_keys=False)

    background = rng.uniform(0, 1, (8, n_cpgs)).astype(np.float32)
    test = rng.uniform(0, 1, (4, n_cpgs)).astype(np.float32)

    return SimpleNamespace(
        model=model,
        bundle_path=str(tmp_path / "train_out" / "bundle.pth"),
        config_path=str(config_path),
        mask_dir=str(mask_root),
        background=background,
        test=test,
        n_cpgs=n_cpgs,
        n_genes=n_genes,
        n_l2=n_l2,
        k=k,
    )
