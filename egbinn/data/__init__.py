__all__ = ["MethylationDataset", "load_data", "infer_input_dim", "load_betas_array"]

import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils import data

logger = logging.getLogger(__name__)


class MethylationDataset(data.Dataset):
    """DNA methylation dataset for aging clocks.

    Yields batches with ``sample`` (sample ID), ``betas`` (shape
    ``[num_features]``), ``label`` (scaled age), and ``label_raw`` (raw age).

    Parameters
    ----------
    methylation_array : np.ndarray or np.memmap
        Beta values, shape ``(n_samples, n_features)``.
    metadata : pd.DataFrame
        Must contain ``label_column``; rows aligned with the methylation
        array. The DataFrame index provides the sample IDs.
    label_column : str
        Column name for the age label.
    label_scaler : StandardScaler or None
        If None, a scaler is fit on this metadata (training split).
        Otherwise the provided (training) scaler is used to transform —
        required for validation/test splits so predictions map back to the
        training age scale.
    imputation_values : np.ndarray or None
        Per-feature mean beta values for NaN imputation, shape
        ``(n_features,)``.

    Raises
    ------
    ValueError
        If the metadata is empty, the label column is missing, the
        methylation array and metadata row counts disagree, or the
        imputation vector length does not match the feature count.
    """

    def __init__(
        self,
        methylation_array,
        metadata,
        label_column="age",
        label_scaler=None,
        imputation_values=None,
    ):
        super().__init__()
        if len(metadata) == 0:
            raise ValueError("Metadata is empty: no samples to load.")
        if label_column not in metadata.columns:
            raise ValueError(
                f"Metadata is missing the label column {label_column!r}. Available columns: {list(metadata.columns)}."
            )
        if methylation_array.shape[0] != len(metadata):
            raise ValueError(
                f"Methylation array has {methylation_array.shape[0]} rows but metadata has "
                f"{len(metadata)} rows. The data file and metadata must describe the same samples."
            )
        n_features = int(methylation_array.shape[1])
        if imputation_values is not None and imputation_values.shape[0] != n_features:
            raise ValueError(
                f"Imputation values have length {imputation_values.shape[0]} but the "
                f"methylation array has {n_features} features."
            )

        n_nan_labels = int(pd.isna(metadata[label_column]).sum())
        if n_nan_labels:
            logger.warning(
                "%d/%d samples have NaN in the label column %r — their predictions "
                "will be returned but excluded from metric computation.",
                n_nan_labels,
                len(metadata),
                label_column,
            )

        self.methylation_array = methylation_array
        self.label_column = label_column
        self.imputation_values = imputation_values.astype(np.float32) if imputation_values is not None else None
        self.sample_ids = metadata.index.astype(str).to_numpy()

        metadata = metadata.reset_index(drop=True)
        if label_scaler is None:
            label_scaler = StandardScaler()
            metadata[f"{label_column}_scaled"] = label_scaler.fit_transform(metadata[[label_column]])
        else:
            metadata[f"{label_column}_scaled"] = label_scaler.transform(metadata[[label_column]])
        self.label_scaler = label_scaler

        self._label_scaled = metadata[f"{label_column}_scaled"].to_numpy(dtype=np.float64)
        self._label_raw = metadata[label_column].to_numpy(dtype=np.float64)

    def _impute_nans(self, x):
        if self.imputation_values is None or not np.any(np.isnan(x)):
            return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x
        x = np.array(x, dtype=np.float32, copy=True)
        nan_mask = np.isnan(x)
        if x.ndim == 1:
            x[nan_mask] = self.imputation_values[nan_mask]
        elif x.ndim == 2:
            x[nan_mask] = np.broadcast_to(self.imputation_values, x.shape)[nan_mask]
        else:
            raise ValueError(f"Expected a 1D or 2D array, got shape {x.shape}.")
        return x

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        if isinstance(index, torch.Tensor):
            index = int(index)

        x = np.asarray(self.methylation_array[index], dtype=np.float32)
        x = self._impute_nans(x)
        return {
            "sample": self.sample_ids[index],
            "betas": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(self._label_scaled[index], dtype=torch.float32),
            "label_raw": torch.tensor(self._label_raw[index], dtype=torch.float32),
        }

    def __getitems__(self, idxs):
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()
        elif isinstance(idxs, np.ndarray):
            idxs = idxs.tolist()

        x = np.asarray(self.methylation_array[idxs], dtype=np.float32)
        x = self._impute_nans(x)
        return [
            {
                "sample": self.sample_ids[idxs[i]],
                "betas": torch.from_numpy(np.ascontiguousarray(x[i])),
                "label": torch.tensor(self._label_scaled[idxs[i]], dtype=torch.float32),
                "label_raw": torch.tensor(self._label_raw[idxs[i]], dtype=torch.float32),
            }
            for i in range(len(idxs))
        ]


def load_data(
    methylation_path,
    metadata_path,
    label_column="age",
    label_scaler=None,
    batch_size=64,
    num_workers=0,
    shuffle=False,
    imputation_path=None,
    is_validation=False,
    data_format=None,
):
    """Load methylation data + metadata into a DataLoader.

    Dispatches to the memmap or CSV backend based on ``data_format`` or the
    file extension. The returned scaler is the one used to scale labels:
    freshly fit when ``label_scaler=None`` (training split), or the scaler
    passed in (validation/test splits — pass the *training* scaler so
    predictions land on the training age scale).

    Returns
    -------
    tuple : (DataLoader, StandardScaler)
    """
    if data_format is None:
        data_format = "memmap" if str(methylation_path).endswith(".memmap") else "csv"

    imputation_values = None
    if imputation_path:
        imputation_values = np.load(imputation_path).astype(np.float32)

    n_features = imputation_values.shape[0] if imputation_values is not None else None
    if data_format == "memmap":
        from .memmap_dataset import _load_memmap_array

        methylation_array, n_features = _load_memmap_array(methylation_path, n_features)
    else:
        from .csv_dataset import _load_csv_array

        methylation_array, n_features = _load_csv_array(methylation_path)

    metadata = pd.read_csv(metadata_path, index_col=0)

    dataset = MethylationDataset(
        methylation_array=methylation_array,
        metadata=metadata,
        label_column=label_column,
        label_scaler=label_scaler,
        imputation_values=imputation_values,
    )

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(num_workers, 1) if is_validation else num_workers,
        persistent_workers=(num_workers > 0) and not is_validation,
        pin_memory=torch.cuda.is_available(),
        drop_last=not is_validation,
        shuffle=shuffle,
    )

    return loader, dataset.label_scaler


def infer_input_dim(methylation_path, imputation_path=None, data_format=None):
    """Infer the number of CpG features in a methylation data file.

    Resolution order: the imputation sidecar (when given), the CSV header,
    or the memmap's ``_avg_beta_values.npy`` sidecar.

    Raises
    ------
    ValueError
        If the feature count cannot be determined.
    """
    if imputation_path:
        return int(np.load(imputation_path).shape[0])
    fmt = data_format or ("memmap" if str(methylation_path).endswith(".memmap") else "csv")
    if fmt == "csv":
        df = pd.read_csv(methylation_path, index_col=0, nrows=0)
        return int(df.shape[1])
    sidecar = str(methylation_path).replace(".memmap", "_avg_beta_values.npy")
    if os.path.exists(sidecar):
        return int(np.load(sidecar).shape[0])
    raise ValueError(
        f"Cannot infer the number of features for {methylation_path}: provide "
        f"imputation_path or a sidecar file at {sidecar}."
    )


def load_betas_array(methylation_path, imputation_path=None, data_format=None):
    """Load methylation betas as a plain ``(n_samples, n_features)`` float32 array.

    NaNs are imputed with per-feature means when ``imputation_path`` is
    given. Unlike :func:`load_data`, no DataLoader is constructed — intended
    for consumers that need raw arrays (e.g. SHAP background/test sets).
    """
    fmt = data_format or ("memmap" if str(methylation_path).endswith(".memmap") else "csv")
    if fmt == "csv":
        from .csv_dataset import _load_csv_array

        arr, _ = _load_csv_array(methylation_path)
    else:
        n_features = infer_input_dim(methylation_path, imputation_path, fmt)
        from .memmap_dataset import _load_memmap_array

        mm, _ = _load_memmap_array(methylation_path, n_features)
        arr = np.array(mm)

    if imputation_path is not None:
        avg = np.load(imputation_path).astype(np.float32)
        if avg.shape[0] != arr.shape[1]:
            raise ValueError(
                f"Imputation values have length {avg.shape[0]} but the methylation array has {arr.shape[1]} features."
            )
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = np.broadcast_to(avg, arr.shape)[nan_mask]
    return arr.astype(np.float32, copy=False)
