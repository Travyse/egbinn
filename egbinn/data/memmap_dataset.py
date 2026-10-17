"""Memmap-backed methylation data loading."""

import os

import numpy as np


def _load_memmap_array(path, n_features=None):
    """Open a float32 memmap methylation array for reading.

    Parameters
    ----------
    path : str
        Path to the ``.memmap`` file.
    n_features : int or None
        Number of features (columns). If None, inferred from the
        ``_avg_beta_values.npy`` sidecar next to the memmap.

    Returns
    -------
    tuple : (np.memmap, n_features)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Memmap file not found: {path}")
    if n_features is None:
        sidecar = str(path).replace(".memmap", "_avg_beta_values.npy")
        if os.path.exists(sidecar):
            n_features = int(np.load(sidecar).shape[0])
        else:
            raise ValueError(
                f"Cannot infer the number of features for {path}: provide n_features or a sidecar file at {sidecar}."
            )

    n_samples = _infer_sample_count(path, n_features)
    memmap = np.memmap(path, dtype=np.float32, mode="r", shape=(n_samples, n_features))
    return memmap, n_features


def _infer_sample_count(path, n_features):
    """Infer the number of samples from the file size."""
    file_size = os.path.getsize(path)
    bytes_per_sample = n_features * 4
    if file_size % bytes_per_sample != 0:
        raise ValueError(
            f"File size {file_size} is not divisible by n_features*4 = "
            f"{bytes_per_sample}. Check the inferred feature count."
        )
    return file_size // bytes_per_sample
