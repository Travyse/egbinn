"""Flat backing for CSV methylation data."""

import os

import pandas as pd


def _load_csv_array(path):
    """Load a CSV methylation array as a float32 ndarray.

    Expects samples as rows and CpG sites as columns.

    Returns
    -------
    tuple : (np.ndarray, n_features)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    array = df.values.astype("float32")
    return array, array.shape[1]
