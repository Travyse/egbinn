__all__ = ["seed_everything", "setup_logger", "resolve_device", "get_git_commit", "attach_stream_logging"]

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_path, log_name, file_name):
    os.makedirs(log_path, exist_ok=True)
    file_path = os.path.join(log_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def attach_stream_logging(level=logging.INFO):
    """Attach a stream handler to the 'egbinn' package logger so library-level
    log records (provenance, warnings) are visible in CLI runs."""
    pkg = logging.getLogger("egbinn")
    pkg.setLevel(level)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in pkg.handlers
    )
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        pkg.addHandler(handler)
    return pkg


def _mps_available():
    mps = getattr(torch.backends, "mps", None)
    return bool(mps) and torch.backends.mps.is_available()


def resolve_device(spec="auto"):
    """Resolve a device spec into a ``torch.device``.

    Parameters
    ----------
    spec : str
        One of ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        ``"auto"`` picks CUDA when available, then Apple MPS, then CPU.
        An explicit backend that is unavailable raises instead of silently
        falling back, so a mistaken ``--device cuda`` on a CPU-only machine
        is caught immediately.

    Returns
    -------
    torch.device

    Raises
    ------
    ValueError
        If ``spec`` is not one of the accepted values.
    RuntimeError
        If an explicit backend is requested but unavailable.
    """
    if spec in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    spec = str(spec).lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device 'cuda' but CUDA is not available on this machine. "
                "Use --device auto to select the best available device."
            )
        return torch.device("cuda")
    if spec == "mps":
        if not _mps_available():
            raise RuntimeError(
                "Requested device 'mps' but Apple MPS is not available on this machine. "
                "Use --device auto to select the best available device."
            )
        return torch.device("mps")
    raise ValueError(f"Unknown device spec: {spec!r}. Expected one of 'auto', 'cuda', 'mps', 'cpu'.")


def get_git_commit(start_dir=None):
    """Best-effort git commit hash of the repository containing ``start_dir``.

    Walks upward from ``start_dir`` looking for a ``.git`` directory; returns
    ``None`` when no repository (or no commits) is found.
    """
    import subprocess

    start = Path(start_dir) if start_dir else Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            try:
                out = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=candidate,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return out.stdout.strip()
            except (subprocess.CalledProcessError, OSError):
                return None
    return None
