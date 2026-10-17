"""Tests for device resolution, logging, and git provenance helpers."""

import logging

import pytest
import torch

from egbinn.utils import attach_stream_logging, get_git_commit, resolve_device, setup_logger


def test_resolve_device_auto_returns_valid_device():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_none_means_auto():
    assert resolve_device(None) == resolve_device("auto")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this machine")
def test_resolve_device_explicit_cuda_unavailable_raises():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_resolve_device_invalid_spec_raises():
    with pytest.raises(ValueError, match="Unknown device spec"):
        resolve_device("tpu")


def test_get_git_commit_returns_str_or_none():
    commit = get_git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) >= 7)


def test_get_git_commit_nonexistent_start_dir():
    assert get_git_commit("/definitely/not/a/real/path") is None


def test_setup_logger_writes_file(tmp_path):
    logger = setup_logger(str(tmp_path), "egbinn_test_setup_logger", "train.log")
    assert len(logger.handlers) == 2
    logger.info("hello from the test")
    content = (tmp_path / "train.log").read_text()
    assert "hello from the test" in content


def test_setup_logger_removes_stale_file_and_duplicate_handlers(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text("stale content")
    logger_name = "egbinn_test_stale_logger"

    setup_logger(str(tmp_path), logger_name, "train.log")
    logger = setup_logger(str(tmp_path), logger_name, "train.log")

    assert "stale content" not in log_path.read_text()
    assert len(logger.handlers) == 2
    logger.info("fresh record")
    assert "fresh record" in log_path.read_text()


def test_attach_stream_logging_idempotent():
    pkg = logging.getLogger("egbinn")
    pkg.handlers.clear()
    try:
        attach_stream_logging()
        first = _count_stream_handlers(pkg)
        assert first == 1
        attach_stream_logging()
        assert _count_stream_handlers(pkg) == 1
    finally:
        pkg.handlers.clear()


def _count_stream_handlers(logger):
    return sum(
        1 for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
