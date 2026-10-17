__all__ = ["AttrDict", "config_from_dict", "load_config", "to_plain_dict", "validate_config"]

import yaml


class AttrDict(dict):
    """Dict subclass with attribute access (replaces easydict.EasyDict)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'") from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'") from None


def _convert(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _convert(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def _to_plain(obj):
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def load_config(path):
    """Load a YAML config file into a nested AttrDict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _convert(raw) if raw else AttrDict()


def config_from_dict(d):
    """Convert a plain nested dict (e.g. a checkpoint's embedded resolved config) into an AttrDict."""
    return _convert(d) if d else AttrDict()


def to_plain_dict(obj):
    """Recursively convert an AttrDict / nested structure into plain, JSON-safe types."""
    return _to_plain(obj)


_REQUIRED_SECTIONS = ("model", "general", "model_params", "data", "output")
_VALID_MODELS = ("MLP", "BINN", "EGBINN")


def validate_config(config, require_train_paths=True):
    """Validate a training config, raising a friendly error on the first problem.

    Parameters
    ----------
    config : AttrDict
        Loaded YAML config.
    require_train_paths : bool, default True
        Whether to require ``data.metadata_path`` and
        ``data.train_methylation_path`` (training / validation runs). Set to
        False for configs that only drive model construction.

    Raises
    ------
    ValueError
        With a message naming the missing/invalid key.
    """
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}. Found: {list(config.keys())}.")
    if config.model not in _VALID_MODELS:
        raise ValueError(f"Config has unknown model {config.model!r}. Expected one of {list(_VALID_MODELS)}.")
    if require_train_paths:
        has_train_meta = config.data.get("train_metadata_path") or config.data.get("metadata_path")
        if not has_train_meta:
            raise ValueError(
                "Config data section is missing required key 'data.metadata_path' (or 'data.train_metadata_path')."
            )
        if not config.data.get("train_methylation_path"):
            raise ValueError("Config data section is missing required key 'data.train_methylation_path'.")
    if not config.output.get("save_dir"):
        raise ValueError("Config output section is missing required key 'output.save_dir'.")
    return True
