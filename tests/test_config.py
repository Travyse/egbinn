"""Tests for config loading, conversion, and validation."""

import pytest

from egbinn.config import AttrDict, config_from_dict, load_config, to_plain_dict, validate_config


def test_attrdict_attribute_access():
    d = AttrDict({"a": 1})
    assert d.a == 1
    d.x = 5
    assert d["x"] == 5


def test_nested_conversion_via_config_from_dict():
    config = config_from_dict({"a": 1, "b": {"c": 2}})
    assert config.a == 1
    assert config.b.c == 2


def test_attrdict_missing_attribute_raises():
    d = AttrDict({"a": 1})
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = d.missing


def test_load_config_roundtrip(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("model: MLP\ngeneral:\n  lr: 0.1\n  nested:\n    - a\n    - b\n")
    config = load_config(str(cfg_path))
    assert config.model == "MLP"
    assert config.general.lr == 0.1
    assert config.general.nested == ["a", "b"]


def test_load_config_empty_file(tmp_path):
    cfg_path = tmp_path / "empty.yml"
    cfg_path.write_text("")
    config = load_config(str(cfg_path))
    assert isinstance(config, AttrDict)
    assert len(config) == 0


def test_config_from_dict_nested():
    config = config_from_dict({"model": "MLP", "model_params": {"stages": [{"k": 1}]}})
    assert config.model == "MLP"
    assert config.model_params.stages[0].k == 1


def test_to_plain_dict_is_json_safe():
    import json

    config = config_from_dict({"model": "MLP", "general": {"lr": 0.1, "flag": True, "none": None}})
    plain = to_plain_dict(config)
    assert not isinstance(plain, AttrDict)
    json.dumps(plain)
    assert plain["general"]["lr"] == 0.1


def test_to_plain_dict_str_fallback_for_exotic_values():
    plain = to_plain_dict({"ts": object()})
    assert isinstance(plain["ts"], str)


def test_validate_config_ok(make_config):
    config = make_config(
        data={"metadata_path": "/x.csv", "train_methylation_path": "/x.memmap"}, output={"save_dir": "/tmp/out"}
    )
    assert validate_config(config)


def test_validate_config_missing_section(make_config):
    config = make_config()
    del config["output"]
    with pytest.raises(ValueError, match="missing required sections.*output"):
        validate_config(config)


def test_validate_config_unknown_model(make_config):
    config = make_config(
        model="NotAModel",
        data={"metadata_path": "/x.csv", "train_methylation_path": "/x.memmap"},
        output={"save_dir": "/tmp/out"},
    )
    with pytest.raises(ValueError, match="unknown model"):
        validate_config(config)


def test_validate_config_missing_train_path(make_config):
    config = make_config(output={"save_dir": "/tmp/out"})
    with pytest.raises(ValueError, match="data.metadata_path"):
        validate_config(config)


def test_validate_config_no_train_paths_when_construct_only(make_config):
    config = make_config(output={"save_dir": "/tmp/out"})
    assert validate_config(config, require_train_paths=False)
