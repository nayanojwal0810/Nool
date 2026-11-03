"""Unit tests verifying application configuration and path resolution."""

from pathlib import Path

import pytest

from nool.config import get_project_root, load_config, resolve_path


def test_config_file_exists() -> None:
    """Verify that configs/app.yaml exists in the project root."""
    root = get_project_root()
    config_file = root / "configs" / "app.yaml"
    assert config_file.exists()
    assert config_file.is_file()


def test_load_config_values() -> None:
    """Verify that the configuration can be loaded and contains expected keys."""
    config = load_config()
    assert isinstance(config, dict)

    assert "app" in config
    assert config["app"]["name"] == "Nool"
    assert config["app"]["logging_level"] == "INFO"

    assert "paths" in config
    paths = config["paths"]
    assert paths["raw_data_dir"] == "data/raw"
    assert paths["interim_data_dir"] == "data/interim"
    assert paths["processed_data_dir"] == "data/processed"
    assert paths["model_cache_dir"] == ".cache/models"
    assert paths["artifact_dir"] == "data/manifests"

    assert "models" in config
    assert config["models"]["cache_dir"] == ".cache/models"


def test_configured_data_paths_resolve() -> None:
    """Verify that configured data paths resolve to existing directories."""
    config = load_config()
    paths = config["paths"]

    raw_path = resolve_path(paths["raw_data_dir"])
    assert raw_path.is_absolute()
    assert raw_path.exists()
    assert raw_path.is_dir()

    interim_path = resolve_path(paths["interim_data_dir"])
    assert interim_path.is_absolute()
    assert interim_path.exists()
    assert interim_path.is_dir()

    processed_path = resolve_path(paths["processed_data_dir"])
    assert processed_path.is_absolute()
    assert processed_path.exists()
    assert processed_path.is_dir()

    manifests_path = resolve_path(paths["artifact_dir"])
    assert manifests_path.is_absolute()
    assert manifests_path.exists()
    assert manifests_path.is_dir()


def test_load_config_missing_file_raises() -> None:
    """Verify that loading a non-existent configuration raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent") / "missing.yaml")
