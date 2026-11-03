"""Configuration loader for the Nool application."""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path: Project root directory.
    """
    return Path(__file__).resolve().parents[3]


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load application configuration from a YAML file.

    Args:
        config_path: Optional path to the configuration file.
            Defaults to configs/app.yaml relative to project root.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty or invalid.
    """
    if config_path is None:
        target_path = get_project_root() / "configs" / "app.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = get_project_root() / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file {target_path} must define a mapping/dictionary."
        )

    return config


def resolve_path(relative_or_absolute_path: Path | str) -> Path:
    """Resolve a path relative to the project root if not already absolute.

    Args:
        relative_or_absolute_path: Path string or Path object.

    Returns:
        Path: Absolute resolved Path.
    """
    p = Path(relative_or_absolute_path)
    if p.is_absolute():
        return p
    return (get_project_root() / p).resolve()
