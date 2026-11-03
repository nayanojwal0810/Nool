"""Unit tests verifying project directory and file structure."""

from nool.config import get_project_root


def test_expected_project_directories_exist() -> None:
    """Verify that all foundational project directories exist."""
    root = get_project_root()
    expected_directories = [
        root / ".github" / "workflows",
        root / "configs",
        root / "data" / "raw",
        root / "data" / "interim",
        root / "data" / "processed",
        root / "data" / "manifests",
        root / "docs",
        root / "notebooks",
        root / "scripts",
        root / "src" / "nool",
        root / "tests" / "unit",
        root / "tests" / "integration",
        root / "tests" / "fixtures",
    ]
    for directory in expected_directories:
        assert directory.exists(), f"Missing directory: {directory}"
        assert directory.is_dir(), f"Not a directory: {directory}"


def test_expected_core_files_exist() -> None:
    """Verify that all foundational project files exist."""
    root = get_project_root()
    expected_files = [
        root / ".gitignore",
        root / ".gitattributes",
        root / ".python-version",
        root / "pyproject.toml",
        root / "README.md",
        root / "configs" / "app.yaml",
        root / "docs" / "system_overview.md",
        root / ".github" / "workflows" / "quality.yml",
        root / "data" / "manifests" / "hotpotqa.yaml",
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"Missing file: {file_path}"
        assert file_path.is_file(), f"Not a file: {file_path}"
