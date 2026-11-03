"""Unit tests verifying package and module imports."""

import importlib

import nool


def test_package_import_and_version() -> None:
    """Verify that the root package imports and has a valid version."""
    assert nool.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    """Verify that all core subpackages can be imported."""
    subpackages = [
        "nool.config",
        "nool.data",
        "nool.retrieval",
        "nool.evidence",
        "nool.security",
        "nool.generation",
        "nool.evaluation",
        "nool.app",
    ]
    for subpkg in subpackages:
        mod = importlib.import_module(subpkg)
        assert mod is not None
