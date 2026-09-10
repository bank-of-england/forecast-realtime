"""Tests for the package-level public exports."""

import importlib.metadata

import forecast_realtime as rt


def test_package_exposes_distribution_version():
    """The package version should match its installed distribution metadata."""
    assert rt.__version__ == importlib.metadata.version("forecast_realtime")
