"""Tests for the public model exports."""

import subprocess
import sys
from types import ModuleType

import pytest

import forecast_realtime.models as models


def test_public_exports_use_forecast_class_names():
    """Lasso and ElasticNet exports should name their forecast classes."""
    assert "ForecastLasso" in models.__all__
    assert "ForecastElasticNet" in models.__all__
    assert "Lasso" not in models.__all__
    assert "ElasticNet" not in models.__all__
    assert models.ForecastLasso.__name__ == "ForecastLasso"
    assert models.ForecastElasticNet.__name__ == "ForecastElasticNet"


def test_all_public_exports_are_available_from_star_import(xgboost_available):
    """Every name in __all__ should be defined and importable."""
    namespace = {}
    exec("from forecast_realtime.models import *", namespace)

    assert set(models.__all__) <= namespace.keys()
    assert namespace["ForecastOLS"] is models.ForecastOLS


def test_fable_exports_resolve_via_lazy_public_api():
    """Fable wrappers should be available through the public lazy exports."""
    assert models.RFableModel.__name__ == "RFableModel"
    assert models.RFableETS.__name__ == "RFableETS"
    assert models.RFableARIMA.__name__ == "RFableARIMA"


def test_xgboost_public_export_is_available(xgboost_available):
    """XGBoost remains available as a public class."""

    assert models.XGBoost.__name__ == "XGBoost"


def test_direct_submodule_import_does_not_shadow_public_export():
    """Directly importing a same-named submodule keeps the public class."""

    script = (
        "import forecast_realtime.models as models\n"
        "from forecast_realtime.models.forecast_bvar import ForecastBVAR\n"
        "assert models.ForecastBVAR is ForecastBVAR\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_missing_declared_dependency_has_installation_guidance(monkeypatch):
    """A missing optional dependency names the model, package, and extra."""

    def missing_dependency(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'bvar'", name="bvar")

    monkeypatch.setattr(models, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="ForecastBVAR.*bvar.*bvar"):
        models._optional_model("ForecastBVAR")


def test_unrelated_import_failure_is_preserved(monkeypatch):
    """Import errors from inside a model module are not mislabelled."""
    error = ModuleNotFoundError("No module named 'transitive'", name="transitive")

    def unrelated_failure(*args, **kwargs):
        raise error

    monkeypatch.setattr(models, "import_module", unrelated_failure)

    with pytest.raises(ModuleNotFoundError) as raised:
        models._optional_model("ForecastBVAR")

    assert raised.value is error


def test_failed_optional_lookup_is_not_cached(monkeypatch):
    """A later successful lookup remains possible after a failed import."""
    calls = []

    def lookup(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ModuleNotFoundError("No module named 'bvar'", name="bvar")
        module = ModuleType("ForecastBVAR")
        module.ForecastBVAR = type("ForecastBVAR", (), {})
        return module

    monkeypatch.setattr(models, "import_module", lookup)
    models.__dict__.pop("ForecastBVAR", None)

    with pytest.raises(ModuleNotFoundError):
        models.__getattr__("ForecastBVAR")

    resolved = models.__getattr__("ForecastBVAR")

    assert resolved.__name__ == "ForecastBVAR"
    assert models.__dict__["ForecastBVAR"] is resolved
    assert len(calls) == 2
