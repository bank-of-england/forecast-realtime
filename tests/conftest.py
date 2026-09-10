import importlib
import pickle
import shutil
import subprocess
from concurrent.futures import Future
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from forecast_evaluation import ForecastData

from tests.realtime_fixtures import generate_synthetic_data

# Load sample_data module from tests/sample_data.py
_spec = importlib.util.spec_from_file_location(
    "sample_data",
    Path(__file__).parent / "sample_data.py",
)
_sample_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sample_data)


@pytest.fixture
def inline_executor(monkeypatch):
    """Test task dispatch without starting processes; integration tests use real pools."""

    class InlineExecutor(AbstractContextManager):
        def __init__(self, max_workers=None):
            pass

        def submit(self, function, task):
            future = Future()
            try:
                result = function(pickle.loads(pickle.dumps(task)))
                future.set_result(pickle.loads(pickle.dumps(result)))
            except Exception as error:
                future.set_exception(error)
            return future

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "forecast_realtime.real_time_model.ProcessPoolExecutor", InlineExecutor
    )


@pytest.fixture(scope="session")
def r_arrow_available():
    """Skip tests when Rscript is not on PATH or R's arrow package is unavailable."""
    reason = "Rscript not found on PATH or R 'arrow' package not installed"
    if shutil.which("Rscript") is None:
        pytest.skip(reason)
    try:
        result = subprocess.run(
            ["Rscript", "-e", "cat(requireNamespace('arrow', quietly=TRUE))"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pytest.skip(reason)
    if "TRUE" not in result.stdout:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def xgboost_available():
    """Skip tests when XGBoost or its native runtime cannot be loaded."""
    try:
        importlib.import_module("xgboost")
    except ModuleNotFoundError:
        pytest.skip("xgboost is not installed")
    except Exception as error:
        if (
            error.__class__.__name__ != "XGBoostError"
            or not error.__class__.__module__.startswith("xgboost")
        ):
            raise
        pytest.skip(f"xgboost native library could not be loaded: {error}")


@pytest.fixture
def bvar_python_kernel(monkeypatch):
    """Run the real small-array kernel without JIT; native parity tests retain JIT."""
    from bvar.forecast import matrices

    kernel = getattr(matrices, "_construct_b_B_M_N_K", None)
    if hasattr(kernel, "py_func"):
        monkeypatch.setattr(matrices, "_construct_b_B_M_N_K", kernel.py_func)


@pytest.fixture(scope="session")
def sample_outturns():
    """Sample outturns for all six variables."""
    return _sample_data.create_sample_mixed_freq_outturns()


@pytest.fixture(scope="session")
def quarterly_outturns():
    """Quarterly outturns with different publication lags."""
    return _sample_data.create_sample_quarterly_outturns()


@pytest.fixture(scope="session")
def monthly_outturns():
    """Monthly outturns with different publication lags."""
    return _sample_data.create_sample_monthly_outturns()


@pytest.fixture(scope="session")
def mixed_frequency_outturns():
    """Quarterly target and monthly regressors with different lags."""
    return _sample_data.create_sample_mixed_frequency_outturns()


@pytest.fixture
def forecast_data(sample_outturns):
    """ForecastData object built from sample outturns.

    Function-scoped and copies the outturns because ForecastData is mutable
    (tests call add_forecasts) and must not alias the session-scoped data.
    """
    return ForecastData(
        outturns_data=sample_outturns.copy(),
        metric="levels",
        compute_levels=False,
    )


@pytest.fixture(scope="session")
def sample_realtime_complete():
    """Compact synthetic panel with every observation available immediately."""
    return generate_synthetic_data(
        N=2,
        seed=20260101,
        first_period="2015-01-31",
        endpoint="2024-12-31",
        publication_lags=False,
    )


@pytest.fixture(scope="session")
def sample_realtime_ragged():
    """Compact synthetic panel with deterministic publication lags."""
    return generate_synthetic_data(
        N=2,
        seed=20260101,
        first_period="2015-01-31",
        endpoint="2024-12-31",
        publication_lags=True,
    )
