import importlib.util
from pathlib import Path

import pytest
from forecast_evaluation import ForecastData

import forecast_realtime as rt

# Load sample_data module from tests/sample_data.py
_spec = importlib.util.spec_from_file_location(
    "sample_data",
    Path(__file__).parent / "sample_data.py",
)
_sample_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sample_data)


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
    return rt.generate_synthetic_data(
        N=2,
        seed=20260101,
        first_period="2015-01-31",
        endpoint="2024-12-31",
        publication_lags=False,
    )


@pytest.fixture(scope="session")
def sample_realtime_ragged():
    """Compact synthetic panel with deterministic publication lags."""
    return rt.generate_synthetic_data(
        N=2,
        seed=20260101,
        first_period="2015-01-31",
        endpoint="2024-12-31",
        publication_lags=True,
    )
