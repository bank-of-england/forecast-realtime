"""End-to-end check that the package drives a Julia model via ``JuliaModel``.

Mirrors ``test_forecast_r.py``: a plain OLS script (``beta = (XᵀX)⁻¹ Xᵀy``,
no Julia stats package) is fitted and forecast through the bundled
``runner.jl``. Skips unless ``julia`` is on ``PATH`` with the runner's
``Parquet2`` / ``DataFrames`` packages installed.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.external_model import JuliaModel

_OLS_JL_SCRIPT = str(Path(__file__).parent / "jl_scripts" / "ols_model.jl")


def _julia_runner_deps_available() -> bool:
    """Return ``True`` when ``julia`` and the ``runner.jl`` packages are present."""
    if shutil.which("julia") is None:
        return False
    probe = 'using Parquet2, DataFrames; print("OK")'
    try:
        result = subprocess.run(
            ["julia", "--startup-file=no", "-e", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception:
        return False
    return "OK" in result.stdout


pytestmark = pytest.mark.skipif(
    not _julia_runner_deps_available(),
    reason="julia not on PATH or Parquet2/DataFrames not installed",
)


@pytest.fixture
def regression_data():
    """24 quarters of ``y`` generated from two regressors, plus the regressors."""
    dates = pd.date_range("2000-03-31", periods=24, freq="QE")
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=24)
    x2 = rng.normal(size=24)
    y = pd.DataFrame(
        {"gdp": 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.01, size=24)},
        index=dates,
    )
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=dates)
    return y, X


def _future_X(X: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Return ``steps`` rows of fresh regressor values on the following quarters."""
    future_dates = pd.date_range(
        X.index[-1] + pd.offsets.QuarterEnd(), periods=steps, freq="QE"
    )
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {c: rng.normal(size=steps) for c in X.columns}, index=future_dates
    )


def test_forecast_julia_ols(regression_data):
    """Fit OLS, forecast 4 steps, and verify output shape."""
    y, X = regression_data
    model = JuliaModel(_OLS_JL_SCRIPT)

    model.fit(y, X=X)
    fc = model.forecast(steps=4, X=_future_X(X, 4))

    assert isinstance(fc, pd.DataFrame)
    assert fc.shape == (4, 1)
    assert list(fc.columns) == ["gdp"]
    assert not fc.isna().any().any()


def test_forecast_julia_ols_uses_future_regressor_rows(regression_data):
    """Forecast-time X changes must affect the Julia model forecast."""
    y, X = regression_data
    X_all = pd.concat([X, _future_X(X, 4)])

    model = JuliaModel(_OLS_JL_SCRIPT)
    model.fit(y, X=X)
    baseline = model.forecast(steps=4, X=X_all).to_numpy()

    changed_X = X_all.copy()
    changed_X.iloc[-4:, :] += 100.0
    changed = model.forecast(steps=4, X=changed_X).to_numpy()

    assert not np.allclose(baseline, changed)


def test_matches_numpy_ols(regression_data):
    """Julia normal-equation OLS matches the same solve done in numpy."""
    y, X = regression_data
    Xm = X.to_numpy()
    yv = y["gdp"].to_numpy()
    beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ yv)  # (XᵀX)⁻¹ Xᵀy

    steps = 5
    X_future = _future_X(X, steps)
    expected = X_future.to_numpy() @ beta

    model = JuliaModel(_OLS_JL_SCRIPT)
    model.fit(y, X=X)
    fc = model.forecast(steps=steps, X=X_future)

    np.testing.assert_allclose(fc["gdp"].to_numpy(), expected, rtol=1e-6)
