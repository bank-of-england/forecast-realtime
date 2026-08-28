"""Minimal custom-model example for the ForecastModel interface."""

import numpy as np
import pandas as pd
import pytest

from forecast_realtime import ForecastModel


class MyOLS(ForecastModel):
    """
    Small OLS model showing the intended custom-model authoring pattern.
    """

    def _fit(self, y, X):
        # y and X are passed as pandas DataFrames
        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)

        # OLS estimate: beta = (X'X)^-1 X'y
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def _forecast(self, steps, y, X, **kwargs):
        X_future = X.loc[X.index > self.last_y_fit_date].iloc[:steps]
        return X_future.to_numpy(dtype=float) @ self.beta

    def _forecast_decomp(self, steps, y, X, **kwargs):
        """Return the required long-form additive decomposition."""
        if X is None:
            return None

        X_future = X.loc[X.index > self.last_y_fit_date].iloc[:steps]
        contributions = X_future.to_numpy(dtype=float)[:, :, None] * self.beta[None, :, :]

        rows = []
        for horizon in range(steps):
            for feature, contribution in zip(
                X_future.columns, contributions[horizon, :, 0]
            ):
                rows.append(
                    {
                        "forecast_horizon": horizon,
                        "component": feature,
                        "contribution": float(contribution),
                        "weight": float(self.beta[X_future.columns.get_loc(feature), 0]),
                    }
                )

        return pd.DataFrame(rows)


def test_custom_ols_uses_numpy_internally_and_array_output_is_wrapped():
    dates = pd.date_range("2020-03-31", periods=8, freq="QE")
    X = pd.DataFrame(
        {
            "x1": np.arange(1.0, 9.0),
            "x2": np.arange(8.0, 0.0, -1.0),
        },
        index=dates,
    )
    beta = np.array([[2.0], [-0.5]])
    y = pd.DataFrame({"target": X.to_numpy() @ beta[:, 0]}, index=dates)

    model = MyOLS()
    assert model.fit(y, X) is model
    np.testing.assert_allclose(model.beta, beta)

    future_dates = pd.date_range("2022-03-31", periods=3, freq="QE")
    X_future = pd.DataFrame(
        {"x1": [9.0, 10.0, 11.0], "x2": [0.0, -1.0, -2.0]},
        index=future_dates,
    )
    forecast = model.forecast(steps=3, X=X_future, decomp=True)

    expected = X_future.to_numpy() @ beta
    expected_index = pd.DatetimeIndex(future_dates.to_numpy(), name="date")
    pd.testing.assert_frame_equal(
        forecast,
        pd.DataFrame(expected, index=expected_index, columns=["target"]),
    )

    decomposition = forecast.decomposition
    assert list(decomposition.columns) == [
        "forecast_horizon",
        "component",
        "contribution",
        "weight",
    ]
    assert len(decomposition) == 3 * 2
    totals = decomposition.groupby("forecast_horizon")["contribution"].sum()
    np.testing.assert_allclose(totals.to_numpy(), expected[:, 0])


def test_fit_requires_custom_fit_to_return_self():
    class InvalidModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            return None

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros(steps)

    dates = pd.date_range("2020-03-31", periods=4, freq="QE")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]}, index=dates)

    with pytest.raises(TypeError, match="_fit must return self"):
        InvalidModel().fit(y)
