"""Generated realtime regressions for the optional BVAR model."""

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

pytest.importorskip("bvar")

import forecast_realtime as rt

_RESULT_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "metric",
    "frequency",
    "source",
    "value",
]
_KEY_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "metric",
    "frequency",
    "source",
]


def _levels_only(panel):
    return panel.loc[
        panel["variable"].isin(["monthly_1", "monthly_2"]) & panel["metric"].eq("levels")
    ].copy()


def _forecast_data(panel):
    return ForecastData(
        outturns_data=panel.copy(deep=True),
        compute_levels=False,
        data_check=False,
    )


def _native_result(realtime_model, label, metric):
    forecasts = (
        realtime_model.data.forecasts
        if metric == "levels"
        else realtime_model.native_forecasts
    )
    if forecasts is None or forecasts.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return forecasts.loc[
        forecasts["source"].eq(label) & forecasts["metric"].eq(metric)
    ].copy()


def _assert_equal(left, right):
    left = left[_RESULT_COLUMNS].sort_values(_KEY_COLUMNS).reset_index(drop=True)
    right = right[_RESULT_COLUMNS].sort_values(_KEY_COLUMNS).reset_index(drop=True)
    for frame in (left, right):
        assert np.isfinite(frame["value"].to_numpy(dtype=float)).all()
        assert not frame.duplicated(_KEY_COLUMNS).any()
    pd.testing.assert_frame_equal(
        left[_KEY_COLUMNS], right[_KEY_COLUMNS], check_dtype=False
    )
    np.testing.assert_allclose(
        left["value"].to_numpy(dtype=float),
        right["value"].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
    )


def _make_models():
    common = {
        "n_lags": 1,
        "nb_restart": 0,
        "n_samples": 50,
        "mode_only": True,
        "progressbar": False,
        "N_draws": 100,
        "N_burn": 50,
        "optim_random_state": 0,
        "sampling_random_state": 0,
        "forecast_random_state": 0,
    }
    return [
        rt.models.ForecastBVAR(
            label="bvar_levels",
            data_transformation={
                "monthly_1": "levels",
                "monthly_2": "levels",
            },
            **common,
        ),
        rt.models.ForecastBVAR(
            label="bvar_logs",
            data_transformation={
                "monthly_1": "logs",
                "monthly_2": "logs",
            },
            **common,
        ),
    ]


def _run_bvar(panel):
    realtime = rt.RealTimeModel(_forecast_data(panel), _make_models())
    realtime.forecast(
        y_variables=["monthly_1", "monthly_2"],
        data_transformation={"monthly_1": "levels", "monthly_2": "levels"},
        steps=2,
        first_vintage="2024-11-30",
        last_vintage="2024-11-30",
        reconstruct_levels=False,
    )
    return realtime


def test_monthly_bivariate_bvar_levels_and_logs_preserve_metric_contracts(
    sample_realtime_complete,
):
    full = _run_bvar(sample_realtime_complete)
    levels = _run_bvar(_levels_only(sample_realtime_complete))

    for label, metric in [("bvar_levels", "levels"), ("bvar_logs", "logs")]:
        full_result = _native_result(full, label, metric)
        levels_result = _native_result(levels, label, metric)
        assert len(full_result) == len(levels_result) == 4
        assert list(dict.fromkeys(full_result["variable"])) == [
            "monthly_1",
            "monthly_2",
        ]
        assert set(full_result["metric"]) == {metric}
        _assert_equal(full_result, levels_result)
