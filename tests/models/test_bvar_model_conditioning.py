"""Model-owned constraints reach real BVARs without changing algorithms."""

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

pytest.importorskip("bvar")

from forecast_realtime import ForecastTree, RealTimeModel, TreeNode
from forecast_realtime.examples.midas_bvar_tree import ConditionalBVAR
from forecast_realtime.models import ForecastBVAR, ForecastOLS

pytestmark = pytest.mark.usefixtures("bvar_python_kernel")


def _history():
    rng = np.random.default_rng(31)
    return pd.DataFrame(
        rng.normal(size=(36, 2)).cumsum(axis=0) + 100,
        columns=["gdp", "cpi"],
        index=pd.date_range("2010-03-31", periods=36, freq="QE"),
    )


def _bvar(cls=ForecastBVAR, **kwargs):
    return cls(n_lags=1, mode_only=True, nb_restart=0, progressbar=False, **kwargs)


def test_bvar_multi_model_sources_and_durations_match_separate_runs():
    history = _history()
    vintage = history.index[-1]
    dates = pd.date_range(vintage + pd.offsets.QuarterEnd(), periods=3, freq="QE")
    outturns = (
        history.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="variable", value_name="value")
        .assign(vintage_date=vintage, frequency="Q", metric="levels")
    )
    forecasts = pd.DataFrame(
        [
            dict(
                date=date,
                variable="gdp",
                value=value,
                vintage_date=vintage,
                source=source,
                frequency="Q",
                metric="levels",
                forecast_horizon=horizon,
                target_minus_vintage=horizon + 1,
            )
            for source, value in (("A", 90.0), ("B", 110.0))
            for horizon, date in enumerate(dates)
        ]
    )
    policies = {
        "baseline": {},
        "short": {"y": {"gdp": {"source": "A", "periods": 1}}},
        "long": {"y": {"gdp": {"source": "B", "periods": 3}}},
    }

    def run(names):
        data = ForecastData(outturns_data=outturns, metric="levels", compute_levels=False)
        data._raw_forecasts = forecasts.copy()
        result = RealTimeModel(
            data, [_bvar(label=name, conditioning=policies[name]) for name in names]
        )
        result.forecast(
            y_variables=["gdp", "cpi"],
            steps=3,
            first_forecast_horizon=1,
            first_vintage=vintage,
            last_vintage=vintage,
        )
        return (
            data.forecasts.query("source in @names and metric == 'levels'")
            .sort_values(["source", "variable", "date"])
            .reset_index(drop=True)
        )

    together = run(list(policies))
    separate = (
        pd.concat([run([name]) for name in policies], ignore_index=True)
        .sort_values(["source", "variable", "date"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(together, separate)
    assert together.query("source == 'short' and variable == 'gdp'")["value"].iloc[
        0
    ] == pytest.approx(90.0)
    np.testing.assert_allclose(
        together.query("source == 'long' and variable == 'gdp'")["value"], 110.0
    )
    assert not np.allclose(
        together.query("source == 'baseline'")["value"],
        together.query("source == 'long'")["value"],
    )


def test_bvar_sparse_and_overlong_direct_paths_use_requested_calendar():
    history = _history()
    model = _bvar(
        conditioning={"y": {"gdp": {"source": "unavailable-source", "periods": 1}}}
    ).fit(history)
    dates = pd.date_range(
        history.index[-1] + pd.offsets.QuarterEnd(), periods=5, freq="QE"
    )
    path = pd.DataFrame(
        {"gdp": [90.0, 110.0], "cpi": [np.nan, np.nan]}, index=dates[[1, 4]]
    )
    forecast = model.forecast(steps=3, y=path)
    assert forecast.loc[dates[1], "gdp"] == pytest.approx(90.0)
    assert len(forecast) == 3
    outside = model.forecast(steps=3, y=path.iloc[1:])
    pd.testing.assert_frame_equal(outside, model.forecast(steps=3))


def test_conditional_bvar_root_does_not_clip_explicit_constraints():
    history = _history()
    leaves = [
        ForecastOLS(label=variable, formula=f"{variable} ~ .") for variable in history
    ]
    root = _bvar(ConditionalBVAR, conditioning_steps=1)
    tree = ForecastTree(TreeNode(root, leaves, name="root")).fit(history)
    dates = pd.date_range(
        history.index[-1] + pd.offsets.QuarterEnd(), periods=3, freq="QE"
    )
    path = pd.DataFrame(np.nan, index=dates, columns=history.columns)
    path.iloc[0, 0] = 90.0
    result = tree.forecast(steps=3, y=path)
    assert result.iloc[0, 0] == pytest.approx(90.0)
    path.iloc[2, 0] = 110.0
    with pytest.raises(ValueError, match="beyond conditioning_steps"):
        tree.forecast(steps=3, y=path)
    # Child-only nowcast behaviour is unchanged after the rejected request.
    assert len(tree.forecast(steps=3)) == 3
