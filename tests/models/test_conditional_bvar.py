"""Tests for the MIDAS-nowcast -> BVAR tree in ``examples/midas_bvar_tree.py``.

``ConditionalBVAR`` is example code rather than a packaged model, so it is
loaded from the script by path (as ``conftest.py`` does for ``sample_data.py``).
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import compute_accuracy_statistics

pytest.importorskip("bvar")
pytest.importorskip("nowcast_midas")

from forecast_realtime.forecast_tree import ForecastTree, TreeNode
from forecast_realtime.models import ForecastBVAR, ForecastMIDAS

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "midas_bvar_tree.py"
_spec = importlib.util.spec_from_file_location("midas_bvar_tree", _EXAMPLE)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)

ConditionalBVAR = _example.ConditionalBVAR

STEPS = 4
HORIZONS = [0, 1, 2, 3]


def _make_data(seed=0, n_quarters=60):
    """Synthetic quarterly targets driven by their own monthly indicator."""
    rng = np.random.default_rng(seed)
    n_months = n_quarters * 3
    dates_q = pd.date_range("2005-03-31", periods=n_quarters, freq="QE")
    dates_m = pd.date_range("2005-01-31", periods=n_months, freq="ME")

    indicators = {}
    for name in ("gdp_indicator", "cpi_indicator"):
        series = np.zeros(n_months)
        for i in range(1, n_months):
            series[i] = 0.7 * series[i - 1] + rng.standard_normal()
        indicators[name] = series
    X = pd.DataFrame(indicators, index=dates_m)

    targets = {}
    for target, indicator in (("gdp", "gdp_indicator"), ("cpi", "cpi_indicator")):
        values = np.zeros(n_quarters)
        for i, quarter_end in enumerate(dates_q):
            window = X.loc[X.index <= quarter_end, indicator]
            values[i] = 0.5 * window.iloc[-3:].mean() + 0.1 * rng.standard_normal()
        targets[target] = values
    y = pd.DataFrame(targets, index=dates_q)

    return y, X


def _make_leaves():
    return [
        ForecastMIDAS(
            label="gdp", formula="gdp ~ gdp_indicator", horizons=HORIZONS, n_lags=3
        ),
        ForecastMIDAS(
            label="cpi", formula="cpi ~ cpi_indicator", horizons=HORIZONS, n_lags=3
        ),
    ]


def _make_bvar(cls=ConditionalBVAR, **kwargs):
    return cls(
        n_lags=2,
        mode_only=True,
        progressbar=False,
        n_samples=200,
        N_draws=200,
        N_burn=100,
        **kwargs,
    )


def _make_tree(**bvar_kwargs):
    spec = TreeNode(
        name="bvar", children=_make_leaves(), transform=_make_bvar(**bvar_kwargs)
    )
    return ForecastTree(spec=spec, label="midas_bvar")


def test_tree_forecast_has_bvar_shape():
    """The tree returns the BVAR's multivariate forecast."""
    y, X = _make_data()
    tree = _make_tree()
    tree.fit(y, X=X)

    forecast = tree.forecast(steps=STEPS, X=X)

    assert list(forecast.columns) == list(y.columns)
    assert len(forecast) == STEPS
    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert forecast.notna().all().all()


def test_conditioning_takes_values_from_midas_nowcasts():
    """Each MIDAS nowcast constrains its own variable at the matching date."""
    y, X = _make_data()
    tree = _make_tree()
    tree.fit(y, X=X)
    tree.forecast(steps=STEPS, X=X)

    transform = tree.spec.transform
    conditioning = transform.conditioning_
    expected_dates = transform._infer_forecast_dates(y.index, STEPS, frequency="Q")

    assert list(conditioning.columns) == list(y.columns)
    assert list(conditioning.index) == list(expected_dates)

    overlaps = 0
    for label, leaf_forecast in tree.leaf_forecasts_.items():
        for date, value in leaf_forecast.iloc[:, 0].items():
            if date > y.index[-1]:
                overlaps += 1
                assert conditioning.loc[date, label] == value
            else:
                # A nowcast of an already-observed quarter must not constrain.
                assert date not in conditioning.index

    # Two leaves x the three nowcast horizons that fall beyond the fit sample.
    assert overlaps == 6


def test_realtime_tree_matches_midas_at_constrained_horizon(forecast_data):
    import forecast_realtime as rt

    test_vintage = pd.Timestamp("2020-01-31")
    data_transformation = {var: "levels" for var in Y_VARIABLES + X_VARIABLES}
    midas = ForecastMIDAS(
        label="quarterly_a",
        formula="quarterly_a ~ monthly_a",
        horizons=HORIZONS,
        n_lags=3,
    )

    rt_model = rt.RealTimeModel(
        data=forecast_data,
        models=[midas, _make_realtime_tree()],
    )
    rt_model.forecast(
        y_variables=Y_VARIABLES,
        X_variables=X_VARIABLES,
        data_transformation=data_transformation,
        steps=STEPS,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
    )

    constrained = rt_model.data._main_table[
        (rt_model.data._main_table["variable"] == "quarterly_a")
        & (rt_model.data._main_table["metric"] == "levels")
        & rt_model.data._main_table["unique_id"].isin(["quarterly_a", "midas_bvar"])
        & (rt_model.data._main_table["target_minus_vintage"] == 0)
    ]
    constrained = constrained.pivot_table(
        index=["date", "vintage_date_outturn"],
        columns="unique_id",
        values="value_forecast",
    )

    pd.testing.assert_series_equal(
        constrained["quarterly_a"], constrained["midas_bvar"], check_names=False
    )

    accuracy = compute_accuracy_statistics(
        rt_model.data,
        source=["quarterly_a", "midas_bvar"],
        variable="quarterly_a",
        same_date_range=False,
    )._df
    constrained_accuracy = accuracy[
        (accuracy["metric"] == "levels") & (accuracy["horizon"] == 0)
    ].sort_values("unique_id")
    assert len(constrained_accuracy) == 2
    np.testing.assert_allclose(
        constrained_accuracy["rmse"].to_numpy()[0],
        constrained_accuracy["rmse"].to_numpy()[1],
    )


def test_conditioning_keeps_published_values_at_the_ragged_edge():
    """Published targets take precedence while MIDAS fills missing targets."""
    y, X = _make_data()
    last_date = y.index[-1]
    published_cpi = y.loc[last_date, "cpi"]
    y.loc[last_date, "gdp"] = np.nan

    tree = _make_tree()
    tree.fit(y, X=X)
    forecast = tree.forecast(steps=STEPS, X=X)

    conditioning = tree.spec.transform.conditioning_
    gdp_nowcast = tree.leaf_forecasts_["gdp"].loc[last_date, "gdp"]

    assert conditioning.loc[last_date, "gdp"] == gdp_nowcast
    assert conditioning.loc[last_date, "cpi"] == published_cpi
    np.testing.assert_allclose(
        forecast.loc[last_date, ["gdp", "cpi"]],
        conditioning.loc[last_date, ["gdp", "cpi"]],
    )


def test_pop_conditioning_matches_the_bvar_fit_scale():
    """The BVAR transforms levels but keeps MIDAS growth forecasts unchanged."""
    y, X = _make_data()
    y = y + 100.0
    y["unused"] = 200.0
    X = X + 100.0
    data_transformation = {
        "gdp": "pop",
        "cpi": "pop",
        "gdp_indicator": "pop",
        "cpi_indicator": "pop",
    }
    tree = _make_tree(
        formula="gdp + cpi ~ gdp + cpi",
        data_transformation={"gdp": "pop", "cpi": "pop"},
    )

    tree.fit(
        y,
        X=X,
        data_transformation=data_transformation,
        frequency="Q",
    )
    tree.forecast(
        steps=STEPS,
        X=X,
        data_transformation=data_transformation,
        frequency="Q",
    )

    transform = tree.spec.transform
    expected_y = (
        y[["gdp", "cpi"]]
        .pct_change(fill_method=None)
        .mul(100.0)
        .reindex(transform.y.index)
    )
    pd.testing.assert_frame_equal(transform.y, expected_y)
    assert transform.native_metric_mapping() == {"gdp": "pop", "cpi": "pop"}

    for variable, leaf_forecast in tree.leaf_forecasts_.items():
        expected = leaf_forecast.iloc[:, 0].reindex(transform.conditioning_.index)
        pd.testing.assert_series_equal(
            transform.conditioning_[variable], expected, check_names=False
        )


def test_matches_bvar_conditioned_directly():
    """Routing children through X equals conditioning a plain BVAR through y."""
    y, X = _make_data()
    tree = _make_tree()
    tree.fit(y, X=X)
    tree_forecast = tree.forecast(steps=STEPS, X=X)

    conditioning = tree.spec.transform.conditioning_

    plain = _make_bvar(cls=ForecastBVAR)
    plain.fit(y)
    plain_forecast = plain.forecast(steps=STEPS, y=conditioning)

    pd.testing.assert_frame_equal(tree_forecast, plain_forecast, rtol=1e-5, atol=1e-5)


def test_conditioning_binds_and_changes_the_forecast():
    """Constrained entries are reproduced exactly, and they move the BVAR path."""
    y, X = _make_data()
    tree = _make_tree()
    tree.fit(y, X=X)
    forecast = tree.forecast(steps=STEPS, X=X)

    conditioning = tree.spec.transform.conditioning_
    constrained = conditioning.notna().to_numpy()
    assert constrained.any()

    np.testing.assert_allclose(
        forecast.to_numpy()[constrained], conditioning.to_numpy()[constrained]
    )

    unconditional = _make_bvar(cls=ForecastBVAR)
    unconditional.fit(y)
    unconditional_forecast = unconditional.forecast(steps=STEPS)

    # Guards against the conditioning silently being a no-op.
    assert not np.allclose(forecast.to_numpy(), unconditional_forecast.to_numpy())
    assert conditioning.iloc[-1].isna().all()


def test_bvar_is_fitted_on_the_full_sample():
    """Leaf fitted values must not truncate the BVAR's estimation sample."""
    y, X = _make_data()
    tree = _make_tree()
    tree.fit(y, X=X)

    pd.testing.assert_frame_equal(tree.spec.transform.y, y)


def test_nowcast_of_unobserved_quarter_conditions_the_bvar():
    """Ragged edge: y ends a quarter before the monthly indicators.

    The MIDAS h=0 nowcast then lands on a quarter the BVAR has not observed,
    so every horizon is constrained.
    """
    y, X = _make_data()
    y_ragged = y.iloc[:-1]

    tree = _make_tree()
    tree.fit(y_ragged, X=X)
    forecast = tree.forecast(steps=STEPS, X=X)

    conditioning = tree.spec.transform.conditioning_
    assert conditioning.notna().all().all()

    for leaf_forecast in tree.leaf_forecasts_.values():
        assert leaf_forecast.index[0] > y_ragged.index[-1]

    np.testing.assert_allclose(forecast.to_numpy(), conditioning.to_numpy())


def test_conditioning_steps_limits_constrained_horizons():
    """Horizons beyond conditioning_steps are left unconstrained."""
    y, X = _make_data()
    tree = _make_tree(conditioning_steps=1)
    tree.fit(y, X=X)
    tree.forecast(steps=STEPS, X=X)

    conditioning = tree.spec.transform.conditioning_

    assert conditioning.iloc[0].notna().any()
    assert conditioning.iloc[1:].isna().all().all()


def test_unmatched_children_raise():
    """A child whose label is not a fitted variable cannot condition anything."""
    y, X = _make_data()
    leaf = ForecastMIDAS(
        label="not_a_variable",
        formula="gdp ~ gdp_indicator",
        horizons=HORIZONS,
        n_lags=3,
    )
    spec = TreeNode(name="bvar", children=[leaf], transform=_make_bvar())
    tree = ForecastTree(spec=spec, label="midas_bvar")
    tree.fit(y, X=X)

    with pytest.raises(ValueError, match="none of the conditioning columns"):
        tree.forecast(steps=STEPS, X=X)


def test_invalid_conditioning_steps_raise():
    with pytest.raises(ValueError, match="conditioning_steps must be"):
        ConditionalBVAR(conditioning_steps=0)


def test_conditioning_components_must_be_dated():
    y, X = _make_data()
    model = _make_bvar()
    model.fit(y)

    components = pd.DataFrame({"gdp": [1.0]}, index=[0])

    with pytest.raises(TypeError, match="DatetimeIndex"):
        model.forecast(steps=STEPS, X=components)


# ---------------------------------------------------------------------- #
# The tree driven by RealTimeModel over a vintage.                        #
# ---------------------------------------------------------------------- #
Y_VARIABLES = ["quarterly_a", "quarterly_b"]
X_VARIABLES = ["monthly_a", "monthly_b"]


def _make_realtime_tree():
    leaves = [
        ForecastMIDAS(
            label="quarterly_a",
            formula="quarterly_a ~ monthly_a",
            horizons=HORIZONS,
            n_lags=3,
        ),
        ForecastMIDAS(
            label="quarterly_b",
            formula="quarterly_b ~ monthly_b",
            horizons=HORIZONS,
            n_lags=3,
        ),
    ]
    spec = TreeNode(name="bvar", children=leaves, transform=_make_bvar())
    return ForecastTree(spec=spec, label="midas_bvar")


def _vintage_frames(forecast_data, vintage):
    """Rebuild the y/X a vintage exposes, mirroring the RealTimeModel loop."""
    outturns = forecast_data.outturns.copy()
    outturns = outturns[
        outturns["variable"].isin(Y_VARIABLES + X_VARIABLES)
        # ForecastData also stores derived metrics; the loop keeps the one named
        # by data_transformation, and mixing them corrupts the pivot.
        & (outturns["metric"] == "levels")
    ]
    at_vintage = outturns[outturns["vintage_date"] <= vintage]
    at_vintage = at_vintage.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )

    y_wide = at_vintage[at_vintage["variable"].isin(Y_VARIABLES)].pivot(
        index="date", columns="variable", values="value"
    )

    X_wide = at_vintage[at_vintage["variable"].isin(X_VARIABLES)].pivot(
        index="date", columns="variable", values="value"
    )
    return y_wide, X_wide


def test_realtime_tree_matches_manual_fit(forecast_data):
    """RealTimeModel drives the tree and reproduces a manual fit on the vintage."""
    import forecast_realtime as rt

    test_vintage = pd.Timestamp("2020-01-31")
    data_transformation = {var: "levels" for var in Y_VARIABLES + X_VARIABLES}

    rt_model = rt.RealTimeModel(data=forecast_data, models=_make_realtime_tree())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rt_model.forecast(
            y_variables=Y_VARIABLES,
            X_variables=X_VARIABLES,
            data_transformation=data_transformation,
            steps=STEPS,
            first_vintage=str(test_vintage.date()),
            last_vintage=str(test_vintage.date()),
        )

    published_warnings = [
        str(warning.message)
        for warning in captured
        if "forecasts cover values already published" in str(warning.message)
    ]
    assert not published_warnings, published_warnings

    produced = rt_model.data.forecasts
    produced = produced[
        (produced["source"] == "midas_bvar")
        & (produced["vintage_date"] == test_vintage)
        & (produced["metric"] == "levels")
    ]
    assert not produced.empty
    assert set(produced["variable"]) == set(Y_VARIABLES)

    y_manual, X_manual = _vintage_frames(forecast_data, test_vintage)
    last_complete_date = y_manual.dropna().index.max()
    manual_tree = _make_realtime_tree()
    manual_tree.fit(y_manual.loc[:last_complete_date], X=X_manual)
    manual_forecast = manual_tree.forecast(steps=STEPS, X=X_manual, y=y_manual)

    # MIDAS anchors its nowcast on the last regressor quarter, so only the
    # horizons it actually covers are constrained.
    conditioning = manual_tree.spec.transform.conditioning_
    assert conditioning.notna().any().any()

    for variable in Y_VARIABLES:
        realtime_rows = produced[produced["variable"] == variable].sort_values("date")
        expected = manual_forecast.loc[pd.DatetimeIndex(realtime_rows["date"]), variable]
        np.testing.assert_allclose(
            realtime_rows["value"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
