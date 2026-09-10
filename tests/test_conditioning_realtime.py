"""Acceptance tests for model-owned realtime conditioning orchestration."""

import copy

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

from forecast_realtime import ForecastModel, RealTimeModel


class EchoAdditiveModel(ForecastModel):
    """Pickleable model whose output exposes the supplied conditioning paths."""

    _supports_target_conditioning = True

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        forecast_dates = self._infer_forecast_dates(
            (y if y is not None else X).index,
            steps,
            frequency=self._forecast_frequency,
            start=kwargs["forecast_origin"],
        )

        def requested(frame):
            return frame.reindex(forecast_dates)

        result = np.zeros((steps, len(self.y.columns)), dtype=float)
        if y is not None:
            result += np.nan_to_num(requested(y).to_numpy())
        if X is not None:
            result += np.nan_to_num(
                requested(X).iloc[:, : len(self.y.columns)].to_numpy()
            )
        return result

    def _forecast_decomp(self, steps, X=None, y=None, **kwargs):
        forecast_dates = self._infer_forecast_dates(
            (y if y is not None else X).index,
            steps,
            frequency=self._forecast_frequency,
            start=kwargs["forecast_origin"],
        )

        def requested(frame):
            return frame.reindex(forecast_dates)

        values = np.zeros(steps, dtype=float)
        if y is not None:
            values += np.nan_to_num(requested(y).iloc[:, 0].to_numpy())
        if X is not None:
            values += np.nan_to_num(requested(X).iloc[:, 0].to_numpy())
        return pd.DataFrame(
            {
                "forecast_horizon": np.arange(steps),
                "component": "echo",
                "contribution": values,
                "weight": np.ones(steps),
            }
        )


def _source_data(*, include_x=True, second_vintage=False, later_only_source=False):
    """Build a small source dataset with two deterministic conditioning sources."""
    history_dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows = []
    for variable, values in {
        "target_a": [10.0, 11.0, 12.0],
        "target_b": [20.0, 21.0, 22.0],
    }.items():
        rows.extend(
            {
                "date": date,
                "variable": variable,
                "vintage_date": date,
                "frequency": "M",
                "value": value,
                "metric": "levels",
            }
            for date, value in zip(history_dates, values, strict=True)
        )
    if include_x:
        rows.extend(
            {
                "date": date,
                "variable": "driver",
                "vintage_date": date,
                "frequency": "M",
                "value": value,
                "metric": "levels",
            }
            for date, value in zip(history_dates, [1.0, 2.0, 3.0], strict=True)
        )

    forecast_dates = pd.date_range("2020-03-31", periods=3, freq="ME")
    for source, value in (("A", 100.0), ("B", 200.0)):
        for variable, offset in (("target_a", 0.0), ("target_b", 10.0)):
            rows.extend(
                {
                    "date": date,
                    "variable": variable,
                    "vintage_date": pd.Timestamp("2020-03-31"),
                    "source": source,
                    "frequency": "M",
                    "value": value + offset + horizon,
                    "metric": "levels",
                    "forecast_horizon": horizon,
                }
                for horizon, date in enumerate(forecast_dates)
            )
        if include_x:
            rows.extend(
                {
                    "date": date,
                    "variable": "driver",
                    "vintage_date": pd.Timestamp("2020-03-31"),
                    "source": source,
                    "frequency": "M",
                    "value": value / 100 + horizon,
                    "metric": "levels",
                    "forecast_horizon": horizon,
                }
                for horizon, date in enumerate(forecast_dates)
            )
    if second_vintage:
        rows.extend(
            {
                "date": pd.Timestamp("2020-04-30"),
                "variable": variable,
                "vintage_date": pd.Timestamp("2020-04-30"),
                "frequency": "M",
                "value": value,
                "metric": "levels",
            }
            for variable, value in (
                ("target_a", 13.0),
                ("target_b", 23.0),
                ("driver", 4.0),
            )
        )
        rows.extend(
            {
                "date": date,
                "variable": "target_a",
                "vintage_date": pd.Timestamp("2020-04-30"),
                "source": "A",
                "frequency": "M",
                "value": value,
                "metric": "levels",
                "forecast_horizon": horizon,
            }
            for horizon, (date, value) in enumerate(
                zip(
                    pd.date_range("2020-05-31", periods=3, freq="ME"),
                    [999.0, 1000.0, 1001.0],
                    strict=True,
                )
            )
        )
        if later_only_source:
            rows.extend(
                {
                    "date": date,
                    "variable": "target_a",
                    "vintage_date": pd.Timestamp("2020-04-30"),
                    "source": "C",
                    "frequency": "M",
                    "value": value,
                    "metric": "levels",
                    "forecast_horizon": horizon,
                }
                for horizon, (date, value) in enumerate(
                    zip(
                        pd.date_range("2020-05-31", periods=3, freq="ME"),
                        [777.0, 778.0, 779.0],
                        strict=True,
                    )
                )
            )
    source_data = pd.DataFrame(rows)
    outturns = source_data.loc[source_data["source"].isna()].drop(columns="source")
    forecasts = source_data.loc[source_data["source"].notna()]
    return ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )


def _run(data, models, **kwargs):
    """Run one forecast batch with the compact source-data defaults."""
    options = {
        "first_vintage": "2020-03-31",
        "last_vintage": "2020-03-31",
    }
    options.update(kwargs)
    return RealTimeModel(data=data, models=models).forecast(
        y_variables=["target_a", "target_b"],
        X_variables=["driver"],
        data_transformation={
            "target_a": "levels",
            "target_b": "levels",
            "driver": "levels",
        },
        steps=3,
        first_forecast_horizon=0,
        **options,
    )


def _values(model):
    """Return the newly published compact forecast values by source."""
    return (
        model.data.forecasts.query("source in ['left', 'right', 'EchoAdditiveModel']")
        .sort_values(["source", "variable", "date"])[
            ["source", "variable", "date", "forecast_horizon", "value"]
        ]
        .reset_index(drop=True)
    )


def test_model_policies_drive_disjoint_y_and_x_sources_and_durations():
    """Each model receives its own y/X paths and positive durations become horizons."""
    left = EchoAdditiveModel(
        label="left",
        formula="target_a ~ driver",
        conditioning={
            "y": {"target_a": {"source": "A", "periods": 1}},
            "X": {"driver": {"source": "A", "periods": 3}},
        },
    )
    right = EchoAdditiveModel(
        label="right",
        formula="target_b ~ driver",
        conditioning={
            "y": {"target_b": {"source": "B", "periods": 3}},
            "X": {"driver": {"source": "B", "periods": 1}},
        },
    )
    model = _run(_source_data(), [left, right])
    values = model.data.forecasts.query(
        "source in ['left', 'right'] and metric == 'levels'"
    )

    left_values = values.query("source == 'left' and variable == 'target_a'").sort_values(
        "date"
    )
    right_values = values.query(
        "source == 'right' and variable == 'target_b'"
    ).sort_values("date")
    np.testing.assert_allclose(left_values["value"], [101.0, 2.0, 3.0])
    np.testing.assert_allclose(right_values["value"], [212.0, 211.0, 212.0])
    assert list(left_values["forecast_horizon"]) == [0, 1, 2]
    assert list(right_values["forecast_horizon"]) == [0, 1, 2]


def test_same_variable_supports_different_periods_and_matches_separate_runs():
    """The same target may use different sources and conditioning durations."""
    specifications = [
        EchoAdditiveModel(
            label="baseline",
            formula="target_a ~ driver",
            conditioning={},
        ),
        EchoAdditiveModel(
            label="A1",
            formula="target_a ~ driver",
            conditioning={"y": {"target_a": {"source": "A", "periods": 1}}},
        ),
        EchoAdditiveModel(
            label="B3",
            formula="target_a ~ driver",
            conditioning={"y": {"target_a": {"source": "B", "periods": 3}}},
        ),
    ]
    combined = _run(_source_data(), copy.deepcopy(specifications))
    separate = pd.concat(
        [
            _run(_source_data(), copy.deepcopy(model)).data.forecasts
            for model in specifications
        ],
        ignore_index=True,
    )
    actual = combined.data.forecasts.query(
        "source in ['baseline', 'A1', 'B3'] and variable == 'target_a' "
        "and metric == 'levels'"
    ).sort_values(["source", "date"])["value"]
    expected = pd.Series(
        [103.0, 0.0, 0.0, 203.0, 201.0, 202.0, 15.0, 0.0, 0.0],
        dtype=float,
    )
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())
    expected_separate = separate.query(
        "source in ['baseline', 'A1', 'B3'] and variable == 'target_a' "
        "and metric == 'levels'"
    ).sort_values(["source", "date"])
    pd.testing.assert_frame_equal(
        combined.data.forecasts.query(
            "source in ['baseline', 'A1', 'B3'] and metric == 'levels'"
        )
        .sort_values(["source", "variable", "date"])
        .reset_index(drop=True),
        expected_separate.sort_values(["source", "variable", "date"]).reset_index(
            drop=True
        ),
    )


def test_x_only_and_mixed_policies_preserve_none_empty_and_imputed_paths():
    """X-only and mixed policies preserve legacy None/{} intent and impute future X."""
    x_only = EchoAdditiveModel(
        label="x-only",
        formula="target_a ~ driver",
        conditioning={"X": {"driver": {"source": "B", "periods": 2}}},
    )
    mixed = EchoAdditiveModel(
        label="mixed",
        formula="target_a ~ driver",
        conditioning={
            "y": {"target_a": {"source": "A", "periods": 2}},
            "X": {"driver": {"source": "B", "periods": 1}},
        },
    )
    data = _source_data()
    RealTimeModel(data=data, models=[x_only, mixed]).forecast(
        y_variables=["target_a"],
        X_variables=["driver"],
        data_transformation={"target_a": "levels", "driver": "levels"},
        steps=3,
        first_forecast_horizon=0,
        first_vintage="2020-03-31",
        last_vintage="2020-03-31",
        y_sources={},
        y_steps_ahead={},
        X_imputation="last",
    )
    values = data.forecasts.query("source in ['x-only', 'mixed'] and metric == 'levels'")
    assert len(values) == 6
    np.testing.assert_allclose(
        values.query("variable == 'target_a'").sort_values(["source", "date"])["value"],
        [102.0, 103.0, 2.0, 14.0, 3.0, 3.0],
    )

    inherited = EchoAdditiveModel(label="inherited", formula="target_a ~ driver")
    disabled = EchoAdditiveModel(
        label="disabled", formula="target_a ~ driver", conditioning={}
    )
    inherited_model = _run(
        _source_data(),
        inherited,
        y_sources={"target_a": "A"},
        y_steps_ahead={"target_a": 0},
        X_sources={"driver": "B"},
        X_steps_ahead={"driver": 0},
        X_imputation="last",
    )
    disabled_model = _run(
        _source_data(),
        disabled,
        y_sources={},
        y_steps_ahead={},
        X_sources={},
        X_steps_ahead={},
        X_imputation="last",
    )
    inherited_values = inherited_model.data.forecasts.query(
        "source == 'inherited' and variable == 'target_a' and metric == 'levels'"
    ).sort_values("date")["value"]
    disabled_values = disabled_model.data.forecasts.query(
        "source == 'disabled' and variable == 'target_a' and metric == 'levels'"
    ).sort_values("date")["value"]
    np.testing.assert_allclose(inherited_values, [102.0, 2.0, 2.0])
    np.testing.assert_allclose(disabled_values, [15.0, 3.0, 3.0])

    explicit = EchoAdditiveModel(
        label="explicit",
        formula="target_a ~ driver",
        conditioning={
            "y": {"target_a": {"source": "A", "periods": 1}},
            "X": {"driver": {"source": "B", "periods": 1}},
        },
    )
    explicit_model = _run(_source_data(), explicit, X_imputation="last")
    explicit_values = explicit_model.data.forecasts.query(
        "source == 'explicit' and variable == 'target_a' and metric == 'levels'"
    ).sort_values("date")["value"]
    np.testing.assert_allclose(explicit_values, inherited_values)


def test_conditioning_and_fallback_are_captured_before_caller_mutation(monkeypatch):
    """Mutating caller mappings after task construction cannot alter execution."""
    policy = {"y": {"target_a": {"source": "A", "periods": 2}}}
    fallback = {"y_sources": {"target_a": "B"}, "y_steps_ahead": {"target_a": 0}}
    model = EchoAdditiveModel(
        label="captured", formula="target_a ~ .", conditioning=policy
    )
    original = RealTimeModel._execute_forecast_tasks
    captured = []

    def capture(tasks, **options):
        captured.extend(copy.deepcopy(tasks))
        policy["y"]["target_a"]["source"] = "B"
        fallback["y_sources"]["target_a"] = "A"
        return original(tasks, **options)

    monkeypatch.setattr(RealTimeModel, "_execute_forecast_tasks", staticmethod(capture))
    result = RealTimeModel(data=_source_data(include_x=False), models=model).forecast(
        y_variables=["target_a"],
        data_transformation={"target_a": "levels"},
        steps=2,
        first_forecast_horizon=0,
        first_vintage="2020-03-31",
        last_vintage="2020-03-31",
        y_sources=fallback["y_sources"],
        y_steps_ahead=fallback["y_steps_ahead"],
    )
    captured_y = captured[0].data.to_wide("y", "conditioning")
    assert captured_y.iloc[0, 0] == 100.0
    assert captured[0].options["y_steps_ahead"] == {"target_a": 1}
    assert result.data.forecasts.query("source == 'captured'")["value"].iloc[0] == 100.0


def test_model_owned_policy_parallel_equals_sequential():
    """The same specifications produce the same frame in serial and process modes."""
    models = [
        EchoAdditiveModel(
            label="left",
            formula="target_a ~ driver",
            conditioning={"y": {"target_a": {"source": "A", "periods": 2}}},
        ),
        EchoAdditiveModel(
            label="right",
            formula="target_b ~ driver",
            conditioning={"y": {"target_b": {"source": "B", "periods": 2}}},
        ),
    ]
    sequential = _run(_source_data(), copy.deepcopy(models), parallel=False)
    parallel = _run(_source_data(), copy.deepcopy(models), parallel=True, max_workers=2)
    sequential_forecasts = sequential.data.forecasts.sort_values(
        ["source", "variable", "date", "vintage_date", "forecast_horizon"]
    ).reset_index(drop=True)
    parallel_forecasts = parallel.data.forecasts.sort_values(
        ["source", "variable", "date", "vintage_date", "forecast_horizon"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(sequential_forecasts, parallel_forecasts)
    np.testing.assert_allclose(
        sequential_forecasts.query(
            "source == 'left' and variable == 'target_a' and metric == 'levels'"
        )["value"],
        [103.0, 101.0, 0.0],
    )


def test_sequential_revisions_reconcile_to_revised_forecasts():
    """A later vintage records both the new level and its forecast revision."""
    model = EchoAdditiveModel(
        label="revision",
        formula="target_a ~ driver",
        conditioning={"y": {"target_a": {"source": "A", "periods": 3}}},
    )
    result = _run(
        _source_data(second_vintage=True),
        model,
        last_vintage="2020-04-30",
        decomp=True,
    )
    forecasts = result.data.forecasts.query(
        "source == 'revision' and variable == 'target_a' and metric == 'levels'"
    ).sort_values(["vintage_date", "date"])
    np.testing.assert_allclose(
        forecasts["value"], [103.0, 101.0, 102.0, 105.0, 999.0, 1000.0]
    )
    revisions = result.decompositions.query(
        "decomposition == 'revision' and variable == 'target_a'"
    ).sort_values(["date", "forecast_horizon"])
    revision_totals = revisions.groupby(["date", "forecast_horizon"])[
        "contribution"
    ].sum()
    np.testing.assert_allclose(revision_totals, [-103.0, 4.0, 897.0, 1000.0])
    levels = result.decompositions.query(
        "decomposition == 'level' and variable == 'target_a'"
    )
    revised = (
        levels.query("vintage_date == '2020-04-30'")
        .groupby(["date", "forecast_horizon"])["contribution"]
        .sum()
    )
    prior = levels.query("vintage_date == '2020-03-31'").set_index(
        ["date", "forecast_horizon"]
    )["contribution"]
    revision = revision_totals
    prior = prior.reindex(revised.index, fill_value=0.0)
    np.testing.assert_allclose(
        revised.sort_index().to_numpy(),
        prior.add(revision, fill_value=0).reindex(revised.index).sort_index().to_numpy(),
    )


def test_later_only_source_does_not_fill_an_earlier_vintage():
    """A source first released later remains unavailable at earlier as-of dates."""
    model = EchoAdditiveModel(
        label="later-only",
        formula="target_a ~ driver",
        conditioning={"y": {"target_a": {"source": "C", "periods": 3}}},
    )
    result = _run(
        _source_data(second_vintage=True, later_only_source=True),
        model,
        last_vintage="2020-04-30",
    )
    values = result.data.forecasts.query(
        "source == 'later-only' and variable == 'target_a' and metric == 'levels'"
    ).sort_values(["vintage_date", "date"])
    assert list(values["vintage_date"].drop_duplicates()) == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    ]
    np.testing.assert_allclose(
        values.groupby("vintage_date")["value"].first(), [15.0, 17.0]
    )
    np.testing.assert_allclose(
        values.loc[values["vintage_date"].eq("2020-03-31"), "value"],
        [15.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        values.loc[values["vintage_date"].eq("2020-04-30"), "value"],
        [17.0, 777.0, 778.0],
    )


def test_real_bvar_uses_actual_conditional_horizons(sample_realtime_ragged, request):
    """The real BVAR wrapper changes its path for an actual conditional horizon."""
    pytest.importorskip("bvar")
    request.getfixturevalue("bvar_python_kernel")
    import forecast_realtime as rt

    y = sample_realtime_ragged.query(
        "metric == 'levels' and variable in ['quarterly_1', 'quarterly_2']"
    ).pivot_table(index="date", columns="variable", values="value")
    y = y.dropna().tail(40)
    plain_model = rt.models.ForecastBVAR(
        n_lags=1, nb_restart=0, mode_only=True, optim_random_state=0
    ).fit(y=y)
    constrained_model = rt.models.ForecastBVAR(
        n_lags=1, nb_restart=0, mode_only=True, optim_random_state=0
    ).fit(y=y)
    horizon = 3
    dates = pd.date_range(
        y.index[-1] + pd.offsets.QuarterEnd(), periods=horizon, freq="QE"
    )
    constraint = pd.DataFrame(np.nan, index=dates, columns=y.columns)
    constraint.iloc[:, 0] = y.iloc[-1, 0]
    plain = plain_model.forecast(steps=horizon)
    constrained = constrained_model.forecast(steps=horizon, y=constraint)
    assert list(constrained.index) == list(dates)
    assert constrained.shape == (horizon, 2)
    assert not np.allclose(plain.iloc[:, 0], constrained.iloc[:, 0])
