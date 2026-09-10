"""Conditioning policies preserve calendar horizons, units and raw-input ownership."""

import numpy as np
import pandas as pd
import pytest

from forecast_realtime import ForecastTree, TreeNode
from forecast_realtime._model_data import ModelData
from forecast_realtime.real_time_model import _resolve_conditioning
from tests.test_conditioning_boundaries import RecordingModel, SupportingModel


def _resolve(model, y_variables, X_variables, **fallback):
    options = dict(y_sources=None, X_sources=None, y_steps_ahead=None, X_steps_ahead=None)
    options.update(fallback)
    return _resolve_conditioning(
        model,
        model.input_requirements(y_variables, X_variables),
        y_variables,
        options,
        {"A", "B"},
        steps=3,
    )


def test_tree_policy_selects_raw_X_not_component_regressors():
    leaf = RecordingModel(label="leaf", formula="target ~ oil")
    root = SupportingModel(label="root-model", formula="target ~ leaf")
    spec = TreeNode(root, [leaf], name="root")
    tree = ForecastTree(spec, conditioning={"X": {"oil": {"source": "A", "periods": 2}}})
    assert _resolve(tree, ["target"], ["oil"])["X_sources"] == {"oil": "A"}
    wrong = ForecastTree(
        spec, conditioning={"X": {"leaf": {"source": "A", "periods": 1}}}
    )
    with pytest.raises(ValueError, match="leaf.*not a selected raw input"):
        _resolve(wrong, ["target"], ["oil"])


def test_tree_policy_uses_root_targets_not_union_of_consumers():
    spec = TreeNode(
        SupportingModel(formula="gdp ~ leaf"),
        [RecordingModel(label="leaf", formula="inflation ~ oil")],
        name="root",
    )
    owned = ForecastTree(
        spec, conditioning={"y": {"inflation": {"source": "A", "periods": 1}}}
    )
    with pytest.raises(ValueError, match="inflation.*not a selected raw input"):
        _resolve(owned, ["gdp", "inflation"], ["oil"])
    inherited = _resolve(
        ForecastTree(spec),
        ["gdp", "inflation"],
        ["oil"],
        y_sources={"gdp": "A", "inflation": "B"},
        y_steps_ahead={"gdp": 0, "inflation": 2},
    )
    assert inherited["y_sources"] == {"gdp": "A"}
    assert inherited["y_steps_ahead"] == {"gdp": 0}


def test_model_owned_mixed_frequency_X_duration_uses_quarters_not_rows():
    vintage = pd.Timestamp("2020-03-31")
    rows = []
    for variable, frequency, dates in (
        ("target", "Q", pd.date_range("2019-03-31", periods=5, freq="QE")),
        ("oil", "M", pd.date_range("2019-01-31", periods=15, freq="ME")),
    ):
        rows.extend(
            dict(
                date=date,
                variable=variable,
                frequency=frequency,
                vintage_date=vintage,
                metric="levels",
                source="survey",
                value=100.0,
            )
            for date in dates
        )
    dates = pd.date_range("2020-04-30", periods=9, freq="ME")
    supplied = pd.DataFrame(
        {
            "date": dates.delete(1),
            "variable": "oil",
            "frequency": "M",
            "vintage_date": vintage,
            "metric": "pop",
            "source": "A",
            "value": 5.0,
        }
    )
    source_data = ModelData.from_archive(
        pd.DataFrame(rows), supplied, frequencies={"target": "Q", "oil": "M"}
    )
    model = RecordingModel(
        formula="target ~ oil",
        data_transformation={"target": "levels", "oil": "pop"},
        conditioning={"X": {"oil": {"source": "A", "periods": 2}}},
    )
    resolved = _resolve(model, ["target"], ["oil"])
    selected = source_data.select(
        model.input_requirements(["target"], ["oil"]), X_sources=resolved["X_sources"]
    )
    path = selected.as_of(vintage).condition(
        "2020-06-30", 3, "Q", X_steps_ahead=resolved["X_steps_ahead"]
    )
    X = path.to_wide("X", "conditioning")
    np.testing.assert_allclose(
        X["oil"],
        [5.0, np.nan, 5.0, 5.0, 5.0, 5.0, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    assert path.metrics("X", "conditioning") == {"oil": "pop"}
    entry = next(entry for entry in path._catalogue if entry["variable"] == "oil")
    assert entry["source"] == "A"


def test_direct_tree_rejects_unknown_root_target_without_formula():
    history = pd.DataFrame(
        {"gdp": [10.0, 11.0, 12.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    tree = ForecastTree(
        TreeNode(SupportingModel(), [RecordingModel(label="leaf")], name="root")
    ).fit(history)
    path = pd.DataFrame({"inflation": [5.0]}, index=pd.to_datetime(["2020-04-30"]))
    with pytest.raises(ValueError, match="inflation.*not fitted targets"):
        tree.forecast(y=path)
