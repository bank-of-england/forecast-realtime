"""Snapshot regression for the compact model demonstration."""

import importlib

import numpy as np
import pandas as pd
import pytest

from tests.realtime_fixtures import generate_synthetic_data
from tests.schemas import decomposition_schema

# The demo builds models from the complete ``[models]`` extra.
for dependency in ("sklearn", "bvar", "nowcast_midas"):
    pytest.importorskip(dependency)

run_demo = pytest.importorskip("forecast_realtime.examples.demo_models").run_demo

_SORT_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "metric",
    "frequency",
    "source",
]

_DECOMPOSITION_SORT_COLUMNS = [
    "date",
    "vintage_date",
    "base_vintage_date",
    "forecast_horizon",
    "variable",
    "component",
    "decomposition",
    "revision_source",
    "forecast_metric",
    "source",
    "frequency",
]

_DECOMPOSITION_KEY_COLUMNS = [
    "source",
    "vintage_date",
    "base_vintage_date",
    "date",
    "forecast_horizon",
    "variable",
    "component",
    "decomposition",
    "revision_source",
]


def test_packaged_example_modules_import():
    """Every shipped example module is importable without external runtimes."""
    for module in ("demo_models", "demo_models_fable", "midas_bvar_tree"):
        imported = importlib.import_module(f"forecast_realtime.examples.{module}")
        assert imported.__package__ == "forecast_realtime.examples"


def snapshot_forecasts(frame):
    """Normalise the demo's forecast records for a deterministic snapshot."""
    result = frame.sort_values(_SORT_COLUMNS).reset_index(drop=True)
    for column in ("date", "vintage_date"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    result["value"] = result["value"].round(5)
    return result.to_dict(orient="records")


def snapshot_decompositions(frame):
    """Normalise decomposition records for a deterministic snapshot."""
    result = frame.sort_values(_DECOMPOSITION_SORT_COLUMNS).reset_index(drop=True)
    for column in ("date", "vintage_date", "base_vintage_date"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
        result[column] = result[column].where(result[column].notna(), None)
    for column in ("contribution", "weight", "news"):
        result[column] = result[column].round(5)
    result["revision_source"] = result["revision_source"].astype(object)
    result["revision_source"] = result["revision_source"].where(
        result["revision_source"].notna(), None
    )
    return result.to_dict(orient="records")


@pytest.fixture(autouse=True)
def synthetic_inputs(monkeypatch):
    """Exercise the demo models without rebuilding transformations per vintage."""
    monkeypatch.setattr(
        "forecast_realtime.examples.demo_models.rt.generate_synthetic_data",
        generate_synthetic_data,
    )


def test_demo_models(snapshot, bvar_python_kernel):
    """The default demo includes every model and matches its numerical snapshot."""
    sequential = run_demo(N_vintages=6)

    expected_sources = {
        "Big OLS",
        "Small OLS",
        "Ridge",
        "LASSO",
        "Elastic Net",
        "Bridge OLS",
        "MIDAS",
        "BVAR",
        "MIDAS BVAR",
    }
    assert set(sequential.data.forecasts["source"]) == expected_sources
    assert snapshot_forecasts(sequential.data.forecasts) == snapshot


def test_demo_models_decompositions(snapshot, bvar_python_kernel):
    """Demo decompositions reconstruct native forecasts and revisions."""
    result = run_demo(
        N_vintages=2,
        decomp=True,
        reconstruct_levels=False,
    )

    assert result.decompositions is not None
    assert not result.decompositions.empty
    assert set(result.decompositions["source"]) == {
        "Big OLS",
        "Small OLS",
        "Ridge",
        "LASSO",
        "Elastic Net",
        "Bridge OLS",
        "MIDAS",
    }
    decomposition_schema.validate(result.decompositions)
    assert np.isfinite(result.decompositions["contribution"].to_numpy()).all()
    for column in ("weight", "news"):
        values = result.decompositions[column].dropna().to_numpy()
        assert np.isfinite(values).all()
    assert not result.decompositions.duplicated(_DECOMPOSITION_KEY_COLUMNS).any()

    level_rows = result.decompositions.query("decomposition == 'level'")
    assert not level_rows.empty
    level_totals = level_rows.groupby(
        [
            "source",
            "vintage_date",
            "date",
            "forecast_horizon",
            "variable",
            "forecast_metric",
        ]
    )["contribution"].sum()
    forecast_values = result.data.forecasts.set_index(
        ["source", "vintage_date", "date", "forecast_horizon", "variable", "metric"]
    )["value"]
    forecast_values.index = forecast_values.index.set_names(level_totals.index.names)
    np.testing.assert_allclose(
        level_totals.to_numpy(),
        forecast_values.reindex(level_totals.index).to_numpy(),
        atol=1e-8,
    )

    revision_rows = result.decompositions.query("decomposition == 'revision'")
    assert not revision_rows.empty
    assert set(revision_rows["revision_source"]) == {
        "news",
        "reestimation",
        "interaction",
    }
    revision_source_counts = revision_rows.groupby(
        [
            "source",
            "vintage_date",
            "base_vintage_date",
            "date",
            "forecast_horizon",
            "variable",
            "component",
        ],
        observed=True,
    )["revision_source"].nunique()
    assert (revision_source_counts == 3).all()
    revision_totals = revision_rows.groupby(
        [
            "source",
            "vintage_date",
            "base_vintage_date",
            "date",
            "forecast_horizon",
            "variable",
            "component",
        ]
    )["contribution"].sum()
    level_contributions = level_rows.groupby(
        [
            "source",
            "vintage_date",
            "date",
            "variable",
            "component",
        ]
    )["contribution"].sum()
    current_index = pd.MultiIndex.from_arrays(
        [
            revision_totals.index.get_level_values("source"),
            revision_totals.index.get_level_values("vintage_date"),
            revision_totals.index.get_level_values("date"),
            revision_totals.index.get_level_values("variable"),
            revision_totals.index.get_level_values("component"),
        ],
        names=level_contributions.index.names,
    )
    current = level_contributions.reindex(current_index).fillna(0).to_numpy()
    previous_index = pd.MultiIndex.from_arrays(
        [
            revision_totals.index.get_level_values("source"),
            revision_totals.index.get_level_values("base_vintage_date"),
            revision_totals.index.get_level_values("date"),
            revision_totals.index.get_level_values("variable"),
            revision_totals.index.get_level_values("component"),
        ],
        names=level_contributions.index.names,
    )
    previous = level_contributions.reindex(previous_index).fillna(0).to_numpy()
    np.testing.assert_allclose(revision_totals.to_numpy(), current - previous, atol=1e-8)
    assert snapshot_decompositions(result.decompositions) == snapshot
