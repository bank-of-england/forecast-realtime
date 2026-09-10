"""Tests for model-owned and run-level conditioning policy contracts."""

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from forecast_realtime.external_model import ExternalModel
from forecast_realtime.forecast_model import (
    ForecastModel,
    _ConditioningPolicy,
    _parse_conditioning,
)
from forecast_realtime.real_time_model import (
    _resolve_conditioning,
    _validate_conditioning,
)


class _ConditioningModel(ForecastModel):
    """Small model double for policy resolution and capability checks."""

    def _fit(self, y, X=None, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


class _ExternalConditioningModel(ExternalModel):
    """External-model double used to inspect constructor parameter routing."""

    def _fit_command(self):
        return []

    def _forecast_command(self, steps):
        return []


def _requirements(model, y_variables, X_variables=None):
    """Return the input requirements for a model's selected formula."""
    return model.input_requirements(y_variables, X_variables)


def _fallback(**overrides):
    """Build a complete legacy conditioning fallback for resolver tests."""
    fallback = {
        "y_sources": None,
        "y_steps_ahead": None,
        "X_sources": None,
        "X_steps_ahead": None,
    }
    fallback.update(overrides)
    return fallback


def test_conditioning_policy_is_copied_frozen_and_pickleable():
    """A policy cannot observe later caller mutations and survives pickling."""
    supplied = {
        "y": {"target": {"source": "nowcast", "periods": 2}},
        "X": {"driver": {"source": "futures", "periods": 3}},
    }

    model = _ConditioningModel(label="captured", conditioning=supplied)
    supplied["y"]["target"]["periods"] = 99
    supplied["X"].clear()

    expected = _ConditioningPolicy(
        y=(model.conditioning.y[0],),
        X=(model.conditioning.X[0],),
    )
    assert model.conditioning == expected
    assert pickle.loads(pickle.dumps(model)).conditioning == expected

    with pytest.raises(FrozenInstanceError):
        model.conditioning.y = ()
    with pytest.raises(FrozenInstanceError):
        model.conditioning.y[0].periods = 4


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        ({"Z": {}}, ValueError, "only 'y' and 'X'"),
        ({"y": []}, TypeError, "y must be a variable mapping"),
        (
            {"y": {"target": {"source": "nowcast"}}},
            ValueError,
            "requires exactly source and periods",
        ),
        (
            {"y": {"target": {"source": "nowcast", "periods": 1, "extra": 2}}},
            ValueError,
            "requires exactly source and periods",
        ),
        (
            {"y": {"target": {"source": True, "periods": 1}}},
            TypeError,
            "source must be a non-empty string",
        ),
        (
            {"y": {"target": {"source": "nowcast", "periods": True}}},
            ValueError,
            "periods must be a positive integer",
        ),
        (
            {"y": {"target": {"source": "nowcast", "periods": 0}}},
            ValueError,
            "periods must be a positive integer",
        ),
        (
            {"y": {"target": {"source": "nowcast", "periods": 1.5}}},
            ValueError,
            "periods must be a positive integer",
        ),
        (
            {"y": {1: {"source": "nowcast", "periods": 1}}},
            TypeError,
            "non-empty string name",
        ),
    ],
)
def test_conditioning_policy_rejects_invalid_schemas(value, exception, message):
    """Malformed policies fail at model construction with useful diagnostics."""
    with pytest.raises(exception, match=message):
        _parse_conditioning(value, "invalid")


def test_inherited_fallback_projects_disjoint_formula_inputs():
    """An inherited fallback is projected onto each formula's raw inputs."""
    fallback = _fallback(
        y_sources={"gdp": "gdp-source", "inflation": "inflation-source"},
        y_steps_ahead={"gdp": 0, "inflation": 2},
        X_sources={"oil": "oil-source", "gas": "gas-source"},
        X_steps_ahead={"oil": 1, "gas": 3},
    )
    sources = {
        "gdp-source",
        "inflation-source",
        "oil-source",
        "gas-source",
    }
    gdp = _ConditioningModel(label="gdp", formula="gdp ~ oil")
    inflation = _ConditioningModel(label="inflation", formula="inflation ~ gas")
    gdp._supports_target_conditioning = True
    inflation._supports_target_conditioning = True

    gdp_resolved = _resolve_conditioning(
        gdp,
        _requirements(gdp, ["gdp", "inflation"], ["oil", "gas"]),
        ["gdp", "inflation"],
        fallback,
        sources,
        steps=4,
    )
    inflation_resolved = _resolve_conditioning(
        inflation,
        _requirements(inflation, ["gdp", "inflation"], ["oil", "gas"]),
        ["gdp", "inflation"],
        fallback,
        sources,
        steps=4,
    )

    assert gdp_resolved == {
        "y_sources": {"gdp": "gdp-source"},
        "y_steps_ahead": {"gdp": 0},
        "X_sources": {"oil": "oil-source"},
        "X_steps_ahead": {"oil": 1},
    }
    assert inflation_resolved == {
        "y_sources": {"inflation": "inflation-source"},
        "y_steps_ahead": {"inflation": 2},
        "X_sources": {"gas": "gas-source"},
        "X_steps_ahead": {"gas": 3},
    }


def test_inherited_projection_preserves_none_and_explicit_empty_mappings():
    """Projection does not collapse omitted and explicitly empty legacy roles."""
    model = _ConditioningModel(label="projection", formula="target ~ driver")
    resolved = _resolve_conditioning(
        model,
        _requirements(model, ["target"], ["driver"]),
        ["target"],
        _fallback(X_sources={}, X_steps_ahead={}),
        set(),
        steps=2,
    )

    assert resolved["y_sources"] is None
    assert resolved["y_steps_ahead"] is None
    assert resolved["X_sources"] == {}
    assert resolved["X_steps_ahead"] == {}


def test_model_owned_X_policy_replaces_the_whole_fallback():
    """An X-only policy does not inherit y settings from the run fallback."""
    model = _ConditioningModel(
        label="owned-X",
        formula="target ~ driver",
        conditioning={"X": {"driver": {"source": "owned", "periods": 2}}},
    )
    resolved = _resolve_conditioning(
        model,
        _requirements(model, ["target"], ["driver"]),
        ["target"],
        _fallback(
            y_sources={"target": "legacy-y"},
            y_steps_ahead={"target": 0},
            X_sources={"driver": "legacy-X"},
            X_steps_ahead={"driver": 0},
        ),
        {"legacy-y", "legacy-X", "owned"},
        steps=3,
    )

    assert resolved == {
        "y_sources": None,
        "y_steps_ahead": None,
        "X_sources": {"driver": "owned"},
        "X_steps_ahead": {"driver": 1},
    }


def test_empty_model_policy_disables_both_fallback_roles():
    """An explicit empty policy disables externally supplied conditioning."""
    model = _ConditioningModel(
        label="disabled",
        formula="target ~ driver",
        conditioning={},
    )
    resolved = _resolve_conditioning(
        model,
        _requirements(model, ["target"], ["driver"]),
        ["target"],
        _fallback(
            y_sources={"target": "legacy-y"},
            y_steps_ahead={"target": 0},
            X_sources={"driver": "legacy-X"},
            X_steps_ahead={"driver": 1},
        ),
        {"legacy-y", "legacy-X"},
        steps=3,
    )

    assert resolved == {
        "y_sources": None,
        "y_steps_ahead": None,
        "X_sources": None,
        "X_steps_ahead": None,
    }


@pytest.mark.parametrize(
    ("conditioning", "steps", "sources", "message"),
    [
        (
            {"y": {"target": {"source": "source", "periods": 3}}},
            2,
            {"source"},
            "owned.*target.*must not exceed steps=2",
        ),
        (
            {"y": {"target": {"source": "missing", "periods": 1}}},
            2,
            {"source"},
            "owned.*target.*unknown.*'missing'",
        ),
        (
            {"y": {"other": {"source": "source", "periods": 1}}},
            2,
            {"source"},
            "owned.*other.*not a selected raw input",
        ),
    ],
)
def test_model_owned_policy_reports_run_duration_source_and_variable_errors(
    conditioning, steps, sources, message
):
    """Effective model-owned entries are checked against this run and model."""
    model = _ConditioningModel(
        label="owned",
        formula="target ~ driver",
        conditioning=conditioning,
    )
    with pytest.raises(ValueError, match=message):
        _resolve_conditioning(
            model,
            _requirements(model, ["target"], ["driver"]),
            ["target"],
            _fallback(),
            sources,
            steps,
        )


def test_legacy_conditioning_validation_rejects_unknown_variables_and_booleans():
    """The legacy run schema keeps its inclusive horizon and rejects booleans."""
    with pytest.raises(ValueError, match="Extra keys.*other"):
        _validate_conditioning(
            "y", ["target"], {"other": 0}, {"other": "source"}, steps=2
        )
    with pytest.raises(ValueError, match="0..1"):
        _validate_conditioning(
            "y", ["target"], {"target": True}, {"target": "source"}, steps=2
        )


@pytest.mark.parametrize(
    ("supports", "horizon", "raises"),
    [(True, 0, False), (False, 0, True), (False, None, False)],
)
def test_target_capability_allows_supporting_models_and_rejects_active_y(
    supports, horizon, raises
):
    """Only an active legacy y constraint requires declared model support."""
    model = _ConditioningModel(label="capability")
    model._supports_target_conditioning = supports
    requirements = _requirements(model, ["target"])

    if raises:
        with pytest.raises(
            ValueError,
            match="capability.*does not support target conditioning.*target",
        ):
            _resolve_conditioning(
                model,
                requirements,
                ["target"],
                _fallback(
                    y_sources={"target": "source"},
                    y_steps_ahead={"target": horizon},
                ),
                {"source"},
                steps=2,
            )
    else:
        resolved = _resolve_conditioning(
            model,
            requirements,
            ["target"],
            _fallback(
                y_sources={"target": "source"},
                y_steps_ahead={"target": horizon},
            ),
            {"source"},
            steps=2,
        )
        assert resolved["y_sources"] == {"target": "source"}
        assert resolved["y_steps_ahead"] == {"target": horizon}


def test_conditioning_constructor_argument_is_not_external_estimator_parameter():
    """External-model parameters exclude the framework-owned conditioning policy."""
    policy = {"X": {"driver": {"source": "futures", "periods": 1}}}
    model = _ExternalConditioningModel(
        "dummy-script",
        conditioning=policy,
        estimator_option=3,
    )
    try:
        assert model.conditioning.X[0].source == "futures"
        assert model.params == {"estimator_option": 3}
        assert "conditioning" not in model.params
    finally:
        model._tmpdir.cleanup()
