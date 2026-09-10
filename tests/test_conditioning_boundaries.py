"""Explicit target constraints stay separate from observations and tree children."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast_realtime import ForecastContext, ForecastModel, ForecastTree, TreeNode
from forecast_realtime._model_data import ModelData
from forecast_realtime.models import ForecastOLS, RandomForest


class RecordingModel(ForecastModel):
    """Record prepared inputs without claiming to enforce target constraints."""

    def _fit(self, y, X=None, **kwargs):
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        self.received_y = y
        self.received_X = X
        return np.zeros((steps, len(self.y.columns)))


class SupportingModel(RecordingModel):
    """Echo explicit or published future targets on the requested calendar."""

    _supports_target_conditioning = True

    def _forecast(self, steps, X=None, y=None, **kwargs):
        super()._forecast(steps, X, y, **kwargs)
        dates = self._conditioning_dates(self._raw_data, kwargs["forecast_origin"], steps)
        return (
            pd.DataFrame(0.0, index=dates, columns=self.y.columns)
            if y is None
            else y.reindex(dates).fillna(0.0)
        )


@pytest.fixture
def history():
    return pd.DataFrame(
        {"target": np.arange(12, dtype=float) + 10},
        index=pd.date_range("2020-01-31", periods=12, freq="ME"),
    )


def _future(history, values=(80.0, 90.0, 100.0)):
    return pd.DataFrame(
        {"target": values},
        index=pd.date_range(
            history.index[-1] + pd.offsets.MonthEnd(), periods=len(values), freq="ME"
        ),
    )


@pytest.mark.parametrize(
    "boundary", ["forecast", "predict", "internal-predict", "internal-forecast"]
)
def test_raw_constraints_rejected_before_overrides_and_transformation(history, boundary):
    class OverrideModel(RecordingModel):
        def predict(self, context, **kwargs):
            self.override_reached = True
            return super().predict(context, **kwargs)

        def forecast(self, *args, **kwargs):
            self.forecast_reached = True
            return super().forecast(*args, **kwargs)

    model = OverrideModel(data_transformation={"target": "logs"}).fit(history)
    # A negative value would become NaN; validate the raw request first.
    constraint = _future(history, (-1.0,))
    context = ForecastContext(
        history, None, constraint, forecast_origin=history.index[-1]
    )
    with pytest.raises(ValueError, match="does not support target conditioning.*target"):
        if boundary == "forecast":
            model.forecast(y=constraint)
        elif boundary == "predict":
            model.predict(context)
        else:
            method = (
                model._predict_from_data
                if boundary == "internal-predict"
                else model._forecast_from_data
            )
            method(
                ModelData.from_context(context, model._raw_data),
                forecast_origin=history.index[-1],
            )
    if boundary.startswith("internal"):
        assert not hasattr(model, "override_reached")
        assert not hasattr(model, "forecast_reached")


@pytest.mark.parametrize("path", ["history", "empty", "nan", "outside", "published"])
def test_unsupported_model_accepts_non_constraints(history, path):
    model = RecordingModel().fit(history)
    future = _future(history)
    explicit = {
        "history": history,
        "empty": future.iloc[:0],
        "nan": future * np.nan,
        "outside": future.iloc[2:],
        "published": None,
    }[path]
    context = ForecastContext(
        history,
        None,
        explicit,
        forecast_origin=history.index[-1],
        y_published=future if path == "published" else None,
    )
    assert len(model.predict(context, steps=2)) == 2
    if path == "published":
        pd.testing.assert_frame_equal(model.received_y.loc[future.index], future)


def test_origin_inclusive_validation_uses_first_output_date(history):
    class NowcastModel(RecordingModel):
        _forecast_dates_include_origin = True

    model = NowcastModel().fit(history)
    with pytest.raises(ValueError, match="does not support target conditioning"):
        model.forecast(y=history.tail(1))
    # For one nowcast step, the next period is outside the requested dates.
    assert len(model.forecast(y=_future(history, (80.0,)))) == 1


@pytest.mark.parametrize("factory", [ForecastOLS, RandomForest])
@pytest.mark.parametrize("strategy", ["direct", "recursive"])
@pytest.mark.parametrize("lags", [0, 1])
def test_regressions_reject_y_but_keep_X_forecasts(history, factory, strategy, lags):
    X = history.rename(columns={"target": "driver"})
    model = factory(forecast_strategy=strategy, steps=3).fit(history, X, y_lags=lags)
    future = _future(history)
    with pytest.raises(ValueError, match="does not support target conditioning"):
        model.forecast(steps=3, y=future, X=future.rename(columns={"target": "driver"}))
    assert (
        len(model.forecast(steps=3, X=future.rename(columns={"target": "driver"}))) == 3
    )


def test_multilevel_tree_routes_constraints_only_to_root(history):
    leaf = RecordingModel(label="leaf")
    middle = RecordingModel(label="middle-model")
    root = SupportingModel(label="root-model")
    tree = ForecastTree(
        TreeNode(root, [TreeNode(middle, [leaf], name="middle")], name="root")
    )
    X = history.rename(columns={"target": "driver"})
    tree.fit(history, X)
    explicit = _future(history)
    published = explicit * 2
    context = ForecastContext(
        history,
        X,
        explicit,
        X.rename(columns={"driver": "driver"}),
        history.index[-1],
        y_published=published,
    )
    result = tree.predict(context, steps=3)
    np.testing.assert_allclose(result, explicit)
    for child in (leaf, middle):
        pd.testing.assert_frame_equal(child.received_y.loc[published.index], published)
    assert list(root.received_X.columns) == ["middle"]
    assert list(leaf.received_X.columns) == ["driver"]


@pytest.mark.parametrize("root_kind", ["unsupported", "callable", "nested"])
def test_supporting_child_does_not_enable_unsupported_tree_routing(history, root_kind):
    leaf = SupportingModel(label="leaf")
    if root_kind == "callable":

        def root(outputs):
            return outputs["leaf"]

    elif root_kind == "nested":
        root = ForecastTree(
            TreeNode(
                SupportingModel(label="inner"),
                [RecordingModel(label="inner-leaf")],
                name="inner-root",
            )
        )
    else:
        root = RecordingModel(label="root-model")
    tree = ForecastTree(TreeNode(root, [leaf], name="root"))
    with pytest.raises(
        ValueError, match="target conditioning|nested-tree target routing"
    ):
        tree._validate_target_conditioning({"target"})


@pytest.mark.parametrize("position", ["leaf", "middle", "root"])
@pytest.mark.parametrize("policy", [{}, {"X": {"driver": {"source": "A", "periods": 1}}}])
def test_tree_rejects_every_contained_archive_policy(position, policy):
    models = {
        name: SupportingModel(
            label=name, conditioning=policy if name == position else None
        )
        for name in ("leaf", "middle", "root")
    }
    with pytest.raises(ValueError, match=f"contained model '{position}'"):
        ForecastTree(
            TreeNode(
                models["root"],
                [TreeNode(models["middle"], [models["leaf"]], name="middle-node")],
                name="root-node",
            )
        )


def test_context_round_trip_preserves_units_and_prepared_values(history):
    published = _future(history, (23.1,))
    explicit = _future(history).iloc[1:].assign(target=5.0)
    context = ForecastContext(
        history,
        None,
        explicit,
        forecast_origin=history.index[-1],
        y_conditioning_input_metrics={"target": "pop"},
        y_published=published,
        y_published_input_metrics={"target": "levels"},
    )
    model = SupportingModel(data_transformation={"target": "pop"}).fit(history)
    original = ModelData.from_context(context, model._raw_data)
    round_trip = ModelData.from_context(
        ForecastContext._from_data(original, context.forecast_origin), model._raw_data
    )
    for kind in ("conditioning", "published"):
        assert original.metrics("y", kind) == round_trip.metrics("y", kind)
        pd.testing.assert_frame_equal(
            original.to_wide("y", kind), round_trip.to_wide("y", kind)
        )
    pd.testing.assert_frame_equal(
        original.transform({"target": "pop"}, combine=True).to_wide("y"),
        round_trip.transform({"target": "pop"}, combine=True).to_wide("y"),
    )
    changed = replace(context, y_conditioning=explicit.assign(target=7.0))
    np.testing.assert_allclose(model.predict(changed, steps=3).iloc[1:], 7.0)
