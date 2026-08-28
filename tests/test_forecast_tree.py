"""Tests for TreeNode and ForecastTree.

Covers TreeNode validation/traversal and ForecastTree fit/forecast for both
kinds of node ``transform``: a plain callable (a fixed rule that reduces its
children to the node's ``target`` column) and a ``ForecastModel`` (a stacking
regression fitted on its children's raw components).
"""

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.forecast_tree import ForecastTree, TreeNode


class StubForecastModel(ForecastModel):
    """Minimal concrete ForecastModel used only to build TreeNode trees."""

    def __init__(self, label):
        super().__init__(label=label)

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return pd.DataFrame()


class RecordingStubForecastModel(ForecastModel):
    """Concrete ForecastModel leaf that records each ``_fit`` call.

    Used to check that every leaf in a (possibly nested) ``TreeNode`` tree is
    actually fitted, and with what lag/dummy configuration.
    """

    def __init__(self, label, formula=None):
        super().__init__(label=label, formula=formula)
        self.fit_calls = []

    def _fit(self, y, X=None, **kwargs):
        self.recorded_kwargs = kwargs
        self.fit_calls.append(
            dict(
                y=y,
                X=X,
                y_lags=self.y_lags,
                X_lags=self.X_lags,
                dummies=self.dummies,
                kwargs=kwargs,
            )
        )
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return pd.DataFrame()


class RecordingTransformModel(ForecastModel):
    """ForecastModel transform that records fit/forecast inputs."""

    def __init__(self, label, data_transformation=None):
        super().__init__(label=label, data_transformation=data_transformation)
        self.fit_calls = []
        self.forecast_calls = []

    def _fit(self, y, X=None, **kwargs):
        self.fit_calls.append({"y": y.copy(), "X": X.copy(), "kwargs": dict(kwargs)})
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        self.forecast_calls.append(
            {
                "steps": steps,
                "X": X.copy() if isinstance(X, pd.DataFrame) else X,
                "y": y.copy() if isinstance(y, pd.DataFrame) else y,
                "kwargs": dict(kwargs),
            }
        )
        if y is not None:
            return self._wrap_forecast(
                y.iloc[-1:, :].to_numpy().repeat(steps, axis=0), steps
            )
        return self._wrap_forecast(np.zeros((steps, self.y.shape[1])), steps)


class CommitRecordingModel(RecordingTransformModel):
    """Model that records how often its fitted state is committed."""

    commit_counts = {}
    tracked_ids = set()

    def _commit_fit(self, candidate):
        if id(self) in self.tracked_ids:
            self.commit_counts[self.label] = self.commit_counts.get(self.label, 0) + 1
        super()._commit_fit(candidate)


class FitKwargsSpyModel(ForecastModel):
    """ForecastModel transform recording the exact kwargs its *public*
    ``fit()``/``forecast()`` receive, before delegating to the base class.

    Unlike recording ``_fit``/``_forecast``, this captures named parameters
    (``data_transformation``/``frequency``/``X_imputation``/
    ``drop_transformation_nans``) too, since the base ``ForecastModel``
    consumes those as explicit parameters and never forwards them into
    ``_fit``/``_forecast``'s own ``**kwargs``.
    """

    def __init__(self, label, data_transformation=None):
        super().__init__(label=label, data_transformation=data_transformation)
        self.fit_calls = []
        self.fit_kwargs_calls = []
        self.forecast_kwargs_calls = []

    def fit(self, y, X=None, **kwargs):
        self.fit_kwargs_calls.append(dict(kwargs))
        return super().fit(y, X=X, **kwargs)

    def forecast(self, steps=1, X=None, y=None, **kwargs):
        self.forecast_kwargs_calls.append(dict(kwargs))
        return super().forecast(steps=steps, X=X, y=y, **kwargs)

    def _fit(self, y, X=None, **kwargs):
        self.fit_calls.append({"y": y.copy(), "X": X.copy(), "kwargs": dict(kwargs)})
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return self._wrap_forecast(np.zeros((steps, self.y.shape[1])), steps)


class RecordingPipelineLeaf(ForecastModel):
    """Leaf that records the (already-transformed) y/X it receives."""

    def __init__(self, label, data_transformation=None):
        super().__init__(label=label, data_transformation=data_transformation)
        self.fit_calls = []
        self.forecast_calls = []

    def _fit(self, y, X=None, **kwargs):
        self.fit_calls.append({"y": y.copy(), "X": X.copy() if X is not None else None})
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        self.forecast_calls.append(
            {
                "steps": steps,
                "X": X.copy() if isinstance(X, pd.DataFrame) else X,
                "y": y.copy() if isinstance(y, pd.DataFrame) else y,
            }
        )
        tail = (
            y.iloc[-steps:]
            if y is not None
            else pd.DataFrame(np.zeros((steps, self.y.shape[1])), columns=self.y.columns)
        )
        return self._wrap_forecast(tail.to_numpy(), steps)


class NonMissingRecordingPipelineLeaf(RecordingPipelineLeaf):
    """Pipeline leaf that uses only complete transformed fit observations."""

    _handles_missing_values = False


class ConstantForecastModel(ForecastModel):
    """Leaf model whose forecast is a constant value at every horizon.

    Fitting is a no-op; forecasting returns ``value`` for every column of the
    fitted ``y`` at each step. This makes ForecastTree's combination
    arithmetic checkable by hand.
    """

    def __init__(self, label, value):
        super().__init__(label=label)
        self.value = value

    def _fit(self, y, X=None, **kwargs):
        self.fitted_values_ = pd.DataFrame(
            self.value, index=y.index, columns=y.columns
        ).astype(float)
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return self._wrap_forecast(np.full((steps, self.y.shape[1]), self.value), steps)


class MutatingFitModel(ForecastModel):
    """Tree component whose failed fit mutates state before raising."""

    def __init__(self, label, fail=False):
        super().__init__(label=label)
        self.fail = fail
        self.fit_attempts = 0

    def _fit(self, y, X=None, **kwargs):
        self.fit_attempts += 1
        self.fitted_values_ = y.copy()
        if self.fail:
            self.marker = "failed candidate"
            raise RuntimeError("fit failed")
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return self._wrap_forecast(
            np.full((steps, self.y.shape[1]), self.y.iloc[-1, 0]), steps
        )


def weighted_average(weights):
    """Return a callable transform computing a fixed-weight average.

    ``weights`` maps each child name to its weight; the returned function
    combines the per-child single-column forecast DataFrames accordingly.
    """

    def _transform(components):
        combined = None
        for name, w in weights.items():
            contribution = components[name] * w
            combined = contribution if combined is None else combined + contribution
        return combined

    return _transform


def _first_value(components):
    """Dummy transform: returns the first component unchanged."""
    return next(iter(components.values()))


def _make_y(columns, n=6):
    index = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame({col: range(1, n + 1) for col in columns}, index=index).astype(
        float
    )


def _make_X(n=6):
    index = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame({"x1": range(10, 10 + n)}, index=index).astype(float)


def _fit_and_forecast_pickled_tree(tree, y, X):
    tree.fit(y, X=X, frequency="M")
    return tree.forecast(steps=2, X=X, frequency="M")


# ---------------------------------------------------------------------- #
# TreeNode: construction, validation and traversal.                      #
# ---------------------------------------------------------------------- #
def test_treenode_child_names_uses_label():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_b")
    inner = TreeNode(transform=_first_value, children=[leaf_a], name="inner")
    outer = TreeNode(transform=_first_value, children=[leaf_b, inner], name="outer")

    assert outer.child_names == ["model_b", "inner"]


def test_treenode_all_leaves_collects_unique_models():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_b")
    leaf_c = StubForecastModel("model_c")
    inner = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="inner")
    outer = TreeNode(transform=_first_value, children=[leaf_c, inner], name="outer")

    leaves = outer.all_leaves()

    assert leaves == [leaf_c, leaf_a, leaf_b]


def test_treenode_nodes_post_order():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_b")
    inner = TreeNode(transform=_first_value, children=[leaf_a], name="inner")
    outer = TreeNode(transform=_first_value, children=[leaf_b, inner], name="outer")

    nodes = outer.nodes()

    assert nodes == [inner, outer]


def test_treenode_nested_structure():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_b")
    leaf_c = StubForecastModel("model_c")
    inner = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="inner")
    outer = TreeNode(transform=_first_value, children=[leaf_c, inner], name="outer")

    assert outer.children == [leaf_c, inner]
    assert outer.all_leaves() == [leaf_c, leaf_a, leaf_b]
    assert outer.nodes() == [inner, outer]


def test_treenode_rejects_invalid_transform():
    leaf = StubForecastModel("model_a")

    with pytest.raises(TypeError):
        TreeNode(transform="not_callable", children=[leaf], name="bad")


def test_treenode_rejects_empty_children():
    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=[], name="bad")


def test_treenode_rejects_bad_child_type():
    leaf = StubForecastModel("model_a")

    with pytest.raises(TypeError):
        TreeNode(transform=_first_value, children=[leaf, 42], name="bad")


def test_treenode_duplicate_names_raises():
    leaf_a = StubForecastModel("dup")
    leaf_b = StubForecastModel("dup")

    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")


def test_treenode_duplicate_names_across_nested_nodes_raises():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_a")
    inner = TreeNode(transform=_first_value, children=[leaf_a], name="inner")

    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=[leaf_b, inner], name="outer")


def test_treenode_cycle_detection_raises():
    leaf = StubForecastModel("model_a")
    inner = TreeNode(transform=_first_value, children=[leaf], name="inner")
    outer = TreeNode(transform=_first_value, children=[inner], name="outer")

    # Manually construct a cycle: inner -> outer -> inner -> ...
    inner.children.append(outer)

    with pytest.raises(ValueError):
        outer.all_leaves()

    with pytest.raises(ValueError):
        outer.nodes()


def test_treenode_shared_leaf_reused_across_subtrees():
    shared_leaf = StubForecastModel("shared")
    leaf_b = StubForecastModel("model_b")
    leaf_c = StubForecastModel("model_c")

    branch_1 = TreeNode(
        transform=_first_value, children=[shared_leaf, leaf_b], name="branch_1"
    )
    branch_2 = TreeNode(
        transform=_first_value, children=[shared_leaf, leaf_c], name="branch_2"
    )
    outer = TreeNode(transform=_first_value, children=[branch_1, branch_2], name="outer")

    assert outer.all_leaves() == [shared_leaf, leaf_b, leaf_c]
    assert branch_1.child_names == ["shared", "model_b"]
    assert branch_2.child_names == ["shared", "model_c"]
    assert outer.nodes() == [branch_1, branch_2, outer]


def test_treenode_shared_subtree_reused_under_two_parents():
    leaf_a = StubForecastModel("model_a")
    shared_inner = TreeNode(
        transform=_first_value, children=[leaf_a], name="shared_inner"
    )
    leaf_b = StubForecastModel("model_b")
    leaf_c = StubForecastModel("model_c")

    branch_1 = TreeNode(
        transform=_first_value, children=[shared_inner, leaf_b], name="branch_1"
    )
    branch_2 = TreeNode(
        transform=_first_value, children=[shared_inner, leaf_c], name="branch_2"
    )
    outer = TreeNode(transform=_first_value, children=[branch_1, branch_2], name="outer")

    nodes = outer.nodes()

    assert nodes == [shared_inner, branch_1, branch_2, outer]
    assert nodes.count(shared_inner) == 1


def test_treenode_identical_object_twice_as_direct_siblings_raises():
    leaf = StubForecastModel("model_a")

    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=[leaf, leaf], name="outer")


def test_treenode_distinct_leaf_and_node_same_name_raises():
    leaf_a = StubForecastModel("model_a")
    inner = TreeNode(transform=_first_value, children=[leaf_a], name="clash")
    leaf_clash = StubForecastModel("clash")

    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=[leaf_clash, inner], name="outer")


def test_treenode_default_name_is_node():
    leaf = StubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf])

    assert spec.name == "node"


def test_treenode_rejects_tuple_children():
    leaf_a = StubForecastModel("model_a")
    leaf_b = StubForecastModel("model_b")

    with pytest.raises(ValueError):
        TreeNode(transform=_first_value, children=(leaf_a, leaf_b), name="bad")


def test_treenode_transform_not_invoked_at_construction():
    leaf = StubForecastModel("model_a")

    def _exploding_transform(components):
        raise AssertionError("transform must not be invoked by TreeNode construction")

    spec = TreeNode(transform=_exploding_transform, children=[leaf], name="outer")

    assert spec.transform is _exploding_transform


def test_treenode_target_on_forecastmodel_transform_raises():
    leaf = StubForecastModel("model_a")
    stacker = StubForecastModel("stacker")

    with pytest.raises(ValueError):
        TreeNode(transform=stacker, children=[leaf], name="outer", target="model_a")


# ---------------------------------------------------------------------- #
# ForecastTree: construction and fit().                                 #
# ---------------------------------------------------------------------- #
def test_forecasttree_rejects_non_treenode():
    with pytest.raises(TypeError):
        ForecastTree(spec="not_a_treenode")


def test_forecasttree_flat_tree_fits_every_leaf():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    spec = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X)

    for leaf in tree.spec.all_leaves():
        assert len(leaf.fit_calls) == 1


def test_forecasttree_nested_tree_fits_all_leaves():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    leaf_c = RecordingStubForecastModel("model_c")
    inner = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="inner")
    outer = TreeNode(transform=_first_value, children=[leaf_c, inner], name="outer")
    tree = ForecastTree(spec=outer)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X)

    fitted_leaves = tree.spec.all_leaves()
    assert len(fitted_leaves) == 3
    for leaf in fitted_leaves:
        assert len(leaf.fit_calls) == 1


def test_forecasttree_commits_shared_components_once():
    CommitRecordingModel.commit_counts = {}
    shared_leaf = CommitRecordingModel("shared")
    leaf_b = CommitRecordingModel("model_b")
    leaf_c = CommitRecordingModel("model_c")
    branch_1_transform = CommitRecordingModel("branch_1_transform")
    branch_2_transform = CommitRecordingModel("branch_2_transform")
    branch_1 = TreeNode(
        transform=branch_1_transform,
        children=[shared_leaf, leaf_b],
        name="branch_1",
    )
    branch_2 = TreeNode(
        transform=branch_2_transform,
        children=[shared_leaf, leaf_c],
        name="branch_2",
    )
    spec = TreeNode(transform=_first_value, children=[branch_1, branch_2], name="root")
    tree = ForecastTree(spec=spec)
    CommitRecordingModel.tracked_ids = {
        id(component)
        for component in (
            shared_leaf,
            leaf_b,
            leaf_c,
            branch_1_transform,
            branch_2_transform,
        )
    }

    tree.fit(_make_y(["target"]), X=_make_X())

    assert CommitRecordingModel.commit_counts == {
        "shared": 1,
        "model_b": 1,
        "model_c": 1,
        "branch_1_transform": 1,
        "branch_2_transform": 1,
    }


def test_forecasttree_forwards_lags_and_dummies_uniformly():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    spec = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X, y_lags=2, X_lags=1, dummies=["2020-03-31"])

    for leaf in tree.spec.all_leaves():
        call = leaf.fit_calls[0]
        assert call["y_lags"] == 2
        assert call["X_lags"] == 1
        assert call["dummies"] == ["2020-03-31"]


def test_forecasttree_infers_target_from_single_column_y():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=_make_X())

    assert tree.y_name == "target"
    assert list(tree.y.columns) == ["target"]
    assert tree._n_output_cols == 1


def test_forecasttree_explicit_target_with_multivariate_y():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer", target="b")
    tree = ForecastTree(spec=spec)

    y = _make_y(["a", "b"])
    tree.fit(y, X=_make_X())

    leaf_fitted = tree.spec.all_leaves()[0]
    recorded_y = leaf_fitted.fit_calls[0]["y"]
    assert list(recorded_y.columns) == ["a", "b"]

    assert list(tree.y.columns) == ["b"]


def test_forecasttree_ambiguous_target_raises():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["a", "b"])

    with pytest.raises(ValueError):
        tree.fit(y, X=_make_X())


def test_forecasttree_target_not_in_columns_raises():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(
        transform=_first_value, children=[leaf], name="outer", target="missing"
    )
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])

    with pytest.raises(ValueError):
        tree.fit(y, X=_make_X())


def test_forecasttree_fit_fits_leaves_in_place():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    spec = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=_make_X())

    assert len(leaf_a.fit_calls) == 1
    assert len(leaf_b.fit_calls) == 1
    assert tree.spec.all_leaves() == [leaf_a, leaf_b]


def test_forecasttree_fit_returns_self():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    result = tree.fit(y, X=_make_X())

    assert result is tree


# ---------------------------------------------------------------------- #
# ForecastTree: forecast() with callable transforms.                    #
# ---------------------------------------------------------------------- #
def test_forecasttree_flat_forecast_applies_transform():
    leaf_a = ConstantForecastModel("model_a", value=10.0)
    leaf_b = ConstantForecastModel("model_b", value=20.0)
    spec = TreeNode(
        transform=weighted_average({"model_a": 0.25, "model_b": 0.75}),
        children=[leaf_a, leaf_b],
        name="node",
    )
    tree = ForecastTree(spec=spec)
    tree.fit(_make_y(["target"]), X=_make_X())

    fc = tree.forecast(steps=3)

    assert list(fc.columns) == ["target"]
    assert len(fc) == 3
    assert isinstance(fc.index, pd.DatetimeIndex)
    # 0.25 * 10 + 0.75 * 20 = 17.5
    assert (fc["target"] == 17.5).all()


def test_forecasttree_two_stage_nested_forecast():
    leaf_a = ConstantForecastModel("model_a", value=10.0)
    leaf_b = ConstantForecastModel("model_b", value=20.0)
    leaf_c = ConstantForecastModel("model_c", value=30.0)

    stage1 = TreeNode(
        transform=weighted_average({"model_a": 0.5, "model_b": 0.5}),
        children=[leaf_a, leaf_b],
        name="stage1",
    )
    final = TreeNode(
        transform=weighted_average({"stage1": 0.5, "model_c": 0.5}),
        children=[stage1, leaf_c],
        name="final",
    )
    tree = ForecastTree(spec=final)
    tree.fit(_make_y(["target"]), X=_make_X())

    fc = tree.forecast(steps=2)

    # stage1 = 0.5*10 + 0.5*20 = 15; final = 0.5*15 + 0.5*30 = 22.5
    assert (fc["target"] == 22.5).all()
    # Intermediate node forecasts are exposed for inspection.
    assert (tree.node_forecasts_["stage1"]["target"] == 15.0).all()
    assert (tree.node_forecasts_["final"]["target"] == 22.5).all()


def test_forecasttree_forecast_exposes_leaf_forecasts():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    spec = TreeNode(
        transform=weighted_average({"model_a": 0.5, "model_b": 0.5}),
        children=[leaf_a, leaf_b],
        name="node",
    )
    tree = ForecastTree(spec=spec)
    tree.fit(_make_y(["target"]), X=_make_X())

    tree.forecast(steps=2)

    assert set(tree.leaf_forecasts_) == {"model_a", "model_b"}
    assert (tree.leaf_forecasts_["model_a"]["target"] == 1.0).all()
    assert (tree.leaf_forecasts_["model_b"]["target"] == 2.0).all()


def test_forecasttree_transform_receives_leaf_name_keys():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    seen = {}

    def recording_transform(components):
        seen["keys"] = set(components)
        return next(iter(components.values()))

    spec = TreeNode(transform=recording_transform, children=[leaf_a, leaf_b], name="node")
    tree = ForecastTree(spec=spec)
    tree.fit(_make_y(["target"]), X=_make_X())

    tree.forecast(steps=1)

    assert seen["keys"] == {"model_a", "model_b"}


def test_forecasttree_nested_transform_receives_node_and_leaf_names():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    leaf_c = ConstantForecastModel("model_c", value=3.0)
    seen = {}

    def recording_outer(components):
        seen["outer_keys"] = set(components)
        return sum(components.values()) / len(components)

    inner = TreeNode(
        transform=weighted_average({"model_a": 0.5, "model_b": 0.5}),
        children=[leaf_a, leaf_b],
        name="inner",
    )
    outer = TreeNode(transform=recording_outer, children=[inner, leaf_c], name="outer")
    tree = ForecastTree(spec=outer)
    tree.fit(_make_y(["target"]), X=_make_X())

    tree.forecast(steps=1)

    assert seen["outer_keys"] == {"inner", "model_c"}


def test_forecasttree_forecast_selects_target_from_multivariate_leaf():
    leaf = ConstantForecastModel("model_a", value=5.0)
    spec = TreeNode(
        transform=weighted_average({"model_a": 1.0}),
        children=[leaf],
        name="node",
        target="cpi",
    )
    tree = ForecastTree(spec=spec)
    tree.fit(_make_y(["gdp", "cpi"]), X=_make_X())

    fc = tree.forecast(steps=2)

    assert list(fc.columns) == ["cpi"]
    assert list(fc.index) == list(pd.date_range("2020-07-31", periods=2, freq="ME"))
    assert (fc["cpi"] == 5.0).all()


def test_forecasttree_two_stage_with_ols_matches_manual():
    from forecast_realtime.models.ols import ForecastOLS

    idx = pd.date_range("2020-03-31", periods=12, freq="QE")
    rng = np.random.default_rng(0)
    reg1 = rng.normal(size=12)
    reg2 = rng.normal(size=12)
    gdp = 1.0 + 0.5 * reg1 - 0.3 * reg2 + rng.normal(scale=0.1, size=12)
    y_fit = pd.DataFrame({"gdp": gdp[:10]}, index=idx[:10])
    X_full = pd.DataFrame({"reg1": reg1, "reg2": reg2}, index=idx)

    ols1 = ForecastOLS(label="ols1", formula="gdp ~ reg1")
    ols2 = ForecastOLS(label="ols2", formula="gdp ~ reg2")
    ols3 = ForecastOLS(label="ols3", formula="gdp ~ reg1 + reg2")

    stage1 = TreeNode(
        transform=weighted_average({"ols1": 0.6, "ols2": 0.4}),
        children=[ols1, ols2],
        name="stage1",
    )
    final = TreeNode(
        transform=weighted_average({"stage1": 0.5, "ols3": 0.5}),
        children=[stage1, ols3],
        name="final",
    )
    tree = ForecastTree(spec=final)
    tree.fit(y_fit, X=X_full)

    tree_fc = tree.forecast(steps=2, X=X_full)

    fc1 = tree.leaf_forecasts_["ols1"]["gdp"]
    fc2 = tree.leaf_forecasts_["ols2"]["gdp"]
    fc3 = tree.leaf_forecasts_["ols3"]["gdp"]
    manual_stage1 = 0.6 * fc1 + 0.4 * fc2
    manual_final = 0.5 * manual_stage1 + 0.5 * fc3

    assert tree_fc.shape == (2, 1)
    pd.testing.assert_series_equal(
        tree.node_forecasts_["stage1"]["gdp"], manual_stage1, check_names=False
    )
    pd.testing.assert_series_equal(tree_fc["gdp"], manual_final, check_names=False)


# ---------------------------------------------------------------------- #
# ForecastTree: ForecastModel (stacking) transforms.                    #
# ---------------------------------------------------------------------- #
def test_forecasttree_model_transform_stacks_children():
    from forecast_realtime.models.ols import ForecastOLS

    idx = pd.date_range("2020-03-31", periods=12, freq="QE")
    rng = np.random.default_rng(1)
    reg1 = rng.normal(size=12)
    reg2 = rng.normal(size=12)
    gdp = 1.0 + 0.5 * reg1 - 0.3 * reg2 + rng.normal(scale=0.1, size=12)
    y_fit = pd.DataFrame({"gdp": gdp[:10]}, index=idx[:10])
    X_full = pd.DataFrame({"reg1": reg1, "reg2": reg2}, index=idx)

    ols1 = ForecastOLS(label="ols1", formula="gdp ~ reg1")
    ols2 = ForecastOLS(label="ols2", formula="gdp ~ reg2")
    # The stacker's formula references the children by their labels; its design
    # matrix is the children's raw forecasts, one column per child.
    stacker = ForecastOLS(label="stacker", formula="gdp ~ ols1 + ols2")

    spec = TreeNode(transform=stacker, children=[ols1, ols2], name="stack")
    tree = ForecastTree(spec=spec)
    tree.fit(y_fit, X=X_full)

    fc = tree.forecast(steps=2, X=X_full)

    assert fc.shape == (2, 1)
    assert list(fc.columns) == ["gdp"]
    assert tree.y_name == "gdp"
    # The stacking transform's own forecast is the root node's output.
    pd.testing.assert_frame_equal(tree.node_forecasts_["stack"], fc.forecast)


def test_forecasttree_model_transform_needs_no_target():
    # A ForecastModel transform picks its own target via its formula, so the
    # tree needs no target even when y has several columns.
    from forecast_realtime.models.ols import ForecastOLS

    idx = pd.date_range("2020-03-31", periods=12, freq="QE")
    rng = np.random.default_rng(2)
    reg1 = rng.normal(size=12)
    gdp = 1.0 + 0.5 * reg1 + rng.normal(scale=0.1, size=12)
    cpi = rng.normal(size=12)
    y_fit = pd.DataFrame({"gdp": gdp[:10], "cpi": cpi[:10]}, index=idx[:10])
    X_full = pd.DataFrame({"reg1": reg1}, index=idx)

    ols1 = ForecastOLS(label="ols1", formula="gdp ~ reg1")
    stacker = ForecastOLS(label="stacker", formula="gdp ~ ols1")

    spec = TreeNode(transform=stacker, children=[ols1], name="stack")
    tree = ForecastTree(spec=spec)
    tree.fit(y_fit, X=X_full)

    fc = tree.forecast(steps=2, X=X_full)

    assert list(fc.columns) == ["gdp"]
    assert list(tree.y.columns) == ["gdp"]


# ---------------------------------------------------------------------- #
# ForecastTree: kwarg forwarding, refit and ForecastModel contract.     #
# ---------------------------------------------------------------------- #
def test_forecasttree_forwards_extra_kwargs_to_every_leaf():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    spec = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(
        y,
        X=X,
        y_lags=2,
        X_lags=1,
        dummies=["2020-03-31"],
        some_custom_kwarg=123,
    )

    for leaf in tree.spec.all_leaves():
        call = leaf.fit_calls[0]
        assert call["y_lags"] == 2
        assert call["X_lags"] == 1
        assert call["dummies"] == ["2020-03-31"]
        assert leaf.recorded_kwargs["some_custom_kwarg"] == 123


def test_forecasttree_fit_with_x_none():
    leaf_a = RecordingStubForecastModel("model_a")
    leaf_b = RecordingStubForecastModel("model_b")
    spec = TreeNode(transform=_first_value, children=[leaf_a, leaf_b], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=None)

    for leaf in tree.spec.all_leaves():
        assert len(leaf.fit_calls) == 1
        assert leaf.fit_calls[0]["X"] is None

    assert list(tree.y.columns) == ["target"]


def test_forecasttree_per_leaf_formula_selects_from_multivariate_data():
    leaf = RecordingStubForecastModel("model_a", formula="gdp ~ x1")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer", target="gdp")
    tree = ForecastTree(spec=spec)

    y = _make_y(["gdp", "cpi"])
    X = _make_X()
    X["x2"] = X["x1"] + 1
    tree.fit(y, X=X)

    fitted_leaf = tree.spec.all_leaves()[0]
    recorded = fitted_leaf.fit_calls[0]
    assert list(recorded["y"].columns) == ["gdp"]
    assert list(recorded["X"].columns) == ["x1"]


def test_forecasttree_refit_refits_leaves():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    result_first = tree.fit(y, X=_make_X())
    result_second = tree.fit(y, X=_make_X())

    assert result_first is tree
    assert result_second is tree
    for leaf_fitted in tree.spec.all_leaves():
        assert len(leaf_fitted.fit_calls) == 2


def test_forecasttree_target_state_after_fit_non_first_target():
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer", target="cpi")
    tree = ForecastTree(spec=spec)

    y = _make_y(["gdp", "cpi"])
    tree.fit(y, X=_make_X())

    assert tree.y_name == "cpi"
    assert list(tree.y.columns) == ["cpi"]
    assert tree._n_output_cols == 1


def test_forecasttree_sets_last_y_fit_date():
    # RealTimeModel reads last_y_fit_date to anchor X imputation, so the
    # tree must expose it just like the base ForecastModel.fit does.
    leaf = RecordingStubForecastModel("model_a")
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=_make_X())

    assert tree.last_y_fit_date == y.index[-1]


def test_forecasttree_publishes_root_fitted_state_and_metadata():
    leaf = ConstantForecastModel("leaf", value=1.0)
    spec = TreeNode(
        transform=_first_value,
        children=[leaf],
        name="root",
        target="target",
    )
    tree = ForecastTree(spec=spec)

    tree.fit(_make_y(["target"]), X=_make_X())

    configuration = tree._fitted_model_configuration
    assert tree._is_fitted is True
    assert configuration.y_columns == ("target",)
    assert configuration.X_columns is None
    assert tree.native_metric_mapping() == {"target": "levels"}
    assert configuration.forecast_origin == tree.last_y_fit_date


def test_failed_initial_forecasttree_fit_remains_unfitted():
    leaf = MutatingFitModel("leaf", fail=True)
    tree = ForecastTree(TreeNode(transform=_first_value, children=[leaf], name="root"))

    with pytest.raises(RuntimeError, match="fit failed"):
        tree.fit(_make_y(["target"]))

    assert tree._is_fitted is False
    assert not hasattr(tree, "_fitted_model_configuration")


def test_forecasttree_callable_root_uses_last_usable_fitted_output_origin():
    leaf = NonMissingRecordingPipelineLeaf(
        "leaf", data_transformation={"target": "diff", "x1": "levels"}
    )
    spec = TreeNode(transform=_first_value, children=[leaf], name="outer")
    tree = ForecastTree(spec=spec)

    y_index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, np.nan]}, index=y_index)
    X = pd.DataFrame({"x1": [10.0, 11.0, 12.0, 13.0]}, index=y_index)
    tree.fit(y, X=X, X_lags=1, frequency="M")

    future_index = pd.date_range("2020-05-31", periods=1, freq="ME")
    future_X = pd.DataFrame({"x1": [14.0]}, index=future_index)
    forecast = tree.forecast(steps=1, X=future_X, X_lags=1, frequency="M")

    effective_origin = pd.Timestamp("2020-03-31")
    assert tree.last_y_fit_date == effective_origin
    assert forecast.index.equals(pd.DatetimeIndex([pd.Timestamp("2020-04-30")]))
    assert leaf.forecast_calls[0]["X"].loc[pd.Timestamp("2020-04-30"), "x1"] == 13.0


def test_forecasttree_model_root_uses_root_transform_origin():
    leaf = NonMissingRecordingPipelineLeaf(
        "leaf", data_transformation={"target": "diff", "x1": "levels"}
    )
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf], name="stack")
    tree = ForecastTree(spec=spec)

    y_index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, np.nan]}, index=y_index)
    tree.fit(y, X=_make_X(n=4), frequency="M")

    forecast = tree.forecast(steps=1, frequency="M")

    assert tree.last_y_fit_date == transform.last_y_fit_date
    assert tree.last_y_fit_date == pd.Timestamp("2020-03-31")
    assert forecast.index.equals(pd.DatetimeIndex([pd.Timestamp("2020-04-30")]))


def test_forecasttree_callable_root_exposes_in_sample_fitted_values_after_fit():
    leaf_a = ConstantForecastModel("model_a", value=10.0)
    leaf_b = ConstantForecastModel("model_b", value=20.0)
    spec = TreeNode(
        transform=weighted_average({"model_a": 0.25, "model_b": 0.75}),
        children=[leaf_a, leaf_b],
        name="root",
        target="target",
    )
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=_make_X())

    expected = pd.DataFrame(17.5, index=y.index, columns=["target"])
    assert isinstance(tree.fitted_values, pd.DataFrame)
    pd.testing.assert_frame_equal(tree.fitted_values, expected)


def test_forecasttree_supports_nested_tree_child_without_fitted_values_failure():
    inner_leaf_a = ConstantForecastModel("inner_a", value=3.0)
    inner_leaf_b = ConstantForecastModel("inner_b", value=5.0)
    inner_spec = TreeNode(
        transform=weighted_average({"inner_a": 0.5, "inner_b": 0.5}),
        children=[inner_leaf_a, inner_leaf_b],
        name="inner_root",
        target="target",
    )
    child_tree = ForecastTree(spec=inner_spec, label="child_tree")

    outer_leaf = ConstantForecastModel("outer_leaf", value=9.0)
    outer_spec = TreeNode(
        transform=weighted_average({"child_tree": 0.25, "outer_leaf": 0.75}),
        children=[child_tree, outer_leaf],
        name="outer_root",
        target="target",
    )
    outer_tree = ForecastTree(spec=outer_spec)

    y = _make_y(["target"])
    outer_tree.fit(y, X=_make_X())

    expected_value = 0.25 * 4.0 + 0.75 * 9.0
    expected = pd.DataFrame(expected_value, index=y.index, columns=["target"])
    pd.testing.assert_frame_equal(outer_tree.fitted_values, expected)


def test_forecasttree_model_transform_fit_receives_custom_kwargs_not_structural_controls():  # noqa: E501
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(
        y,
        X=_make_X(),
        y_lags=2,
        X_lags=1,
        dummies=["2020-03-31"],
        custom_alpha=0.7,
    )

    assert len(transform.fit_calls) == 1
    kwargs = transform.fit_calls[0]["kwargs"]
    assert kwargs["custom_alpha"] == 0.7
    assert "y_lags" not in kwargs
    assert "X_lags" not in kwargs
    assert "dummies" not in kwargs


def test_forecasttree_model_transform_forecast_receives_y_and_custom_kwargs():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X)

    tree.forecast(steps=2, X=X, y=y, custom_beta=123)

    assert len(transform.forecast_calls) == 1
    call = transform.forecast_calls[0]
    pd.testing.assert_frame_equal(call["y"], y)
    assert call["kwargs"]["custom_beta"] == 123


def test_forecasttree_forwards_x_only_conditioning_with_lags_and_fitted_origin():
    class OriginRecordingTransformModel(RecordingTransformModel):
        def _forecast(self, steps=1, X=None, y=None, **kwargs):
            self.forecast_origin = kwargs.get("forecast_origin")
            return super()._forecast(steps=steps, X=X, y=y, **kwargs)

    leaf = OriginRecordingTransformModel("leaf")
    stacker = RecordingTransformModel("stacker")
    spec = TreeNode(transform=stacker, children=[leaf], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X, X_lags=1)

    X_conditioning = pd.DataFrame(
        {"x1": [16.0, 17.0]},
        index=pd.date_range("2020-07-31", periods=2, freq="ME"),
    )
    tree.forecast(steps=2, X=X_conditioning, X_lags=1)

    assert leaf.forecast_origin == y.index[-1]
    leaf_X = leaf.forecast_calls[0]["X"]
    np.testing.assert_allclose(leaf_X["x1"].to_numpy()[-2:], [16.0, 17.0])
    np.testing.assert_allclose(leaf_X["x1_lag1"].to_numpy()[-2:], [15.0, 16.0])


def test_forecasttree_model_root_exposes_in_sample_fitted_values_after_fit():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    tree.fit(y, X=_make_X())

    assert isinstance(tree.fitted_values, pd.DataFrame)
    pd.testing.assert_frame_equal(tree.fitted_values, transform.fitted_values)


def test_failed_forecasttree_refit_preserves_leaf_and_tree_state():
    leaf = MutatingFitModel("leaf")
    spec = TreeNode(transform=_first_value, children=[leaf], name="root")
    tree = ForecastTree(spec=spec)
    y = _make_y(["target"])

    tree.fit(y)
    previous_forecast = tree.forecast()
    previous_y = tree.y.copy()
    previous_fit_date = tree.last_y_fit_date
    leaf.fail = True

    with pytest.raises(RuntimeError, match="fit failed"):
        tree.fit(y.assign(target=y["target"] + 10.0))

    pd.testing.assert_frame_equal(tree.forecast(), previous_forecast)
    pd.testing.assert_frame_equal(tree.y, previous_y)
    assert tree.last_y_fit_date == previous_fit_date
    assert leaf.fit_attempts == 1
    assert not hasattr(leaf, "marker")


def test_forecasttree_aggregates_component_capability_flags():
    leaf = RecordingStubForecastModel("leaf")
    leaf._needs_ragged_edge_imputation = False
    leaf._handles_missing_values = False
    transform = RecordingTransformModel("stacker")
    transform._needs_ragged_edge_imputation = True
    transform._handles_missing_values = True
    spec = TreeNode(transform=transform, children=[leaf], name="stack")
    tree = ForecastTree(spec=spec)

    assert tree._needs_ragged_edge_imputation is True
    assert tree._handles_missing_values is False


def test_forecasttree_keeps_scalar_y_name_for_multiple_outputs():
    leaf = ConstantForecastModel("leaf", value=1.0)
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf], name="stack")
    tree = ForecastTree(spec=spec)

    tree.fit(_make_y(["a", "b"]), X=_make_X())

    assert tree.y_name == "a"
    assert list(tree.y.columns) == ["a", "b"]


def test_forecasttree_model_transform_forecast_filters_structural_controls():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = RecordingTransformModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(y, X=X)

    tree.forecast(
        steps=2,
        X=X,
        y=y,
        y_lags=3,
        X_lags=4,
        dummies=["2020-04-30"],
        custom_beta=123,
    )

    assert len(transform.forecast_calls) == 1
    call = transform.forecast_calls[0]
    pd.testing.assert_frame_equal(call["y"], y)
    kwargs = call["kwargs"]
    assert kwargs["custom_beta"] == 123
    assert "y_lags" not in kwargs
    assert "X_lags" not in kwargs
    assert "dummies" not in kwargs


# ---------------------------------------------------------------------- #
# ForecastTree: data_transformation configuration/precedence.  #
# ---------------------------------------------------------------------- #
def _make_levels_y(values, freq="ME"):
    index = pd.date_range("2020-01-31", periods=len(values), freq=freq)
    return pd.DataFrame({"target": values}, index=index)


def test_forecasttree_leaves_with_different_pipelines_receive_raw_levels():
    """Two leaves owning different pipelines both receive the same raw
    levels and transform them independently (levels vs diff)."""
    y = _make_levels_y([100.0, 110.0, 121.0, 133.1])

    levels_leaf = RecordingPipelineLeaf(
        "levels_leaf", data_transformation={"target": "levels"}
    )
    diff_leaf = RecordingPipelineLeaf("diff_leaf", data_transformation={"target": "diff"})
    spec = TreeNode(
        transform=_first_value, children=[levels_leaf, diff_leaf], name="root"
    )
    tree = ForecastTree(spec=spec)

    tree.fit(y, frequency="M", data_transformation={"target": "levels"})

    levels_call = levels_leaf.fit_calls[0]
    diff_call = diff_leaf.fit_calls[0]
    np.testing.assert_allclose(
        levels_call["y"]["target"].to_numpy(), [100.0, 110.0, 121.0, 133.1]
    )
    np.testing.assert_allclose(diff_call["y"]["target"].to_numpy(), [10.0, 11.0, 12.1])


def test_forecasttree_owned_transformation_is_fallback_for_leaves_without_one():
    """A tree-owned transformation is the fallback for a leaf with none of its
    own; a leaf's own transformation still wins over both tree and call-level."""
    y = _make_levels_y([100.0, 110.0, 121.0, 133.1])

    plain_leaf = RecordingPipelineLeaf("plain_leaf")
    own_leaf = RecordingPipelineLeaf("own_leaf", data_transformation={"target": "levels"})
    spec = TreeNode(transform=_first_value, children=[plain_leaf, own_leaf], name="root")
    tree = ForecastTree(spec=spec, data_transformation={"target": "diff"})

    tree.fit(y, frequency="M", data_transformation={"target": "levels"})

    plain_call = plain_leaf.fit_calls[0]
    own_call = own_leaf.fit_calls[0]
    np.testing.assert_allclose(plain_call["y"]["target"].to_numpy(), [10.0, 11.0, 12.1])
    np.testing.assert_allclose(
        own_call["y"]["target"].to_numpy(), [100.0, 110.0, 121.0, 133.1]
    )


def test_forecasttree_call_level_data_transformation_is_last_resort():
    """The call-level ``data_transformation`` fallback is used only when
    neither the tree nor the leaf owns a pipeline."""
    y = _make_levels_y([100.0, 110.0, 121.0, 133.1])

    plain_leaf = RecordingPipelineLeaf("plain_leaf")
    spec = TreeNode(transform=_first_value, children=[plain_leaf], name="root")
    tree = ForecastTree(spec=spec)

    tree.fit(y, frequency="M", data_transformation={"target": "diff"})

    call = plain_leaf.fit_calls[0]
    np.testing.assert_allclose(call["y"]["target"].to_numpy(), [10.0, 11.0, 12.1])


def test_forecasttree_preserves_native_target_metric_for_model_transform():
    index = pd.date_range("2020-03-31", periods=4, freq="QE")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]}, index=index)
    leaf = RecordingPipelineLeaf("leaf", data_transformation={"target": "pop"})
    transform = RecordingTransformModel(
        "stacker", data_transformation={"target": "pop", "leaf": "pop"}
    )
    tree = ForecastTree(TreeNode(transform=transform, children=[leaf], name="stack"))

    tree.fit(
        y,
        frequency="Q",
        data_transformation={"target": "pop"},
        y_input_metrics={"target": "pop"},
    )

    pd.testing.assert_frame_equal(transform.fit_calls[0]["y"], y)


def test_forecasttree_nested_tree_precedence_is_nearest_tree_owned_then_leaf_owned():
    """A nested tree's own pipeline is the fallback for its own children
    (recursively), regardless of what the outer tree forwards; a leaf-owned
    pipeline still wins over its nearest owning tree."""
    y = _make_levels_y([100.0, 110.0, 121.0, 133.1])

    inner_plain_leaf = RecordingPipelineLeaf("inner_plain")
    inner_own_leaf = RecordingPipelineLeaf(
        "inner_own", data_transformation={"target": "levels"}
    )
    inner_spec = TreeNode(
        transform=_first_value,
        children=[inner_plain_leaf, inner_own_leaf],
        name="inner_root",
    )
    inner_tree = ForecastTree(
        spec=inner_spec,
        label="inner_tree",
        data_transformation={"target": "diff"},
    )

    outer_plain_leaf = RecordingPipelineLeaf("outer_plain")
    outer_spec = TreeNode(
        transform=_first_value, children=[inner_tree, outer_plain_leaf], name="outer_root"
    )
    outer_tree = ForecastTree(spec=outer_spec)

    outer_tree.fit(y, frequency="M", data_transformation={"target": "levels"})

    # outer_tree owns no pipeline: the call-level "levels" is its own fallback.
    np.testing.assert_allclose(
        outer_plain_leaf.fit_calls[0]["y"]["target"].to_numpy(),
        [100.0, 110.0, 121.0, 133.1],
    )
    # inner_tree's own "diff" pipeline wins over what outer_tree forwarded,
    # and becomes ITS OWN fallback for inner_plain (nearest tree-owned).
    np.testing.assert_allclose(
        inner_plain_leaf.fit_calls[0]["y"]["target"].to_numpy(), [10.0, 11.0, 12.1]
    )
    # inner_own's own pipeline wins over inner_tree's.
    np.testing.assert_allclose(
        inner_own_leaf.fit_calls[0]["y"]["target"].to_numpy(),
        [100.0, 110.0, 121.0, 133.1],
    )


def test_forecasttree_leaf_forecast_transforms_history_and_conditioning_once():
    """ForecastTree forwards raw history/conditioning; a leaf combines and
    transforms them exactly once inside its own forecast()."""
    y_hist = _make_levels_y([100.0, 110.0, 121.0, 133.1])
    future_index = pd.date_range("2020-05-31", periods=2, freq="ME")
    y_cond = pd.DataFrame({"target": [140.0, 150.0]}, index=future_index)

    diff_leaf = RecordingPipelineLeaf("diff_leaf", data_transformation={"target": "diff"})
    spec = TreeNode(transform=_first_value, children=[diff_leaf], name="root")
    tree = ForecastTree(spec=spec)

    tree.fit(y_hist, frequency="M", data_transformation={"target": "levels"})
    tree.forecast(
        steps=2, y=y_cond, frequency="M", data_transformation={"target": "levels"}
    )

    forecast_y = diff_leaf.forecast_calls[0]["y"]
    expected_diff_tail = [140.0 - 133.1, 150.0 - 140.0]
    np.testing.assert_allclose(
        forecast_y.iloc[-2:]["target"].to_numpy(), expected_diff_tail
    )


def test_forecasttree_node_transform_without_own_pipeline_ignores_raw_input_context():
    """A node transform with no pipeline of its own never sees the original
    ``data_transformation``/``X_imputation``/frequency
    context, since its X is synthetic child-output columns."""
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = FitKwargsSpyModel("stacker")
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(
        y,
        X=X,
        data_transformation={"target": "diff", "x1": "levels"},
        frequency="M",
        X_imputation="last",
        drop_transformation_nans=False,
    )

    fit_kwargs = transform.fit_kwargs_calls[0]
    assert fit_kwargs["data_transformation"] is None
    assert "X_frequencies" not in fit_kwargs
    assert "X_imputation" not in fit_kwargs
    assert "frequency" not in fit_kwargs
    assert "drop_transformation_nans" not in fit_kwargs

    tree.forecast(
        steps=2,
        X=X,
        y=y,
        data_transformation={"target": "diff", "x1": "levels"},
        frequency="M",
        X_imputation="last",
    )
    forecast_kwargs = transform.forecast_kwargs_calls[0]
    assert forecast_kwargs["data_transformation"] is None
    assert "X_imputation" not in forecast_kwargs
    assert "frequency" not in forecast_kwargs


def test_forecasttree_node_transform_with_own_pipeline_keeps_frequency():
    """A node transform that owns its own pipeline resolves it normally
    (``data_transformation`` still forced to ``None``, letting the node's
    own setting take over) and keeps the ``frequency`` it needs to apply it."""
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = FitKwargsSpyModel(
        "stacker",
        data_transformation={
            "target": "levels",
            "model_a": "levels",
            "model_b": "levels",
        },
    )
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    y = _make_y(["target"])
    X = _make_X()
    tree.fit(
        y,
        X=X,
        data_transformation={"target": "levels", "x1": "levels"},
        frequency="M",
        X_imputation="last",
    )

    fit_kwargs = transform.fit_kwargs_calls[0]
    assert fit_kwargs["data_transformation"] is None
    assert fit_kwargs["frequency"] == "M"
    assert "X_frequencies" not in fit_kwargs
    assert "X_imputation" not in fit_kwargs


def test_forecasttree_node_pipeline_transforms_synthetic_child_columns():
    leaf_a = ConstantForecastModel("model_a", value=1.0)
    leaf_b = ConstantForecastModel("model_b", value=2.0)
    transform = FitKwargsSpyModel(
        "stacker",
        data_transformation={
            "target": "levels",
            "model_a": "diff",
            "model_b": "diff",
        },
    )
    spec = TreeNode(transform=transform, children=[leaf_a, leaf_b], name="stack")
    tree = ForecastTree(spec=spec)

    tree.fit(
        _make_y(["target"]),
        X=_make_X(),
        frequency="M",
        drop_transformation_nans=False,
    )

    node_fit = transform.fit_calls[0]
    assert list(node_fit["X"].columns) == ["model_a", "model_b"]
    assert node_fit["X"]["model_a"].isna().iloc[0]
    np.testing.assert_allclose(node_fit["X"]["model_a"].iloc[1:], 0.0)
    assert node_fit["X"]["model_b"].isna().iloc[0]
    np.testing.assert_allclose(node_fit["X"]["model_b"].iloc[1:], 0.0)


def test_forecasttree_node_pipeline_missing_synthetic_mapping_names_node():
    transform = FitKwargsSpyModel(
        "stacker",
        data_transformation={
            "target": "levels",
            "model_a": "levels",
        },
    )
    spec = TreeNode(
        transform=transform,
        children=[
            ConstantForecastModel("model_a", value=1.0),
            ConstantForecastModel("model_b", value=2.0),
        ],
        name="stack",
    )

    with pytest.raises(ValueError, match="node 'stack'.*model_b"):
        ForecastTree(spec=spec).fit(_make_y(["target"]), X=_make_X(), frequency="M")


def test_forecasttree_node_pipeline_missing_y_mapping_names_node():
    transform = FitKwargsSpyModel(
        "stacker",
        data_transformation={
            "model_a": "levels",
            "model_b": "levels",
        },
    )
    spec = TreeNode(
        transform=transform,
        children=[
            ConstantForecastModel("model_a", value=1.0),
            ConstantForecastModel("model_b", value=2.0),
        ],
        name="stack",
    )

    with pytest.raises(ValueError, match="node 'stack'.*target"):
        ForecastTree(spec=spec).fit(_make_y(["target"]), X=_make_X(), frequency="M")


def test_forecasttree_node_transformation_takes_precedence_over_tree_and_call_mapping():
    transform = FitKwargsSpyModel(
        "stacker",
        data_transformation={
            "target": "levels",
            "model_a": "diff",
            "model_b": "diff",
        },
    )
    spec = TreeNode(
        transform=transform,
        children=[
            ConstantForecastModel("model_a", value=1.0),
            ConstantForecastModel("model_b", value=2.0),
        ],
        name="stack",
    )
    tree = ForecastTree(
        spec=spec,
        data_transformation={"target": "diff", "x1": "levels"},
    )

    tree.fit(
        _make_y(["target"]),
        X=_make_X(),
        data_transformation={"target": "diff", "x1": "diff"},
        frequency="M",
        drop_transformation_nans=False,
    )

    node_fit = transform.fit_calls[0]
    assert node_fit["y"]["target"].iloc[0] == 1.0
    assert not node_fit["X"]["model_a"].isna().any()
    np.testing.assert_allclose(node_fit["X"]["model_a"], 1.0)


def test_forecasttree_nested_node_pipelines_are_isolated():
    inner_transform = FitKwargsSpyModel(
        "inner_stacker",
        data_transformation={
            "target": "levels",
            "inner_a": "diff",
            "inner_b": "levels",
        },
    )
    inner_spec = TreeNode(
        transform=inner_transform,
        children=[
            ConstantForecastModel("inner_a", value=1.0),
            ConstantForecastModel("inner_b", value=2.0),
        ],
        name="inner_node",
    )
    inner_tree = ForecastTree(spec=inner_spec, label="inner_tree")
    outer_transform = FitKwargsSpyModel(
        "outer_stacker",
        data_transformation={
            "target": "levels",
            "inner_tree": "diff",
            "outer_leaf": "levels",
        },
    )
    outer_spec = TreeNode(
        transform=outer_transform,
        children=[inner_tree, ConstantForecastModel("outer_leaf", value=3.0)],
        name="outer_node",
    )

    ForecastTree(spec=outer_spec).fit(
        _make_y(["target"]),
        X=_make_X(),
        frequency="M",
        drop_transformation_nans=False,
    )

    inner_X = inner_transform.fit_calls[0]["X"]
    outer_X = outer_transform.fit_calls[0]["X"]
    assert list(inner_X.columns) == ["inner_a", "inner_b"]
    assert list(outer_X.columns) == ["inner_tree", "outer_leaf"]
    assert inner_X["inner_a"].isna().iloc[0]
    assert outer_X["inner_tree"].isna().iloc[0]


def test_forecasttree_node_pipeline_pickle_and_parallel_equivalence():
    spec = TreeNode(
        transform=FitKwargsSpyModel(
            "stacker",
            data_transformation={
                "target": "levels",
                "model_a": "diff",
                "model_b": "levels",
            },
        ),
        children=[
            ConstantForecastModel("model_a", value=1.0),
            ConstantForecastModel("model_b", value=2.0),
        ],
        name="stack",
    )
    y = _make_y(["target"])
    X = _make_X()

    sequential = ForecastTree(spec=spec)
    sequential_forecast = _fit_and_forecast_pickled_tree(
        pickle.loads(pickle.dumps(sequential)), y, X
    )
    with ProcessPoolExecutor(max_workers=1) as executor:
        parallel_forecast = executor.submit(
            _fit_and_forecast_pickled_tree, sequential, y, X
        ).result()

    pd.testing.assert_frame_equal(sequential_forecast, parallel_forecast)


def test_forecasttree_callable_root_defaults_to_levels_metric():
    leaf = ConstantForecastModel("leaf", value=1.0)
    tree = ForecastTree(
        TreeNode(
            transform=_first_value,
            children=[leaf],
            name="root",
            target="target",
        ),
        data_transformation={"target": "diff"},
    )

    tree.fit(_make_levels_y([100.0, 110.0, 121.0]), frequency="M")

    assert tree.native_metric_mapping() == {"target": "levels"}


def test_forecasttree_model_root_uses_fitted_root_transformation():
    leaf = ConstantForecastModel("leaf", value=1.0)
    transform = RecordingTransformModel(
        "stacker", data_transformation={"target": "diff", "leaf": "levels"}
    )
    tree = ForecastTree(TreeNode(transform=transform, children=[leaf], name="root"))

    tree.fit(_make_levels_y([100.0, 110.0, 121.0]), frequency="M")

    assert tree.native_metric_mapping() == {"target": "diff"}
