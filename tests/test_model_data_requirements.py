"""Tests for model input requirements and archive selection contracts."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from forecast_realtime._model_data import ModelData, ModelInputRequirements
from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.forecast_tree import ForecastTree, TreeNode


class _RequirementsModel(ForecastModel):
    """Concrete model stub used only to inspect input requirements."""

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return None


def _identity_component(components):
    """Return the first component without fitting or forecasting it."""
    return next(iter(components.values()))


def test_input_requirements_without_mapping_defaults_y_and_X_to_levels():
    model = _RequirementsModel(label="plain", formula="target ~ driver")

    assert model.input_requirements(["target"], ["driver"]) == (
        ModelInputRequirements(
            consumer="plain",
            y=(("target", "levels"),),
            X=(("driver", "levels"),),
        ),
    )


def test_input_requirements_preserves_formula_declaration_order():
    model = _RequirementsModel(
        label="ordered",
        formula="target_b + target_a ~ driver_b + driver_a",
    )

    requirements = model.input_requirements(
        ["target_a", "target_b"], ["driver_a", "driver_b"]
    )

    assert requirements[0].y == (("target_b", "levels"), ("target_a", "levels"))
    assert requirements[0].X == (("driver_b", "levels"), ("driver_a", "levels"))


def test_input_requirements_prefers_model_pipeline_over_fallback():
    model = _RequirementsModel(
        label="owned",
        data_transformation={"target": "diff", "driver": "pop"},
    )

    requirements = model.input_requirements(
        ["target"],
        ["driver"],
        {"target": "logs", "driver": "levels"},
    )

    assert requirements[0] == ModelInputRequirements(
        consumer="owned",
        y=(("target", "diff"),),
        X=(("driver", "pop"),),
        explicit=True,
    )


def test_forecasttree_preserves_distinct_leaf_metrics_and_roles():
    diff_leaf = _RequirementsModel(
        label="diff",
        data_transformation={"target": "diff", "driver": "diff"},
    )
    pop_leaf = _RequirementsModel(
        label="pop",
        data_transformation={"target": "pop", "driver": "pop"},
    )
    tree = ForecastTree(
        TreeNode(
            transform=_identity_component,
            children=[diff_leaf, pop_leaf],
            name="root",
        )
    )

    assert tree.input_requirements(["target"], ["driver"]) == (
        ModelInputRequirements(
            consumer="diff",
            y=(("target", "diff"),),
            X=(("driver", "diff"),),
            explicit=True,
        ),
        ModelInputRequirements(
            consumer="pop",
            y=(("target", "pop"),),
            X=(("driver", "pop"),),
            explicit=True,
        ),
    )


def test_forecasttree_mapping_free_leaf_still_requests_X():
    leaf = _RequirementsModel(label="plain")
    tree = ForecastTree(
        TreeNode(transform=_identity_component, children=[leaf], name="root")
    )

    requirements = tree.input_requirements(["target"], ["driver"])

    assert requirements[0].X == (("driver", "levels"),)
    assert requirements[0].explicit is False


def test_nested_forecasttree_uses_nearest_tree_owned_mapping():
    leaf = _RequirementsModel(label="leaf")
    inner_tree = ForecastTree(
        TreeNode(transform=_identity_component, children=[leaf], name="inner"),
        label="inner-tree",
        data_transformation={"target": "diff"},
    )
    outer_tree = ForecastTree(
        TreeNode(transform=_identity_component, children=[inner_tree], name="outer"),
        label="outer-tree",
        data_transformation={"target": "pop"},
    )

    assert outer_tree.input_requirements(["target"]) == (
        ModelInputRequirements(
            consumer="leaf",
            y=(("target", "diff"),),
            explicit=True,
        ),
    )


def test_stacker_owns_raw_target_and_prefixed_component_X():
    component = _RequirementsModel(
        label="component",
        formula="target_b + target_a ~ driver",
    )
    stacker = _RequirementsModel(
        label="stacker",
        formula="target_a ~ component_target_b + component_target_a",
        data_transformation={
            "target_a": "logs",
            "component_target_b": "diff",
            "component_target_a": "pop",
        },
    )
    tree = ForecastTree(
        TreeNode(transform=stacker, children=[component], name="stack"),
        label="tree",
    )

    requirements = tree.input_requirements(["target_a", "target_b"], ["driver"])

    assert requirements == (
        ModelInputRequirements(
            consumer="component",
            y=(("target_b", "levels"), ("target_a", "levels")),
            X=(("driver", "levels"),),
        ),
        ModelInputRequirements(
            consumer="tree/stack/stacker",
            y=(("target_a", "logs"),),
            X=(("component_target_b", "diff"), ("component_target_a", "pop")),
            explicit=True,
            X_kind="component",
        ),
    )
    assert requirements[1].items("X") == ()


def test_stacker_uses_native_for_implicit_synthetic_X():
    component = _RequirementsModel(
        label="component",
        formula="target_b + target_a ~ driver",
    )
    stacker = _RequirementsModel(
        label="stacker",
        formula="target_a ~ component_target_b + component_target_a",
    )
    tree = ForecastTree(
        TreeNode(transform=stacker, children=[component], name="stack"),
        label="tree",
    )

    requirement = tree.input_requirements(["target_a", "target_b"], ["driver"])[1]

    assert requirement.X == (
        ("component_target_b", "native"),
        ("component_target_a", "native"),
    )
    assert requirement.items("X") == ()


def test_requirements_are_frozen_and_do_not_mutate_input_mapping():
    fallback = {"target": "diff", "driver": "pop"}
    model = _RequirementsModel(label="plain")

    requirement = model.input_requirements(["target"], ["driver"], fallback)[0]

    assert fallback == {"target": "diff", "driver": "pop"}
    with pytest.raises(FrozenInstanceError):
        requirement.explicit = True


def test_model_data_select_uses_common_levels_without_vintage_metric_fallback():
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29", "2024-03-31"]
            ),
            "variable": ["z"] * 4,
            "value": [1.0, 2.0, 3.0, 99.0],
            "vintage_date": pd.to_datetime(
                ["2024-02-01", "2024-03-01", "2024-03-01", "2024-03-01"]
            ),
            "metric": ["levels", "levels", "levels", "logs"],
        }
    )
    requirements = (
        ModelInputRequirements(consumer="logs", y=(("z", "logs"),)),
        ModelInputRequirements(consumer="diff", y=(("z", "diff"),)),
    )

    selected = ModelData.from_archive(outturns).select(requirements).as_of("2024-03-15")

    assert selected.metrics("y") == {"z": "levels"}
    assert selected.to_wide().to_dict()["z"] == {
        pd.Timestamp("2024-01-31"): 2.0,
        pd.Timestamp("2024-02-29"): 3.0,
    }
    assert pd.Timestamp("2024-03-31") not in selected.to_wide().index
