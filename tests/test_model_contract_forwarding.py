import pytest

from forecast_realtime.external_model import ExternalModel
from forecast_realtime.linear_regression import LinearRegression
from forecast_realtime.tree_regression import TreeRegression


class _LinearStub(LinearRegression):
    def _fit_reg(self, y, X):
        raise NotImplementedError


class _TreeStub(TreeRegression):
    def _build_estimator(self):
        raise NotImplementedError


class _ExternalStub(ExternalModel):
    def _fit_command(self):
        return []

    def _forecast_command(self, steps):
        return []


def test_common_contract_arguments_forward_through_base_models():
    models = [
        _LinearStub(
            label="linear",
            formula="target ~ x",
            data_transformation={"target": "diff", "x": "levels"},
        ),
        _TreeStub(
            label="tree",
            data_transformation={"target": "levels"},
        ),
    ]

    for model in models:
        assert model.data_transformation is not None

    external = _ExternalStub(
        "model.R",
        label="external",
        data_transformation={"target": "levels"},
        custom_parameter=1,
    )
    assert external.data_transformation == {"target": "levels"}
    assert external.params == {"custom_parameter": 1}
    assert "data_transformation" not in external.params
    assert "data_transformation" not in external.params


def test_external_common_contract_rejects_invalid_data_transformation():
    with pytest.raises(TypeError, match="data_transformation"):
        _ExternalStub("model.R", data_transformation={"target": 1})
