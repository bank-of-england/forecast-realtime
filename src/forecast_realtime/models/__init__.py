import sys
from importlib import import_module
from types import ModuleType

_MODEL_SPECS = {
    "ForecastRidge": ("ridge", ("sklearn",), "ridge"),
    "ForecastLasso": ("lasso", ("sklearn",), "lasso"),
    "RandomForest": ("random_forest", ("sklearn",), "random_forest"),
    "ForecastBVAR": ("forecast_bvar", ("bvar",), "bvar"),
    "ForecastElasticNet": ("elastic_net", ("sklearn",), "elasticnet"),
    "XGBoost": ("xg_boost", ("xgboost",), "xgboost"),
    "ForecastRlm": ("r_lm", (), None),
    "ForecastOLS": ("ols", (), None),
    "ForecastBridgeOLS": ("bridge_ols", (), None),
    "RFableModel": ("fable", (), None),
    "RFableETS": ("fable", (), None),
    "RFableARIMA": ("fable", (), None),
    "ForecastMIDAS": ("midas", ("nowcast_midas",), "nowcast_midas"),
    "ForecastMIDASCombo": ("midas_combo", ("nowcast_midas",), "nowcast_midas"),
    "ForecastMultiMIDAS": ("multi_midas", ("nowcast_midas",), "nowcast_midas"),
}


def _optional_model(name):
    module_name, dependencies, extra = _MODEL_SPECS[name]
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        if not any(
            missing_name == dependency or missing_name.startswith(f"{dependency}.")
            for dependency in dependencies
        ):
            raise
        extra_hint = f".[{extra}]" if extra else ""
        raise ModuleNotFoundError(
            f"Optional model {name!r} requires the missing dependency "
            f"{missing_name!r}. Install the project extra with "
            f"pip install -e {extra_hint}.",
            name=missing_name,
        ) from error
    return getattr(module, name)


def __getattr__(name):
    if name not in _MODEL_SPECS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = _optional_model(name)
    globals()[name] = model
    return model


__all__ = [
    "ForecastRidge",
    "ForecastLasso",
    "ForecastElasticNet",
    "ForecastBVAR",
    "RandomForest",
    "XGBoost",
    "ForecastRlm",
    "RFableModel",
    "RFableETS",
    "RFableARIMA",
    "ForecastMIDAS",
    "ForecastMIDASCombo",
    "ForecastMultiMIDAS",
    "ForecastOLS",
    "ForecastBridgeOLS",
]


class _LazyModelsModule(ModuleType):
    """Resolve registered model attributes to their classes."""

    def __getattribute__(self, name):
        value = ModuleType.__getattribute__(self, name)
        if name in _MODEL_SPECS and (
            isinstance(value, ModuleType)
            or getattr(value, "__module__", None) != f"{__name__}.{_MODEL_SPECS[name][0]}"
        ):
            value = _optional_model(name)
            ModuleType.__setattr__(self, name, value)
        return value


sys.modules[__name__].__class__ = _LazyModelsModule
