from importlib.metadata import version

from .external_model import ExternalModel, JuliaModel, MATLABModel, RModel
from .forecast_model import ForecastContext, ForecastModel, ForecastResult
from .forecast_tree import ForecastTree, TreeNode
from .formula import Formula
from .real_time_model import RealTimeModel
from .sample_realtime_data import generate_synthetic_data

__version__ = version("forecast_realtime")

# Optional import - only load models if available
try:
    from . import models
except ImportError:
    models = None

__all__ = [
    "RealTimeModel",
    "ForecastModel",
    "ForecastContext",
    "ForecastResult",
    "ForecastTree",
    "TreeNode",
    "ExternalModel",
    "Formula",
    "RModel",
    "MATLABModel",
    "JuliaModel",
    "generate_synthetic_data",
    "models",
    "__version__",
]
