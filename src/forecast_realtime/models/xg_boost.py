"""XGBoost regression model for time series forecasting."""

from xgboost import XGBRegressor

from forecast_realtime.tree_regression import TreeRegression


class XGBoost(TreeRegression):
    """XGBoost forecaster with optional y and X lags appended to X.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default 100.
    max_depth : int
        Maximum tree depth. Default 6.
    learning_rate : float
        Step size shrinkage. Default 0.1.
    subsample : float
        Row subsampling ratio per boosting round. Default 1.0.
    colsample_bytree : float
        Column subsampling ratio per tree. Default 1.0.
    random_state : int
        Random seed. Default 42.
    standardise : bool
        Standardise features and target before fitting. Default False.
    forecast_strategy : str
        Forecasting strategy ("recursive" or "direct"). Default "recursive".
    steps : int | None
        For direct forecasting, the horizon to fit.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional patsy-style formula selecting the regressors.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        standardise: bool = False,
        forecast_strategy: str = "recursive",
        steps: int | None = None,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
    ):
        super().__init__(
            forecast_strategy=forecast_strategy,
            steps=steps,
            standardise=standardise,
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def _build_estimator(self):
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )
