"""Random Forest model for time series forecasting."""

from sklearn.ensemble import RandomForestRegressor

from forecast_realtime.tree_regression import TreeRegression


class RandomForest(TreeRegression):
    """Random Forest forecaster with optional y and X lags appended to X.

    Parameters
    ----------
    n_estimators : int
        Number of trees. Default 100.
    max_depth : int | None
        Maximum tree depth. Default None.
    min_samples_leaf : int
        Minimum samples required at a leaf node. Default 1.
    max_features : str | int | float | None
        Number of features to consider at each split. Default 1.0 (all).
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
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = 1.0,
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
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _build_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
