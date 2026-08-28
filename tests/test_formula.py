import pandas as pd
import pytest

import forecast_realtime as rt
from forecast_realtime.formula import Formula


class TestFormulaInit:
    """Test formula parsing and initialisation."""

    def test_valid_basic_formula(self):
        f = Formula("y ~ x1 + x2")
        assert f.y_col == "y"
        assert f.y_cols == ["y"]
        assert f.X_cols == ["x1", "x2"]
        assert not f.has_wildcard

    def test_valid_multivariate_formula(self):
        f = Formula("y1 + y2 ~ x1")
        assert f.y_cols == ["y1", "y2"]
        assert f.X_cols == ["x1"]
        assert not f.has_wildcard

    def test_valid_wildcard_formula(self):
        f = Formula("target ~ .")
        assert f.y_col == "target"
        assert f.has_wildcard
        assert f.X_cols is None

    def test_whitespace_handling(self):
        f = Formula("  y  ~  x1  +  x2  ")
        assert f.y_col == "y"
        assert f.X_cols == ["x1", "x2"]

    def test_missing_tilde(self):
        with pytest.raises(ValueError, match="must contain '~'"):
            Formula("y x1 x2")

    def test_multiple_tildes(self):
        with pytest.raises(ValueError, match="exactly one '~'"):
            Formula("y ~ x1 ~ x2")

    def test_empty_y(self):
        with pytest.raises(ValueError, match="Left side.*cannot be empty"):
            Formula("~ x1 + x2")

    def test_empty_X(self):
        with pytest.raises(ValueError, match="Right side.*cannot be empty"):
            Formula("y ~")

    def test_single_X_variable(self):
        f = Formula("y ~ x1")
        assert f.X_cols == ["x1"]


class TestExtractY:
    """Test y extraction."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {"target": [1, 2, 3], "feature1": [10, 20, 30], "feature2": [100, 200, 300]}
        )

    def test_extract_existing_column(self, df):
        f = Formula("target ~ .")
        result = f.extract_y(df)
        assert list(result.columns) == ["target"]
        assert len(result) == 3

    def test_extract_multiple_columns(self, df):
        f = Formula("target + feature1 ~ .")
        result = f.extract_y(df)
        assert list(result.columns) == ["target", "feature1"]
        assert len(result) == 3

    def test_extract_y_column_not_found(self, df):
        f = Formula("missing ~ .")
        with pytest.raises(ValueError, match="'missing' not found"):
            f.extract_y(df)


class TestExtractX:
    """Test X extraction."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "y": [1, 2, 3],
                "x1": [10, 20, 30],
                "x2": [100, 200, 300],
                "x3": [1000, 2000, 3000],
            }
        )

    def test_extract_specific_columns(self, df):
        f = Formula("y ~ x1 + x2")
        X = df[["x1", "x2", "x3"]]
        result = f.extract_X(X)
        assert list(result.columns) == ["x1", "x2"]
        assert len(result) == 3

    def test_extract_wildcard(self, df):
        f = Formula("y ~ .")
        X = df[["x1", "x2", "x3"]]
        result = f.extract_X(X)
        assert list(result.columns) == ["x1", "x2", "x3"]

    def test_extract_X_column_not_found(self, df):
        f = Formula("y ~ x1 + missing")
        X = df[["x1", "x2"]]
        with pytest.raises(ValueError, match="not found in X"):  # Changed regex
            f.extract_X(X)

    def test_extract_X_none_with_wildcard(self):
        f = Formula("y ~ .")
        result = f.extract_X(None)
        assert result is None

    def test_extract_X_none_with_specified_columns(self):
        f = Formula("y ~ x1 + x2")
        with pytest.raises(ValueError, match="X is None"):
            f.extract_X(None)

    def test_extract_single_X_column(self, df):
        f = Formula("y ~ x1")
        X = df[["x1", "x2"]]
        result = f.extract_X(X)
        assert list(result.columns) == ["x1"]


class TestIntegration:
    """Integration tests: parse, extract y, extract X."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "gdpkp": [100, 110, 120],
                "unemployment": [4.5, 4.0, 3.8],
                "inflation": [2.0, 2.5, 2.3],
                "other": [1, 2, 3],
            }
        )

    def test_full_pipeline(self, df):
        f = Formula("gdpkp ~ unemployment + inflation")
        y = f.extract_y(df)
        X = f.extract_X(df[["unemployment", "inflation", "other"]])

        assert list(y.columns) == ["gdpkp"]
        assert list(X.columns) == ["unemployment", "inflation"]
        assert y.shape == (3, 1)
        assert X.shape == (3, 2)

    def test_wildcard_pipeline(self, df):
        f = Formula("gdpkp ~ .")
        y = f.extract_y(df)
        X = f.extract_X(df[["unemployment", "inflation", "other"]])

        assert list(y.columns) == ["gdpkp"]
        assert len(X.columns) == 3


class TestFormulaWithForecastModel:
    """Test formula integration with ForecastModel subclasses."""

    @pytest.fixture
    def forecast_data(self):
        """Create sample data with DatetimeIndex."""
        import numpy as np

        dates = pd.date_range("2020-01-01", periods=20, freq="MS")
        return pd.DataFrame(
            {
                "target": np.arange(20, dtype=float),
                "feature1": np.arange(100, 120, dtype=float),
                "feature2": np.arange(200, 220, dtype=float),
                "feature3": np.arange(300, 320, dtype=float),
            },
            index=dates,
        )

    def test_model_init_with_formula(self):
        """Test ForecastModel accepts formula parameter."""

        model = rt.models.ForecastOLS(formula="target ~ feature1 + feature2")
        assert model._formula is not None
        assert model._formula.y_col == "target"
        assert model._formula.X_cols == ["feature1", "feature2"]

    def test_model_fit_applies_formula(self, forecast_data):
        """Test that fit() applies formula to filter y and X."""

        y = forecast_data[["target"]]
        X = forecast_data[["feature1", "feature2", "feature3"]]

        # Model with formula selects only feature1 + feature2
        model = rt.models.ForecastOLS(formula="target ~ feature1 + feature2")
        model.fit(y, X)

        # Check that only selected columns are stored
        assert list(model.X.columns) == ["feature1", "feature2"]

    def test_model_forecast_with_formula(self, forecast_data):

        y = forecast_data[["target"]]
        X = forecast_data[["feature1", "feature2", "feature3"]]

        model = rt.models.ForecastOLS(formula="target ~ feature1")
        model.fit(y, X)

        # Create future X (2 periods beyond training data)
        future_dates = pd.date_range(
            forecast_data.index[-1] + pd.DateOffset(months=1), periods=2, freq="MS"
        )
        X_future = pd.DataFrame(
            {
                "feature1": [120.0, 121.0],
                "feature2": [220.0, 221.0],
                "feature3": [320.0, 321.0],
            },
            index=future_dates,
        )

        # Pass full X - formula will filter to feature1 only
        forecast = model.forecast(steps=2, X=X_future)
        assert forecast.shape == (2, 1)
        assert list(forecast.columns) == ["target"]

    def test_model_without_formula_uses_all_columns(self, forecast_data):
        """Test that models without formula use all provided columns."""

        y = forecast_data[["target"]]
        X = forecast_data[["feature1", "feature2", "feature3"]]

        model = rt.models.ForecastOLS()  # No formula
        model.fit(y, X)

        # All X columns should be preserved
        assert list(model.X.columns) == ["feature1", "feature2", "feature3"]

    def test_model_formula_with_wildcard(self, forecast_data):
        """Test wildcard formula uses all X columns."""

        y = forecast_data[["target"]]
        X = forecast_data[["feature1", "feature2", "feature3"]]

        model = rt.models.ForecastOLS(formula="target ~ .")
        model.fit(y, X)

        assert list(model.X.columns) == ["feature1", "feature2", "feature3"]

    def test_model_formula_missing_column_raises(self, forecast_data):
        """Test that formula with missing column raises ValueError."""

        y = forecast_data[["target"]]
        X = forecast_data[["feature1", "feature2"]]

        model = rt.models.ForecastOLS(formula="target ~ feature1 + missing")
        with pytest.raises(ValueError, match="not found"):  # Changed regex
            model.fit(y, X)

    def test_model_formula_y_lags(self, forecast_data):
        """Test wildcard formula uses all X columns."""

        y = forecast_data[["target"]].copy()
        X = forecast_data.copy()
        X.rename(columns={"target": "target_lag1"}, inplace=True)
        model = rt.models.ForecastOLS(formula="target ~ feature1 + target_lag1")
        model.fit(y, X)

        assert ["feature1", "target_lag1"] == list(model.X.columns)
