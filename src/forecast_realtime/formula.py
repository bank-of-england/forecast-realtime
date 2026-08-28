import pandas as pd


class Formula:
    """Parse and apply R-style formulas for selective variable usage.

    Supports:
    - Basic: "y ~ x1 + x2" (select y and specific X columns)
    - Wildcard: "y ~ ." (y with all X columns)
    - Future: interactions, polynomial terms, transformations
    """

    def __init__(self, formula_str: str):
        """
        Parse R-style formula string.

        Args:
            formula_str : str
                Formula like "y ~ x1 + x2" or "y ~ ."

        Raises:
            ValueError: If formula is malformed
        """
        if not formula_str or "~" not in formula_str:
            raise ValueError(
                f"Formula must contain '~': got '{formula_str}'. "
                "Expected format: 'y ~ x1 + x2' or 'y ~ .'"
            )

        parts = formula_str.split("~")
        if len(parts) != 2:
            raise ValueError(f"Formula must have exactly one '~': got '{formula_str}'")

        lhs = parts[0].strip()
        rhs = parts[1].strip()

        self.y_cols = [column.strip() for column in lhs.split("+") if column.strip()]
        if not self.y_cols:
            raise ValueError("Left side of formula (y) cannot be empty")
        self.y_col = self.y_cols[0] if len(self.y_cols) == 1 else lhs
        if not rhs:
            raise ValueError("Right side of formula (X variables) cannot be empty")

        self.has_wildcard = rhs == "."
        if self.has_wildcard:
            self.X_cols = None  # Will be determined at extraction time
        else:
            self.X_cols = [c.strip() for c in rhs.split("+") if c.strip()]
            if not self.X_cols:
                raise ValueError(f"No X variables found in '{rhs}'")

    def extract_y(self, y: pd.DataFrame) -> pd.DataFrame:
        """Select y column(s) from DataFrame.

        Args:
            y : pd.DataFrame
                Input data containing y_col

        Returns:
            pd.DataFrame
                Columns listed on the left side of the formula

        Raises:
            ValueError: If y_col not in y.columns
        """
        if y is None:
            return None

        missing = [column for column in self.y_cols if column not in y.columns]
        if missing:
            if len(self.y_cols) == 1:
                message = (
                    f"Formula y column '{self.y_cols[0]}' not found. "
                    f"Available columns: {list(y.columns)}"
                )
            else:
                message = (
                    f"Formula y columns {missing} not found. "
                    f"Available columns: {list(y.columns)}"
                )
            raise ValueError(message)
        return y[self.y_cols]

    def extract_available_inputs(
        self, y: pd.DataFrame, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Select formula inputs available before design construction.

        Formula terms produced later, such as lags and dummies, are deliberately
        left for :meth:`extract_X` to validate once the full design exists.
        """
        y = self.extract_y(y)
        if X is None or self.has_wildcard:
            return y, X

        available_columns = [column for column in self.X_cols if column in X.columns]
        return y, X[available_columns]

    def extract_X(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        """Select X column(s) from DataFrame, expanding wildcards if needed.

        Args:
            X : pd.DataFrame or None
                Input data containing X columns

        Returns:
            pd.DataFrame or None
                Selected X columns, or None if X is None

        Raises:
            ValueError: If specified X columns not in X or X is None when required
        """
        if X is None:
            if not self.has_wildcard and self.X_cols:
                raise ValueError(
                    f"Formula specifies X variables {self.X_cols} but X is None"
                )
            return None

        if self.has_wildcard:
            # Use all columns in X
            return X
        else:
            # Check that all specified columns exist
            missing = set(self.X_cols) - set(X.columns)
            if missing:
                raise ValueError(
                    f"Formula X columns {missing} not found in X. "
                    f"Available columns: {list(X.columns)}"
                )

            return X[self.X_cols]
