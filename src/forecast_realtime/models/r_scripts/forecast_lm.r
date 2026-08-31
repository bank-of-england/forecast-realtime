# forecast_lm.r
# Linear regression on a time trend plus any regressors, using base R's lm().
# Toy example for demonstrating and testing the R integration in forecast-realtime.
# Not intended for production use.
#
# Only defines fit() and forecast() — the runner handles the rest.

library(stats)

# ---- FIT -------------------------------------------------------------------
# Returns a model object (the runner saves it to model.rds).
# `X` is a data.frame of regressors (NULL when there are none). The target is
# the first (and only) column of `y`.
fit <- function(y, X, params) {
  y <- y[, setdiff(colnames(y), "date"), drop = FALSE]
  if (!is.null(X)) {
    X <- X[, setdiff(colnames(X), "date"), drop = FALSE]
  }
  target <- colnames(y)[1]
  n_obs  <- nrow(y)

  # Always include a linear time trend; append any supplied regressors.
  training_data <- data.frame(
    time_index = seq_len(n_obs),
    value      = as.numeric(y[[1]])
  )

  regressors <- character(0)
  if (!is.null(X) && ncol(X) > 0) {
    training_data <- cbind(training_data, X)
    regressors    <- colnames(X)
  }

  # value ~ time_index + `regressor 1` + `regressor 2` + ...
  predictors <- c("time_index", sprintf("`%s`", regressors))
  model_formula <- as.formula(paste("value ~", paste(predictors, collapse = " + ")))
  # OLS via QR (numerically stable); lm_model$coefficients holds the named
  # coefficient vector solving value ~ time_index + regressors, equivalent to
  # coefficients = (X'X)^-1 X'Y. The whole lm_model object (not just the
  # coefficients) is stored in estimated_model$model below and saved to
  # model.rds by the runner, then reused directly by predict() in forecast().
  #
  # na.action = na.exclude keeps fitted()/residuals() the same length as the
  # input data (padding dropped rows with NA) without affecting the
  # estimated coefficients.
  lm_model      <- lm(model_formula, data = training_data, na.action = na.exclude)

  estimated_model <- list(
    model      = lm_model,
    n_obs      = n_obs,
    col_names  = colnames(y),
    regressors = regressors
  )
  return(estimated_model)
}

# ---- FORECAST --------------------------------------------------------------
# Takes the saved model object and returns a data.frame of forecasts
# (the runner writes it to forecasts.parquet).
# At forecast time `X` holds the future regressor values (one row per step).
forecast <- function(model, steps, X, y, params) {
  estimated_model <- model
  target          <- estimated_model$col_names[1]
  regressors      <- estimated_model$regressors

  # Continue the time trend into the future.
  newdata <- data.frame(time_index = estimated_model$n_obs + seq_len(steps))

  # Attach the future regressor values the model was trained on.
  if (length(regressors) > 0) {
    if (is.null(X) || nrow(X) < steps) {
      stop("Future regressor values (X) with one row per step are required to forecast.")
    }
    X <- X[, setdiff(colnames(X), "date"), drop = FALSE]
    newdata <- cbind(newdata, tail(X, steps)[, regressors, drop = FALSE])
  }

  preds <- predict(estimated_model$model, newdata = newdata)

  forecast <- data.frame(setNames(list(as.numeric(preds)), target))
  return(forecast)
}

# ---- FITTED VALUES (optional) -----------------------------------------------
# Returns the in-sample fitted values aligned to the input rows, as a
# data.frame with a single `value` column (the runner writes it to
# fitted_values.parquet). Optional — omit this function if not needed.
fitted_values <- function(model, y, X, params) {
  data.frame(value = as.numeric(fitted(model$model)))
}
