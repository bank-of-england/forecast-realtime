# ma_model.R — only defines fit() and forecast()

fit <- function(y, X, params) {
  window_size <- as.integer(params$window_size)

  y <- y[, setdiff(colnames(y), "date"), drop = FALSE]

  n       <- nrow(y)
  tail_df <- y[max(1, n - window_size + 1):n, , drop = FALSE]

  windown_mean <- sapply(tail_df, mean)

  # Return a model object — the runner saves it to model.rds
  list(windown_mean = windown_mean, col_names = colnames(y))
}

forecast <- function(model, steps, X, y, params) {
  windown_mean <- model$windown_mean
  n_vars         <- length(windown_mean)

  fcst <- matrix(rep(windown_mean, each = steps), nrow = steps, ncol = n_vars)

  out <- as.data.frame(fcst)
  colnames(out) <- model$col_names
  # Return a data.frame — the runner writes it to forecasts.parquet
  out
}
