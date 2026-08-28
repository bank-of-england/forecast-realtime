# runner.r - shipped with forecast_realtime
# Handles CLI dispatch, parameter reading, data loading, model serialisation,
# and forecast output. The user script only defines:
#   fit(y, X, params)              → returns a model object
#   forecast(model, steps, X, y, params) → returns a data.frame

library(arrow)

read_params <- function(params_path) {
  if (file.exists(params_path)) {
    df <- read_parquet(params_path)
    setNames(as.list(df[1, ]), names(df))
  } else {
    list()
  }
}

args        <- commandArgs(trailingOnly = TRUE)
user_script <- args[1]
action      <- args[2]
cache_dir   <- args[3]

# Load y from cache_dir
y <- read_parquet(file.path(cache_dir, "y.parquet"))

# Optional regressors — NULL when the model was fit without any X.
# At fit time these are the training regressors; at forecast time they are
# the future regressor values (one row per step).
x_path <- file.path(cache_dir, "X.parquet")
X      <- if (file.exists(x_path)) read_parquet(x_path) else NULL

# Source the user's script (must define fit and forecast)
source(user_script)

if (action == "fit") {
  params_path <- args[4]
  params      <- read_params(params_path)
  model       <- fit(y, X, params)
  saveRDS(model, file.path(cache_dir, "model.rds"))

  # Optional: if the user script defines fitted_values(), write the
  # in-sample fitted values so Python can populate fitted_values_.
  if (exists("fitted_values", mode = "function")) {
    fitted_df <- fitted_values(model, y, X, params)
    write_parquet(
      as.data.frame(fitted_df),
      file.path(cache_dir, "fitted_values.parquet")
    )
  }

} else if (action == "forecast") {
  steps       <- as.integer(args[4])
  params_path <- args[5]
  params      <- read_params(params_path)
  model       <- readRDS(file.path(cache_dir, "model.rds"))
  result      <- forecast(model, steps, X, y, params)
  write_parquet(as.data.frame(result), file.path(cache_dir, "forecasts.parquet"))

} else {
  stop(paste("Unknown action:", action))
}
