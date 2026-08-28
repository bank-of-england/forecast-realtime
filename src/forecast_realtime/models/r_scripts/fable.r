suppressPackageStartupMessages({
  library(arrow)
  library(fable)
  library(fabletools)
  library(tsibble)
})

RESERVED_COLUMNS <- c("date", "value", "index")

get_param <- function(params, name, default) {
  value <- params[[name]]
  if (is.null(value) || length(value) == 0 || is.na(value[[1]])) {
    return(default)
  }
  value[[1]]
}

infer_index_type <- function(dates) {
  dates <- sort(unique(as.Date(dates)))
  quarter_index <- yearquarter(dates)
  if (
    length(dates) >= 2 &&
    length(unique(quarter_index)) == length(dates) &&
    all(diff(as.numeric(quarter_index)) == 1)
  ) {
    return("quarter")
  }

  month_index <- yearmonth(dates)
  if (
    length(dates) >= 2 &&
    length(unique(month_index)) == length(dates) &&
    all(diff(as.numeric(month_index)) == 1)
  ) {
    return("month")
  }

  "date"
}

make_index <- function(dates, index_type) {
  dates <- as.Date(dates)
  if (index_type == "quarter") {
    return(yearquarter(dates))
  }
  if (index_type == "month") {
    return(yearmonth(dates))
  }
  if (index_type == "date") {
    return(dates)
  }
  stop(sprintf("Unknown fable index type: %s", index_type))
}

validate_regressor_names <- function(column_names, context) {
  overlapping <- intersect(column_names, RESERVED_COLUMNS)
  if (length(overlapping) > 0) {
    stop(
      sprintf(
        "%s regressor names must not use reserved columns: %s",
        context,
        paste(overlapping, collapse = ", ")
      )
    )
  }
}

prepare_data <- function(y, X, params) {
  y_columns <- setdiff(names(y), "date")
  if (length(y_columns) != 1) {
    stop("RFableModel currently supports exactly one target column")
  }

  dates <- as.Date(y$date)
  training <- data.frame(
    date = dates,
    value = as.numeric(y[[y_columns[[1]]]])
  )

  x_columns <- character(0)
  if (!is.null(X)) {
    x_columns <- setdiff(names(X), "date")
    validate_regressor_names(x_columns, "Training")
    if (length(x_columns) > 0 && !isTRUE(get_param(params, "allow_xreg", FALSE))) {
      stop("This fable model does not accept regressors")
    }
    if (any(x_columns %in% names(training))) {
      stop("Regressor names must not overlap with the target column")
    }
    if (length(x_columns) > 0) {
      x_data <- data.frame(date = as.Date(X$date), X[x_columns])
      training <- merge(training, x_data, by = "date", all.x = TRUE, sort = TRUE)
    }
  }

  index_type <- as.character(get_param(params, "index", "auto"))
  if (index_type == "auto") {
    index_type <- infer_index_type(training$date)
  }
  if (!index_type %in% c("quarter", "month", "date")) {
    stop(sprintf("Unknown fable index type: %s", index_type))
  }

  training$index <- make_index(training$date, index_type)
  training$date <- NULL
  training <- training[, c("index", setdiff(names(training), "index")), drop = FALSE]

  list(
    data = as_tsibble(training, index = index),
    index_type = index_type,
    last_date = max(dates),
    x_columns = x_columns
  )
}

fit <- function(y, X, params) {
  prepared <- prepare_data(y, X, params)
  spec_text <- as.character(get_param(params, "spec", ""))
  if (!nzchar(spec_text)) {
    stop("The fable model spec cannot be empty")
  }

  # `spec` is an explicitly trusted R expression supplied by the caller.
  model_spec <- eval(parse(text = spec_text), envir = globalenv())
  fitted <- fabletools::model(prepared$data, fable_model = model_spec)
  if (is.null(fitted[[1]][[1]])) {
    stop("fable returned a null model")
  }

  list(
    model = fitted,
    index_type = prepared$index_type,
    last_date = prepared$last_date,
    x_columns = prepared$x_columns
  )
}

forecast <- function(model, steps, X, y, params) {
  new_data <- NULL
  if (!is.null(X)) {
    x_columns <- setdiff(names(X), "date")
    if (length(x_columns) > 0) {
      validate_regressor_names(x_columns, "Forecast")
      if (!identical(sort(x_columns), sort(model$x_columns))) {
        stop("Forecast regressors must match the fitted regressor columns")
      }
      x_dates <- as.Date(X$date)
      future_dates <- sort(unique(x_dates[x_dates > model$last_date]))
      if (length(future_dates) < steps) {
        stop("Future regressor values must include one row per forecast step")
      }
      future_dates <- future_dates[seq_len(steps)]
      future_rows <- X[match(future_dates, x_dates), x_columns, drop = FALSE]
      future_data <- data.frame(
        index = make_index(future_dates, model$index_type),
        future_rows
      )
      new_data <- as_tsibble(future_data, index = index)
    }
  }

  if (is.null(new_data)) {
    forecasts <- fabletools::forecast(model$model, h = steps)
  } else {
    forecasts <- fabletools::forecast(model$model, new_data = new_data)
  }
  forecast_data <- as.data.frame(forecasts)
  if (!".mean" %in% names(forecast_data)) {
    stop("fable forecast output does not contain a .mean column")
  }

  means <- forecast_data[[".mean"]]
  if (!is.numeric(means) || length(means) != steps) {
    stop("RFableModel currently expects a univariate numeric fable forecast")
  }
  data.frame(value = as.numeric(means))
}
