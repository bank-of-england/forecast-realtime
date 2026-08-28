# runner.jl — shipped with forecast_realtime
# Handles CLI dispatch, parameter reading, data loading, model serialisation,
# and forecast output. The user script only defines:
#   fit(y, X, params)              → returns a model object
#   forecast(model, steps, X, y, params) → returns a DataFrame

using Parquet2, DataFrames, Serialization

function read_params(params_path)
    isfile(params_path) ? Dict(pairs(DataFrame(Parquet2.Dataset(params_path))[1, :])) : Dict{String,Any}()
end

user_script = ARGS[1]
action      = ARGS[2]
cache_dir   = ARGS[3]

# Load y from cache_dir
y = DataFrame(Parquet2.Dataset(joinpath(cache_dir, "y.parquet")))

# Optional regressors — `nothing` when the model was fit without any X.
# At fit time these are the training regressors; at forecast time they are
# the future regressor values (one row per step).
x_path = joinpath(cache_dir, "X.parquet")
X = isfile(x_path) ? DataFrame(Parquet2.Dataset(x_path)) : nothing

# Include the user's script (must define fit and forecast)
include(user_script)

if action == "fit"
    params_path = ARGS[4]
    params      = read_params(params_path)
    model       = fit(y, X, params)
    serialize(joinpath(cache_dir, "model.jls"), model)

elseif action == "forecast"
    steps       = parse(Int, ARGS[4])
    params_path = ARGS[5]
    params      = read_params(params_path)
    model       = deserialize(joinpath(cache_dir, "model.jls"))
    result      = forecast(model, steps, X, y, params)
    Parquet2.writefile(joinpath(cache_dir, "forecasts.parquet"), DataFrame(result))

else
    error("Unknown action: $action")
end
